
import os
import io
import requests
from PIL import Image
import pandas as pd
import spacy
import cv2
import numpy as np
import csv
from nltk.tokenize import RegexpTokenizer
import pickle

#import keras
#from keras.applications.vgg16 import VGG16, preprocess_input

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import scipy.io
from progressbar import ProgressBar
pbar = ProgressBar()

class VideoEmbeds():
  def __init__(self):
    #model = VGG16(weights='imagenet', include_top = True, input_shape=(224,224,3))
    #self.m = keras.models.Model(inputs=model.input, outputs=[model.layers[-3].output])
    self.hook_output = [] 
    self.vggnet = models.vgg16(pretrained = True)
    self.vggnet.classifier[3].register_forward_hook(self.layer4_hook)
    self.vggnet.eval()
    self.vggnet.cuda()
  
  def CountFramesAndDiff(self, video_path):
    video = cv2.VideoCapture(video_path)
    frame_num = 0
    while True:
      try:
        okay, frame = video.read()
        if not okay:
          break
        frame_num += 1
      except KeyboardInterrupt:
        break
    diff = int(frame_num / 100)
    return frame_num, diff

  def convertGIFtoFrames(self, video_path, diff):
    frames = []
    pil_image = PIL.Image.open(video_path)
    for frame in ImageSequence.Iterator(pil_image):
        temp_img = frame.convert("RGB")
        frame = np.array(temp_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
    return frames
  
  def ConvertVideoToFrames(self, video_path, diff):
    video = cv2.VideoCapture(video_path)
    frame_num = 0
    frame_list = []
    while True:
        try:
            okay, frame = video.read()
            if not okay:
                break
            if frame_num % (diff + 1) == 0:
              frame_list.append(frame)
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break
    return frame_list

  def layer4_hook(self, module, inp, out):
    self.hook_output.append(out)

  def preprocess_img_keras(self, video_name):
    video = cv2.VideoCapture(video_name)
    frame_list = []
    frame_num = 0
    while True:
      try:
        okay, frame = video.read()
        if not okay:
          break
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(frame)
        frame_list.append(frame.reshape(1, 224, 224, 3))
        frame_num += 1
      except KeyboardInterrupt:  # press ^C to quit
        break
    feat_vec = np.concatenate(frame_list)   
    # keras sample (num_samples, 224,224,3)
    feat_vec = self.m.predict(feat_vec).reshape(1, -1, 4096)
    """
    #concat the frames
    dump = np.empty((1, len(frame_list), 4096))
    for i in range(len(frame_list)):
      dump[0,i,] = frame_list[i]
    """
    return feat_vec

  def preprocess_img(self, video_name):
    #torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    frame_num, diff = self.CountFramesAndDiff(video_name)
    frames = self.ConvertVideoToFrames(video_name, diff) #numpy array
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = transforms.Compose([transforms.ToPILImage(), 
                                          transforms.Resize((224,224)), 
                                          transforms.ToTensor()])
    timeDepth = len(frames)
    if(timeDepth > 100):
        model = "cpu"
    else:
        model = "cuda"
    t_frames = torch.Tensor(timeDepth, 3, 224, 224)
    t_frames = Variable(t_frames)
    for f in range(timeDepth):
        if frames[f] is None:
            continue
        frame = torch.from_numpy(frames[f])
        #tensor from numpy
        frame = resize(frame)
        frame = normalize(frame)#transform
        t_frames[f] = frame #frame at fth 
    t_frames = t_frames.unsqueeze(1)
    vid_embeds = torch.FloatTensor(timeDepth,4096)
    #vid_embeds = Variable(vid_embeds)
    for x in range(timeDepth):
        temp_cuda = Variable(t_frames[x])
        if model == "cuda":
            self.vggnet.cuda()
            temp_cuda = temp_cuda.cuda()
            _ = self.vggnet(temp_cuda)
        if model == "cpu":
            self.vggnet.cpu()
            _ = self.vggnet(temp_cuda)
        vid_embeds[x] = self.hook_output.pop()
        del temp_cuda
    var = vid_embeds.detach().numpy()
    return var

class TextPreProcess():
  def __init__(self):
    """Initialise the spacy NLP Corpus"""
    self.nlp = spacy.load('en_core_web_sm')

  def tokenize(self, sentence):
    """Return tokens of the words and symbols in the sentence"""
    token_list = []
    doc = self.nlp(sentence)
    for token in doc:
        token_list.append(token.text)
    return token_list


class AnswerEmbeds():
  def __init__(self):
    fp = open('./vocabulary', 'rb')
    self.vocab = list(pickle.load(fp))
    self.nltk_tokenizer = RegexpTokenizer(r'\w+')
    
  def find_index(self, word):
    return self.vocab.index(word)
  
  def tokenize(self, sentence):
    return self.nltk_tokenizer.tokenize(sentence.lower())
  
  def create_embedding(self, sentence):
    embeds = []
    for x in sentence:
      embeds.append(self.find_index(x))
    return embeds


def createDict(videopath, vqa_dict):
    dataset = {}
    dataset["ques"] = []
    dataset["ans"] = []
    dataset["visual"] = v.preprocess_img(videopath)
    for q in vqa_dict["ques"]:
        ques = t.tokenize(q)
        ques = g.create_embedding(ques)
        dataset["ques"].append(ques.numpy())
    for a in vqa_dict["ans"]:
        ans = amodel.tokenize(a)
        ans.append('<EOS>')
        ans = amodel.create_embedding(ans)
        dataset["ans"].append(np.asarray(ans))    
    return dataset

def wholeDict(csv_reader):
    global_dict = {}
    for row in csv_reader:
        if row[0] in global_dict:
            global_dict[row[0]]["ques"].append(row[1])
            global_dict[row[0]]["ans"].append(row[2])
        else:
            global_dict[row[0]] = {}
            global_dict[row[0]]["ques"] = []
            global_dict[row[0]]["ques"].append(row[1])
            global_dict[row[0]]["ans"] = []
            global_dict[row[0]]["ans"].append(row[2])
    return global_dict


def main(VideoIdFile, VideoFolder):
    count = 0
    csvfile = open('./QA.csv')
    csv_reader = csv.reader(csvfile, delimiter = ',')
    global_dict = wholeDict(csv_reader)
    print(len(global_dict))
    videoids = open(VideoIdFile, "r")    
    allVideoIds = videoids.readlines()
    totalids = len(allVideoIds)
    """for id in pbar(allVideoIds):
      if(os.path.isfile("/media/s3/New Volume/MAT_Files/" + id + ".mat")):
          print("Skipped")
          continue
      file = open('./progress.txt', 'w+')
      id = id[:-1]
      try:
        scipy.io.savemat("/media/s3/New Volume/Practice/" + id + ".mat", createDict(VideoFolder+id+".gif", global_dict[id]))
      except KeyError:
        string2 = "No QA " + str(id)
        file.write(string2)
        continue #If QA Pair not found for video skip
      string1 = "Done " + str(id) + "/" + str(totalids) + "\n"
      file.write(string1)
      file.close()"""
    scipy.io.savemat("/media/s3/New Volume/Practice/10022.mat", createDict(VideoFolder+"10022"+".gif", global_dict["10022"]))
   

class GloVE():
  """Class that generates a GloVE Embedding for any word provided"""
  def __init__(self):
    """Initialise the embedding and model from txt file"""
    #glove_file = datapath(os.path.abspath('./glove.840B.300d.txt'))
    #tmp_file = get_tmpfile(os.path.abspath("./glove_word2vec.txt"))
    #_ = glove2word2vec(glove_file, tmp_file)
    self.model = gensim.models.KeyedVectors.load_word2vec_format(os.path.abspath("./glove/glove_word2vec.txt"))
    self.weights = torch.FloatTensor(self.model.vectors)
    self.embedding = nn.Embedding.from_pretrained(self.weights)

  def create_embedding(self, question):
    """Create a list of embeddings from the token provided"""
    e_len = len(question)
    embeds = torch.Tensor(e_len,300)
    for x in range(e_len):
      try:
        input = torch.LongTensor([self.model.vocab[question[x]].index])
        embeds[x] = self.embedding(input)
      except KeyError:
        input = torch.LongTensor([self.model.vocab["unk"].index])
        embeds[x] = self.embedding(input)
    embeds = embeds.unsqueeze(1) #Make it LSTM Compatible. Tensor is now (Seq Size, Batch Size, Input Size)
    return embeds
"""
v = VideoEmbeds()    
g = GloVE()
amodel = AnswerEmbeds()
t = TextPreProcess()
VideoIdFile = "./videoid.txt"
VideoFolder = "./GIFs/"
main(VideoIdFile, VideoFolder)
files = os.listdir(VideoFolder)
count = 0
for x in files:
  if(x[-4:] == ".gif"):
#      a = v.preprocess_img('./GIFs/' + x)
      a = v.ConvertVideoToFrames('./GIFs/' + x)
      if(len(a) >= 100):
        count += 1
print(count)
      os.system('nvidia-smi ')
"""
