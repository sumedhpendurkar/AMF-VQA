import os
import io
import requests
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import VideoUtils
import TextUtils
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_printoptions(threshold=5000)

class GloVE():
  def __init__(self):
  #  glove_file = datapath(os.path.abspath('./glove.840B.300d.txt'))
  #  tmp_file = get_tmpfile(os.path.abspath("./glove_word2vec.txt"))
  #  _ = glove2word2vec(glove_file, tmp_file)
    self.model = gensim.models.KeyedVectors.load_word2vec_format(os.path.abspath("./glove_word2vec.txt"))
    self.weights = torch.FloatTensor(model.vectors)
    self.embedding = nn.Embedding.from_pretrained(self.weights)

  def create_embedding(self, question):
    embeds = []
    for x in question:
      try:
        input = torch.LongTensor([self.model.vocab[x].index])
        embeds.append(self.embedding(input))
      except KeyError:
        input = torch.LongTensor([self.model.vocab["unk"].index])
        embeds.append(self.embedding(input))
    return embeds
 

class FCUniClassifier(nn.Module):
  def __init__(self):
    super(FCUniClassifier, self).__init__()
    self.fc_first = nn.Linear(1024,1000)
    self.dropout_1 = nn.Dropout(p=0.5)
    self.fc_second = nn.Linear(1000,1000)
    self.dropout_2 = nn.Linear(1000,1000)
    self.tanh = nn.Tanh()
  def forward(self, img_que):
    output_1 = self.fc_first(img_que)
    output_2 = self.dropout_1(output_1)
    output_3 = self.fc_second(output_2)
    output_4 = self.dropout_2(output_3)
    output = self.tanh(output_4)
    return output

hook_output = []  
def layer4_hook(module, inp, out):
  hook_output.append(out)

class VQA(nn.Module):
  def __init__(self):
    super(VQA, self).__init__()
    #Image
    self.vggnet = models.vgg16()
    self.vggnet.classifier[3].register_forward_hook(layer4_hook)
    self.fc_image = nn.Linear(4096, 1024, True)
    #Question
    self.glove = GloVE()
    self.hidden = self.init_hidden()
    self.lstm = nn.LSTM(300,512,2)
    self.fc_question = nn.Linear(2048,1024, True)
    #Unified
    self.fc_unified_classifier = FCUniClassifier()
    self.softmax = nn.Softmax()

  def init_hidden(self):
    return (torch.zeros(1, 1, 512), torch.zeros(1, 1, 512))
  
  def forward(self, frames, question, answer):
    frame_softmax_outputs = [] #Store Results for all frames
    que_embed = self.glove.create_embedding(question)
    #PROCESS ALL TOKENS OF WORDS IN LSTM NOT JUST ONE
    output, self.hidden = self.lstm(que_embed, self.hidden)
    fc_que_output = self.fc_question(output)
    for x in frames:
      _ = self.vggnet(x)
      img_embed = hook_output.pop()
      fc_img_output = self.fc_image(img_embed)
      fc_img_output = self.tanh(fc_img_output)      
      elem_mul = torch.mul(fc_img_output, fc_que_output)
      fc_unified = self.fc_unified_classifier(elem_mul)
      softmax = self.softmax(fc_unified)
      frame_softmax_outputs.append(softmax)
    return torch.mean(frame_softmax_outputs)
    #TODO ADD LSTM PROCESSING AND IMAGE COMBINATION   

class VideoQADataset(Dataset):
  def __init__(self, vid_path, data_file, transform):  
    self.video_dir = vid_path
    self.data_file = pd.read_csv(data_file, sep='\t')
    self.channels = 3
    self.xSize = 1280
    self.ySize = 720
  def __getitem__(self, idx):
    vid_name = os.path.join(self.video_dir, self.data_file.iloc[idx, 0])
    frames = VideoUtils.ConvertVideoToFrames(vid_name)
    frames = Frames2Tensor(frames)
        #Apply transforms if necessary to frames
    QA_pair = self.data_file.iloc[idx, 1:].tolist()
    return frames, QA_pair
  
  def Frames2Tensor(self, frames):
    """
    Mohsen Fayyaz __ Sensifai Vision Group
    http://www.Sensifai.com
    Modified version of repository
    https://github.com/MohsenFayyaz89/PyTorch_Video_Dataset
    """
    timeDepth = len(frames)
    t_frames = torch.FloatTensor(self.channels, timeDepth, self.xSize, self.ySize)
    for f in range(timeDepth):
        frame = torch.from_numpy(frames[f])
        frame = frame.permute(2, 1, 0)
        t_frames[:, f, :, :] = frame
    return t_frames

if __name__ == "__main__":
  model = VQA()
  textprep = TextUtils.TextPreProcess()
  #LOAD DATA FROM PICKLE OR SOME SHIT
  #loadData()
  # train()
  # else
  # test() 
  #PREPROCESS QUESTION
  #PREPROCESS IMAGE
  #ADD TRAINING FUNCTION
  #ADD EVAL FUNCTION

