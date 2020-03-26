from DatasetCreation import AnswerEmbeds, VideoEmbeds, GloVE, TextPreProcess
from Model import AttentionBasedMultiModalFusion
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import pickle

class Demo_VQA():
    def __init__(self):
        self.g = GloVE()
        self.model = AttentionBasedMultiModalFusion().cuda()
        self.model.load_state_dict(torch.load('./final.model'))
        self.model.train(False)
        fp = open('./vocabulary', 'rb')
        self.vocab = list(pickle.load(fp))
        self.t = TextPreProcess()
        self.v = VideoEmbeds()

    def vqa(self, gif_path, q):
        #gif_path = input("Enter the video path")
        visual = self.v.preprocess_img(gif_path)
        video = torch.from_numpy(visual.astype(np.float32))
        video = video.cuda()
        #q = input("Enter a question")
        ques_tokens = self.t.tokenize(q)
        ques = self.g.create_embedding(ques_tokens)
        ques = ques.numpy()
        question = (torch.from_numpy(ques.astype(np.float32)))
        inputs = [video, question]
        predict = self.model.forward(inputs)
        answer = []
        for (_,words) in enumerate(predict):
            for word in range(len(words)):
                answer.append(self.vocab[torch.argmax(words[word])])
        return answer