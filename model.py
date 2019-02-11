from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class VideoEmbedding(nn.Module):
    
    def __init__(self, embedding_dim, input_dim, bidirectional=True):
        
        self.lstm = nn.LSTM(input_dim, embedding_dim, num_layers=1, bidirectional=birdirectional)
        
    def forward(self, inputs):

        #maybe reshape it
        outputs, _ = self.lstm(inputs)
        return outputs

class WordEmbedding(nn.Module):

    def __init__(self, embedding_dim, input_dim, bidirectional = True):
        self.lstm = nn.LSTM(input_dim, embedding_dim, num_layers = 1, bidirectional = birdirectional)

    def forward(self, inputs):

        outputs, _ = self.lstm(inputs)
        return outputs
        
class AttentionBasedMultiModalFusion(nn.Module):

    def __init__(self, output_size, embedding_img, embedding_q, hidden_size 
            input_img, input_q, max_length_img, max_length_q, max_modalities):

        self.output_size = output_size
        self.max_length_img = max_length_img
        self.max_length_q = max_length_q
        self.embedding_img = embedding_img
        self.embedding_q = embedding_q
        self.hidden_size = hidden_size
        self.input_img = input_img
        self.input_q = input_q
        self.max_modalities = max_modalities

        self.VideoEmbedding = VideoEmbedding(self.embedding_img, self.input_img)
        self.QuestionEmbedding = WordEmbedding(self.embedding_q, self.input_q)

        attn_img = nn.Linear(self.output_size + self.embedding_img, self.max_length_img)
        attn_q = nn.Linear(self.output_size + self.embedding_q, self.max_length_q)

        attn_mod = nn.Linear(self.embedding_q + self.embedding_img, self.max_modalities)
        
        fusion_img = nn.Linear(self.embedding_img, self.hidden_size)
        fusion_q = nn.Linear(self.embedding_q, self.hidden_size)

        fusion = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden):
        pass
