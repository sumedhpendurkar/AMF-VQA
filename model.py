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

    def forward(self, inputs, h):

        outputs, _ = self.lstm(inputs)
        return outputs
        
class AttentionBasedMultiModalFusion(nn.Module):

    def __init__(self, output_size, embedding_img, embedding_q, hidden_size, vocab,
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
        self.vocab_length = vocab

        self.VideoEmbedding = VideoEmbedding(self.embedding_img, self.input_img)
        self.QuestionEmbedding = WordEmbedding(self.embedding_q, self.input_q)

        attn_img = nn.Linear(self.output_size + self.embedding_img, self.max_length_img)
        attn_q = nn.Linear(self.output_size + self.embedding_q, self.max_length_q)

        attn_mod = nn.Linear(self.embedding_q+ self.embedding_img, self.max_modalities)
        
        fusion_img = nn.Linear(self.embedding_img, self.hidden_size)
        fusion_q = nn.Linear(self.embedding_q, self.hidden_size)

        fusion = nn.Linear(self.output_size + self.hidden_size, self.output_size)
        self.decoder = nn.LSTM(self.output_size, self.vocab_length)
        #init hidden for decoder

    def forward(self, inputs, hidden):
       
        img_emb = self.VideoEmbedding(inputs[0])
        q_emb = self.QuestionEmbedding(inputs[1])

        img_attn_weights = F.softmax(self.attn_img(torch.cat((img_emb,
            self.prev_output), 1)), dim = 1)
        q_attn_weights = F.softmax(self.attn_q(torch.cat((q_emb,
            self.prev_output), 1)), dim = 1)

        img_attn_applied = torch.bmm(img_attn_weights.unsqueeze(0),
                img_emb.unsqueeze(0))
        q_attn_applied = torch.bmm(q_attn_weights.unsqueeze(0),
                q_emb.unsqueeze(0))

        mod_attn_weights = F.softmax(self.attn_mod_img(torch.cat((img_emb, #concatenate img_emb with q_emb, how that I am not sure of
            self.prev_output), 1)), dim = 1)

        fs_img = fusion_img(self.embedding_img)
        fs_q = fusion_q(self.embedding_q)

        fs_apply = torch.bmm(mod_attn_weights.unsqueeze(0),
                torch.cat((img_emb, q_emb), dim = 1).unsqueeze(0)) #this should be done above, but not sure of dimensions

        fs_output = fusion(torch.cat((self.prev_output, fs_apply), dim = 1))

        output, hidden = self.decoder(fs_output, hidden)
       
        self.prev_output = output
        return output, hidden
