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
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io as sio
import copy
import time
import gc
import sys
from tqdm import tqdm
import datetime


if torch.cuda.is_available():
    device = torch.device('cuda')
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    #torch.set_default_tensor_type('torch.FloatTensor')
print(device)

class PaddingTensors():
    def __init__(self, vocab_length):
        pad_index = 5617 #PAD #Refer Vocabulary_final_SEP in Dataset
        self.pad_vec_i = torch.zeros(vocab_length, dtype=torch.float, device = "cpu")
        self.pad_vec_t = torch.tensor([pad_index], dtype= torch.long, device = "cpu")
        self.pad_vec_i[pad_index] = 1

    def pad_tensors(self, input_cuda, target_cuda):
        global device
        input = input_cuda.cpu()
        target = target_cuda.cpu()
        try:
            target_l = list(target.size())[0]
        except Exception:
            #print("error: " + str(e))
            return None, None
        try:
            input_l = list(input.size())[0]
        except Exception:
            #print("error: " + str(e))
            return None, None
        diff = target_l - input_l
        if diff == 0:
            return [input, target]
        op = "pad_input" if diff > 0 else "pad_target"
        diff = abs(diff)
        if op == "pad_input":
            pad_vecs = torch.stack([self.pad_vec_i for _ in range(diff)], dim = 0)
            input = torch.cat([input, pad_vecs])
        elif op == "pad_target":
            pad_vecs = torch.cat([self.pad_vec_t for _ in range(diff)], dim = 0)
            target = torch.cat([target, pad_vecs])
        return [input, target]

class VideoQADataset(Dataset):
    def __init__(self, vid_path, filepath):
    #MATFile Structure =   Video  =>  [No. of frames, embeddings]
    #                   Question  =>  list(tokens,embeddings)
    #                   Answer    =>  list(tokens,embeddings)
        self.vid_dir = vid_path
        self.samples = self.parse_files(filepath)
        self.num_samples = len(self.samples)
    def __len__(self):
        return  self.num_samples#vid dir format "./embeddings"
    
    def __getitem__(self, idx):
        idx = self.samples[idx]
        
        #idx = 31264
        
        
        file_path = os.path.join(self.vid_dir, str(idx) + ".mat")
        #print("file:", file_path)
        mat = sio.loadmat(file_path)
        video = torch.from_numpy(mat["visual"].astype(np.float32))
        question = []
        for x in mat["ques"][0]:
            if x.shape == (1,300):
                question.append(torch.from_numpy(mat["ques"][0].astype(np.float32)))
                break
            question.append(torch.from_numpy(x))
        question = [x.squeeze(1) for x in question]       
        answer = []
        if type(mat["ans"][0][0]) is np.ndarray:
            for x in mat["ans"][0]:
                answer.append(torch.squeeze(torch.from_numpy(x)))
        else:
            for x in mat["ans"]:
                answer.append(torch.squeeze(torch.from_numpy(x)))
        return [video, question], answer
    
    def parse_files(self, filename):
        f = open(filename, 'r')
        names = f.readlines()
        ids = []
        for x in range(len(names)):
            ids.append(names[x][:-1])
        return ids

class VideoEmbedding(nn.Module):
    
    def __init__(self, embedding_dim, input_dim, bidirectional=True):
        
        super(VideoEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_dim, embedding_dim, num_layers=1, bidirectional=bidirectional)
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        
    def forward(self, inputs, hidden):
        outputs, hidden = self.lstm(inputs.view(len(inputs), 1, -1), hidden)
        return outputs, hidden

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros((1 + int(self.bidirectional), 1, self.embedding_dim), device = device),
                torch.zeros((1 + int(self.bidirectional), 1, self.embedding_dim), device = device))


class WordEmbedding(nn.Module):

    def __init__(self, embedding_dim, input_dim, bidirectional = True):
        
        super(WordEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_dim, embedding_dim, num_layers = 1, bidirectional = bidirectional)
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional

    def forward(self, inputs, hidden):
        outputs, hidden = self.lstm(inputs.view(len(inputs), 1, -1), hidden)
        return outputs, hidden
    
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros((1 + int(self.bidirectional), 1, self.embedding_dim), device=device),
                torch.zeros((1 + int(self.bidirectional), 1, self.embedding_dim), device = device))

        
class AttentionBasedMultiModalFusion(nn.Module):

    def __init__(self, output_size = 300, embedding_img = 300, embedding_q = 300, vocab = 8834,
            input_img = 4096, input_q = 300, max_length_img = 50, max_length_q = 50, max_modalities = 2):

        super(AttentionBasedMultiModalFusion, self).__init__()
        self.output_size = output_size
        self.max_length_img = max_length_img
        self.max_length_q = max_length_q
        self.embedding_img = embedding_img
        self.embedding_q = embedding_q
        self.input_img = input_img
        self.input_q = input_q
        self.max_modalities = max_modalities
        self.vocab_length = vocab

        self.GloveEmbeds = nn.Embedding(self.vocab_length, self.input_q)
        self.load_glove_weights()
        
        self.VideoEmbedding = VideoEmbedding(self.embedding_img, self.input_img)
        self.QuestionEmbedding = WordEmbedding(self.embedding_q, self.input_q)

        self.attn_img = nn.Linear(self.output_size + self.embedding_img * 2, self.embedding_img) #account for bidirectionality
        self.attn_q = nn.Linear(self.output_size + self.embedding_q *2, self.embedding_q) #account for bidirectioality
        self.attn_img_h = nn.Linear(self.embedding_img, 1, bias = False)
        self.attn_q_h = nn.Linear(self.embedding_q, 1, bias = False)
        #self.attn_img_dropout = nn.Dropout(0.2)
        #self.attn_q_dropout = nn.Dropout(0.2)
        
        self.attn_mod = nn.Linear(self.output_size, self.output_size)
        self.attn_mod_img = nn.Linear(self.embedding_img * 2, output_size, bias = False)
        self.attn_mod_q = nn.Linear(self.embedding_q * 2, output_size, bias = False)
        self.attn_mod_h = nn.Linear(self.output_size, 1, bias = False)
        

        self.fusion_img = nn.Linear(self.embedding_img * 2, self.output_size, bias = False)
        self.fusion_q = nn.Linear(self.embedding_q * 2, self.output_size, bias = False)

        self.fusion = nn.Linear(self.output_size, self.output_size)
        self.decoder = nn.LSTM(self.output_size + self.input_q, self.output_size)
        self.final = nn.Linear(self.output_size, self.vocab_length)

    def load_glove_weights(self):
        path = './vocab_glove_embeds_20-03.npy'
        arr = np.load(path)
        self.GloveEmbeds.weight.data.copy_(torch.from_numpy(arr))
        
    def forward(self, inputs, file = None):
        s_input_img = inputs[0].to(device)
        hidden_img = self.VideoEmbedding.init_hidden()
        img_emb, hidden_img = self.VideoEmbedding.forward(s_input_img.view(-1, 1, self.input_img), hidden_img)
        
        final_vec = []
        for t in range(len(inputs[1])):
            #print("NEW QUESTION" * 10)
            hidden_q = self.QuestionEmbedding.init_hidden()
            emb_output = torch.zeros(1, 1, self.input_q, device=device)
            decoder_hidden = self.init_hidden()
            s_input_que = inputs[1][t].to(device)
            q_emb, hidden_q = self.QuestionEmbedding.forward(s_input_que.view(-1, 1, self.input_q), hidden_q) 

            outputs = []
            num_words = 0
            print('*' * 50, file = file) if file is not None else 0
            while (num_words < 17):
              #calculate e(i, t) by pa ssing S(i - 1) and h(t) through a linear layer without bias
              #calculate alpha 
                img_attn_weights = []
                q_attn_weights = []
                for i in range(len(img_emb)):
                    img_attn_weights.append(self.attn_img_h(self.attn_img(torch.cat((decoder_hidden[0][0],
                        img_emb[i].view(1, 1, -1)[0]), 1))))
                for i in range(len(q_emb)):
                    q_attn_weights.append(self.attn_q_h(self.attn_q(torch.cat((decoder_hidden[0][0],
                        q_emb[i].view(1, 1, -1)[0]), 1))))
                #print('*' * 50)
                ##print(torch.cat(img_attn_weights, 1))
                #print(torch.cat(q_attn_weights, 1))
                img_attn_weights = F.softmax(torch.cat(img_attn_weights, 1), dim = 1)
                q_attn_weights = F.softmax(torch.cat(q_attn_weights, 1), dim =1)
                #print(img_attn_weights)
                #print(q_attn_weights)
                #calculate c(i)
                img_attn_applied = torch.bmm(img_attn_weights.unsqueeze(1),
                                               img_emb.view(1, -1, self.embedding_img * 2))
                q_attn_applied = torch.bmm(q_attn_weights.unsqueeze(1),
                                             q_emb.view(1, -1, self.embedding_q * 2))

                 #calculate beta 
                tmp = self.attn_mod(decoder_hidden[0])
                mod_attn_weight_img_unnorm = self.attn_mod_h((tmp + self.attn_mod_img(img_attn_applied)).tanh())
                mod_attn_weight_q_unnorm = self.attn_mod_h((tmp + self.attn_mod_q(q_attn_applied)).tanh())

                mod_attn_weights = F.softmax(torch.cat((mod_attn_weight_img_unnorm, 
                    mod_attn_weight_q_unnorm), dim = 2), dim = 2)

                if file:
                    print(mod_attn_weights, file = file)
                    print(img_attn_weights, file = file)
                    print(q_attn_weights, file = file)
                
                d_img_i = self.fusion_img(img_attn_applied)
                d_q_i = self.fusion_q(q_attn_applied)

                fs_apply = torch.bmm(mod_attn_weights,
                    torch.cat((d_img_i, d_q_i), dim = 1)) #this should be done above, but not sure of dimensions

                fs_output = (self.fusion(decoder_hidden[0]) + fs_apply).tanh()
                output, decoder_hidden = self.decoder(torch.cat((fs_output[0], emb_output[0]), dim = 0).view(1,1, -1), decoder_hidden) #Sameer Addition
                #output, decoder_hidden = self.decoder(emb_output, fs_output)

                outputs.append(self.final(output[0]))
                num_words += 1
                word_index = torch.argmax(outputs[-1])
                if word_index == 6148: #EOS #Refer vocabulary in MAT 11-03
                    #print(num_words)
                    break
                emb_output = self.GloveEmbeds(word_index).view(1,1,-1)
                
            final_vec.append(torch.cat(outputs, dim = 0))
            
        return final_vec

    def init_hidden(self):
        return (torch.zeros((1, 1, self.output_size), device = device), 
                torch.zeros((1, 1, self.output_size), device = device))