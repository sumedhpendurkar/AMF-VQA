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
import keras.backend as K
import pickle

import TextUtils

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
        return (torch.zeros(1 + int(self.bidirectional), 1, self.embedding_dim),
                torch.zeros(1 + int(self.bidirectional), 1, self.embedding_dim))


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
        return (torch.zeros(1 + int(self.bidirectional), 1, self.embedding_dim),
                torch.zeros(1 + int(self.bidirectional), 1, self.embedding_dim))

        
class AttentionBasedMultiModalFusion(nn.Module):

    def __init__(self, output_size, embedding_img, embedding_q, vocab,
            input_img, input_q, max_length_img, max_length_q, max_modalities, 
            vocab_path, glove_path):

        super(AttentionBasedMultiModalFusion, self).__init__()
        self.output_size = output_size
        self.max_length_img = max_length_img
        self.max_length_q = max_length_q
        self.embedding_img = embedding_img
        self.embedding_q = embedding_q
        self.input_img = input_img
        self.input_q = input_q
        self.max_modalities = max_modalities
        self.vocabulary = self.load_vocabulary(vocab_path)
        self.vocab_length = len(self.vocabulary)
        self.glove_output = None

        self.GloveEmbeds = nn.Embedding(self.vocab_length, self.embedding_q)
        self.load_glove_weights(glove_path) #create matrix for glove embeddings

        self.VideoEmbedding = VideoEmbedding(self.embedding_img, self.input_img)
        self.QuestionEmbedding = WordEmbedding(self.embedding_q, self.input_q)

        self.attn_img = nn.Linear(self.output_size + self.embedding_img * 2, 1) #account for bidirectionality
        self.attn_q = nn.Linear(self.output_size + self.embedding_q *2, 1) #account for bidirectioality

        self.attn_mod = nn.Linear(self.output_size, 1) #self.max_modalities)
        self.attn_mod_img = nn.Linear(self.embedding_img * 2, 1, bias = False) #self.max_modalities)
        self.attn_mod_q = nn.Linear(self.embedding_q * 2, 1, bias = False) #self.max_modalities)

        self.fusion_img = nn.Linear(self.embedding_img * 2, self.output_size, bias = False)
        self.fusion_q = nn.Linear(self.embedding_q * 2, self.output_size, bias = False)

        self.fusion = nn.Linear(self.output_size, self.output_size)
        self.decoder = nn.LSTM(self.output_size, self.output_size)
        self.final = nn.Linear(self.output_size, self.vocab_length)

    def forward(self, inputs):

        hidden_img = self.VideoEmbedding.init_hidden()
        hidden_q = self.QuestionEmbedding.init_hidden()
        self.prev_output = torch.zeros(1, 1, self.output_size)
        decoder_hidden = self.init_hidden()
        img_emb, hidden_img = self.VideoEmbedding.forward(inputs[0], hidden_img)
        q_emb, hidden_q = self.QuestionEmbedding.forward(inputs[1], hidden_q)

        outputs = []
        #TODO: store the results
        while True:#self.prev_output != '<EOS>':
            #calculate e(i, t) by passing S(i - 1) and h(t) through a linear layer without bias
            #calculate alpha 
            print(img_emb)
            print(hidden_img[0])
            print('-'*100)
            img_attn_weights = []
            q_attn_weights = []
            for i in range(len(img_emb)):
                img_attn_weights.append(self.attn_img(torch.cat((decoder_hidden[0][0],
                    img_emb[i].view(1, 1, -1)[0]), 1)))
            for i in range(len(q_emb)):
                q_attn_weights.append(self.attn_q(torch.cat((decoder_hidden[0][0],
                    q_emb[i].view(1, 1, -1)[0]), 1)))
            
            print(len(img_attn_weights))
            img_attn_weights = F.softmax(torch.cat(img_attn_weights, 1), dim = 1)
            q_attn_weights = F.softmax(torch.cat(q_attn_weights, 1), dim =1)
      
            #calculate c(i)
            img_attn_applied = torch.bmm(img_attn_weights.unsqueeze(1),
                                         img_emb.view(1, -1, self.embedding_img * 2))
            q_attn_applied = torch.bmm(q_attn_weights.unsqueeze(1),
                                       q_emb.view(1, -1, self.embedding_q * 2))

            #calculate beta
            tmp = self.attn_mod(decoder_hidden[0])
            mod_attn_weight_img_unnorm = (tmp + self.attn_mod_img(img_attn_applied)).tanh()
            mod_attn_weight_q_unnorm = (tmp + self.attn_mod_q(q_attn_applied)).tanh()
            
            mod_attn_weights = F.softmax(torch.cat((mod_attn_weight_img_unnorm, 
                mod_attn_weight_q_unnorm), dim = 2))

            d_img_i = self.fusion_img(img_attn_applied)
            d_q_i = self.fusion_q(q_attn_applied)

            print(mod_attn_weights.shape)
            print(torch.cat((d_img_i, d_q_i), dim = 1).shape)
            fs_apply = torch.bmm(mod_attn_weights,
                    torch.cat((d_img_i, d_q_i), dim = 1)) #this should be done above, but not sure of dimensions

            print(decoder_hidden[0].shape)
            fs_output = (self.fusion(decoder_hidden[0]) + fs_apply).tanh()

            output, decoder_hidden = self.decoder(fs_output, decoder_hidden)
            final_output = self.final(output[0])
            index = torch.argmax(final_output, dim=0) #find index
            self.glove_output = self.GloveEmbeds(index)
            #self.glove_output is a 300D vector
            outputs.append()
            print("First word outputted successfully!")

        return torch.cat(outputs, dim = 0)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.output_size), 
                torch.zeros(1, 1, self.output_size))

    def load_vocabulary(self, vocab_path):
        fp = open(vocab_path, 'rb')
        return list(pickle.load(fp))
    
    def load_glove_weights(self, glove_path):
        self.GloveEmbeds.weight.data.copy_(torch.from_numpy(glove_path))


def train(model, epochs=10, batch_size = 4, learning_rate = 0.0001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    metric = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for dataitem in (len(data) / batch_size):
      # x_v, x_q, label = unpack dataitem
            optimizer.zero_grad()
            for i in range(batch_size):
                predict = model.forward([x_v, x_q])
        #may want to concatenate the words to apply cross entroppy and sum up the loss
                loss += metric(predict, label)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    v = AttentionBasedMultiModalFusion(4, 2, 2, 6, 4, 4, 5, 5, 2)
    v.forward([torch.randn(6,4), torch.randn(2, 4)])
