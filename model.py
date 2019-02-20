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

class VideoEmbedding(nn.Module):
    
    def __init__(self, embedding_dim, input_dim, bidirectional=True):
        
        super(VideoEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_dim, embedding_dim, num_layers=1, bidirectional=bidirectional)
        
    def forward(self, inputs, hidden):

        #maybe reshape it
        outputs, hidden = self.lstm(inputs.reshape(1, 1, -1), hidden)
        return outputs, hidden

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.embedding_dim),
                torch.zeros(1, 1, self.embedding_dim))


class WordEmbedding(nn.Module):

    def __init__(self, embedding_dim, input_dim, bidirectional = True):
        
        super(WordEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_dim, embedding_dim, num_layers = 1, bidirectional = bidirectional)

    def forward(self, inputs, hidden):

        outputs, hidden = self.lstm(inputs.view(1, 1, -1), hidden)
        return outputs, hidden
    
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.embedding_dim),
                torch.zeros(1, 1, self.embedding_dim))

        
class AttentionBasedMultiModalFusion(nn.Module):

    def __init__(self, output_size, embedding_img, embedding_q, vocab,
            input_img, input_q, max_length_img, max_length_q, max_modalities):

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

        self.VideoEmbedding = VideoEmbedding(self.embedding_img, self.input_img)
        self.QuestionEmbedding = WordEmbedding(self.embedding_q, self.input_q)

        attn_img = nn.Linear(self.output_size + self.embedding_img, self.max_length_img)
        attn_q = nn.Linear(self.output_size + self.embedding_q, self.max_length_q)

        attn_mod = nn.Linear(self.output_size, 1) #self.max_modalities)
        attn_mod_img = nn.Linear(self.embedding_img, 1, bias = False) #self.max_modalities)
        attn_mod_q = nn.Linear(self.embedding_q, 1, bias = False) #self.max_modalities)

        fusion_img = nn.Linear(self.embedding_img, self.output_size, bias = False)
        fusion_q = nn.Linear(self.embedding_q, self.output_size, bias = False)

        fusion = nn.Linear(self.output_size, self.output_size)
        self.decoder = nn.LSTM(self.output_size, self.vocab_length)

    def forward(self, inputs):

        self.hidden_img = self.VideoEmbedding.init_hidden()
        self.hidden_q = self.QuestionEmbedding.init_hidden()
        self.prev_output = torch.zeros(1, 1, self.output_size)
        self.decoder_hidden = self.init_hidden()
        img_emb, self.hidden_img = self.VideoEmbedding(inputs[0], self.hidden_img)
        q_emb, self.hidden_q = self.QuestionEmbedding(inputs[1], self.hidden_q)

        #TODO: store the results
        while self.prev_output != '<EOS>':
            #calculate e(i, t) by passing S(i - 1) and h(t) through a linear layer without bias
            #calculate alpha
            img_attn_weights = F.softmax(self.attn_img(torch.cat((self.decoder_hidden[0],
                self.hidden_img[0]), 1)), dim = 1)
            q_attn_weights = F.softmax(self.attn_q(torch.cat((self.decoder_hidden[0],
                self.hidden_q[0]), 1)), dim = 1)

            #calculate c(i)
            img_attn_applied = torch.bmm(img_attn_weights.unsqueeze(0),
                    img_emb.unsqueeze(0))
            q_attn_applied = torch.bmm(q_attn_weights.unsqueeze(0),
                    q_emb.unsqueeze(0))

            #calculate beta
            tmp = attn_mod(self.decoder_hidden)
            mod_attn_weight_img_unnorm = (tmp + attn_mod_img(img_attn_applied)).tanh()
            mod_attn_weight_q_unnorm = (tmp + attn_mod_q(q_attn_applied)).tanh()
            mod_attn_weights = F.softmax(torch.cat((mod_attn_weight_img_unnorm, 
                mod_attn_weight_q_unnorm), dim = 0))

            d_img_i = fusion_img(img_attn_applied)
            d_q_i = fusion_q(q_attn_applied)

            fs_apply = torch.bmm(mod_attn_weights.unsqueeze(0),
                    torch.cat((d_img_i, d_q_i), dim = 1).unsqueeze(0)) #this should be done above, but not sure of dimensions

            fs_output = (fusion(self.decoder_hidden) + fs_apply).tanh()

            output, self.decoder_hidden = self.decoder(fs_output, self.decoder_hidden)
           
            self.prev_output = output
        return output

    def init_hidden(self):
        return (torch.zeros(1, 1, self.vocab_length), 
                torch.zeros(1, 1, self.vocab_length))


if __name__ == "__main__":
    v = AttentionBasedMultiModalFusion(4, 2, 2, 6, 4, 4, 5, 5, 2)
    v.forward()
