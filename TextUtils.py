import os

import spacy

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
class TextPreProcess():
    """Class that handles Text Preprocessing using SpaCy"""
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

if __name__ == "__main__":
    a = TextPreProcess()
    print(a.tokenize("This is a template sentence"))

class GloVE():
  """Class that generates a GloVE Embedding for any word provided"""
  def __init__(self):
    """Initialise the embedding and model from txt file"""
  #  glove_file = datapath(os.path.abspath('./glove.840B.300d.txt'))
  #  tmp_file = get_tmpfile(os.path.abspath("./glove_word2vec.txt"))
  #  _ = glove2word2vec(glove_file, tmp_file)
    self.model = gensim.models.KeyedVectors.load_word2vec_format(os.path.abspath("./glove_word2vec.txt"))
    self.weights = torch.FloatTensor(self.model.vectors)
    self.embedding = nn.Embedding.from_pretrained(self.weights)

  def create_embedding(self, question):
    """Create a tensor of embeddings from the list of tokens provided"""
    e_len = len(question)
    embeds = torch.Tensor(e_len,300) #300 is dimension of GloVE output
    for x in range(e_len):
      try:
        input = torch.Tensor([self.model.vocab[question[x]].index])
        embeds[x] = self.embedding(input)
      except KeyError:
        input = torch.Tensor([self.model.vocab["unk"].index])
        embeds[x] = self.embedding(input)
    embeds = embeds.unsqueeze(1) #Make it LSTM Compatible. Tensor is now (Seq Size, Batch Size, Input Size)
    return embeds