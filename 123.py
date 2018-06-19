from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import sys
import gensim
import logging
import pprint
from load import loadPrepareData
from load import SOS_token, EOS_token, PAD_token
from train import batch2TrainData, pairs_transform
import random
import itertools
from torch import optim
import re
import unicodedata

def pair_batch_transform(pair_batch):
    new_pair_batch = []
    end_of_group = []
    count = 0
    dict_pairs = {}
    for group in pair_batch:
        for pair in group:
            new_pair_batch.append(pair)
            count += 1
            dict_pairs[count - 1] = pair
        end_of_group.append(count)
    return new_pair_batch, end_of_group, dict_pairs

corpus_index = './data/movie_conversations.txt'
corpus = './data/movie_lines.txt' 
voc, pairs = loadPrepareData(corpus, corpus_index, 3)
#pprint.pprint(pairs[0])
#print("length of pair: ",len(pairs))
pairs = pairs_transform(pairs)
#pprint.pprint(pairs[:5])

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seq, input_lengths, voc, end_of_inputs, hidden=None):
        embedded = self.embedding(input_seq)
        #embedded = index_seq2vector_seq_google(input_seq, w2v_model, voc)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        #packed2 = torch.nn.utils.rnn.pack_padded_sequence(embedded, [end_of_inputs[0][1]])
        outputs, hidden = self.gru(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs (1, batch, hidden)
        return outputs, hidden
training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(3)], False)
                        for _ in range(1)]
training_batch = training_batches[0]
input_variable, lengths, target_variable, mask, max_target_len = training_batch

#print(input_variable)
'''
hidden_size = 300
embedding = nn.Embedding(voc.n_words, hidden_size)
encoder = EncoderRNN(300, hidden_size, embedding, 1)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
encoder_optimizer.zero_grad()
loss = 0
print_losses = []
n_totals = 0 

encoder_outputs, encoder_hidden = encoder(input_variable, lengths, voc, end_of_inputs, None)
print(encoder_hidden.data)
'''
#print(packed2.data)
#print("length of pair: ",len(pairs))

#pair_batch = pair_batch_transform([random.choice(pairs) for _ in range(3)])
#pprint.pprint(pair_batch)
'''
line = []
input_variable_words = []
for seq in input_variable.data:
    for word in seq:

        line.append(voc.index2word[word])
        #print(voc.index2word[word_index]," ",end=' ')
    input_variable_words.append(line)
    line = []
pprint.pprint(input_variable_words)
pprint.pprint(end_of_group)
'''



'''
def pairs_transform(pairs):
    new_pairs = []
    new_pair = []
    input_seq = ''
    output_seq = []
    for pair in pairs:
        for i in range(len(pair)-1):
            input_seq += pair[i] + ' '
            new_pair.append(input_seq.strip())
            new_pair.append(pair[i+1].strip())
            new_pairs.append(new_pair)
            new_pair = []
        input_seq = ''
    return new_pairs
def pair_batch_transform(pair_batch):
    new_pair_batch = []
    end_of_group = []
    count = 0
    for group in pair_batch:
        for pair in group:
            new_pair_batch.append(pair)
            count += 1
        end_of_group.append(count)
    return new_pair_batch, end_of_group








def pairs_transform(pairs):
    new_pairs = []
    new_pair = []
    input_seq = ''
    output_seq = []
    end_of_inputs = []
    end_of_input = []
    for pair in pairs:
        for i in range(len(pair)):
            if i < len(pair)-1:
                input_seq += pair[i] + ' '
                end_of_input.append(len(input_seq.strip().split(" "))-1)
            if i > 0:
                output_seq.append(pair[i])
        new_pair.append(input_seq)
        new_pair.append(output_seq)
        new_pairs.append(new_pair)
        end_of_inputs.append(end_of_input)
        new_pair = []
        input_seq = ''
        output_seq = []
        end_of_input = []
    return new_pairs, end_of_inputs



def pairs_transform(pairs):
    new_pairs = []
    new_pair = []
    output_seq = []
    start_of_group = 0
    count = 0
    out_seq2in_index= {}
    pair_buf = []
    for pair in pairs:
        for i in range(len(pair)-1):
            input_seq = str(count) + ' ' + pair[i]#add index of the seq at the top of the seq
            new_pair.append(input_seq.strip())
            new_pair.append(pair[i+1].strip())
            pair_buf.append(new_pair)
            out_seq2in_index[new_pair[1]] = [i for i in range(start_of_group, count + 1)]# count = the index of pair(unsorted) now
            new_pair = []
            count += 1
        start_of_group = count
        new_pairs.append(pair_buf)
        pair_buf = []
    #new_pairs= [ [ [A1,A2], [A2,A3], [A3,A4] ] , [ [B1,B2] ],...]
    return new_pairs, out_seq2in_index
'''