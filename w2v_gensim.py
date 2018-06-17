import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import sys
import logging
from gensim.models import Word2Vec
from load import loadPrepareData
import os
from config import MAX_LENGTH, USE_CUDA, teacher_forcing_ratio, save_dir
from gensim.corpora import Dictionary
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-tr', '--train', help='Train the model with corpus')
    parser.add_argument('-te', '--test', help='Test the saved model')
    parser.add_argument('-l', '--load', help='Load the model and train')
    parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')
    parser.add_argument('-r', '--reverse', action='store_true', help='Reverse the input sequence')
    parser.add_argument('-f', '--filter', action='store_true', help='Filter to small training data set')
    parser.add_argument('-i', '--input', action='store_true', help='Test the model by input the sentence')
    parser.add_argument('-it', '--iteration', type=int, default=100000, help='Train the model with it iterations')
    parser.add_argument('-p', '--print', type=int, default=5000, help='Print every p iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-la', '--layer', type=int, default=1, help='Number of layers in encoder and decoder')
    parser.add_argument('-hi', '--hidden', type=int, default=300, help='Hidden size in encoder and decoder')
    parser.add_argument('-be', '--beam', type=int, default=1, help='Hidden size in encoder and decoder')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-s', '--save', type=float, default=10000, help='Save every s iterations')
    parser.add_argument('-pre', '--pretrained_model', help='Pretrained tri-gram model')
    parser.add_argument('-lo', '--loss', help='Draw the loss trend graph while the seq2seq model was being trained')
    parser.add_argument('-dc', '--diff_corpus', help='Different corpus for testing')
    parser.add_argument('-fb', '--frequency_boundary', type=int, default=2, help='The frequency_boundary for vocabulary')
    args = parser.parse_args()
    return args
def parseFilename(filename, test=False):
    filename = filename.split('/')
    dataType = filename[-1][:-4] # remove '.tar'
    parse = dataType.split('_')
    reverse = 'reverse' in parse
    layers, hidden = filename[-2].split('_')
    n_layers = int(layers.split('-')[0])
    hidden_size = int(hidden)
    return n_layers, hidden_size, reverse
def run(args):
    if args.train and not args.load:
        trainWord2vec(args.train, args.iteration, args.hidden, args.frequency_boundary)
    elif args.test:
        testWord2vec(args.test, args.corpus)

def trainWord2vec(corpus, iteration, hidden_size, frequency_boundary):
    voc, pairs = loadPrepareData(corpus)
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    sentences = []
    for pair in pairs:
        sentences.append(pair[0].split(' '))
    print("Sentences ready, start training...")
    model = Word2Vec(iter=iteration, size=hidden_size, window=10, min_count=frequency_boundary, workers=4)  
    model.build_vocab(sentences) 
    model.train(sentences,total_examples=model.corpus_count,epochs=model.iter)
    directory = os.path.join(save_dir, 'model', corpus_name, 'gensim','hi{}fb{}'.format(hidden_size,frequency_boundary))
    if not os.path.exists(directory):
        os.makedirs(directory)
    model.save(os.path.join(directory,'mymodel{}'.format(iteration)))
def testWord2vec(modelfile, corpus):
    model = Word2Vec.load(modelfile)

    while(1):
        try:
            test_word = input("Input the test word: ")
            if test_word == 'q':
                break
            print("\nThe most similar word of <{}> is: ".format(test_word))
            print(model.wv.most_similar([test_word]))
        except KeyError:
            print("Incorrect spelling.")
if __name__ == '__main__':
    args = parse()
    run(args)