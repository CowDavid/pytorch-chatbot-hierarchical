import torch
import re
import os
import unicodedata
import pprint
from tqdm import tqdm
from config import MAX_LENGTH, save_dir

SOS_token = 0
EOS_token = 1
PAD_token = 2

class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS":0, "EOS":1, "PAD":2, "UNK":3}
        self.word2count = {"SOS":1, "EOS":1, "PAD":1, "UNK":1}
        self.index2word = {0: "SOS", 1: "EOS", 2:"PAD", 3:"UNK"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"[^a-zA-Z]+", r" ", s)#strip off .?!, which do not exist in google's model
    s = re.sub(r"\s+", r" ", s).strip()
    return s
def line_combine(line_split):
    line_comb = ''
    for i in line_split:
        line_comb += i
        line_comb += ' '
    return line_comb
def makepairs(lines, index_lines):
    pairs = []
    pair = []
    print("Making pairs...")
    index2group = {}
    for line in lines:
        line_sp = line.split(' ')
        index = line_sp[0]
        index2group[index] = line_combine(line_sp[8:])
    for group in index_lines:
        for i in group:
            pair.append(index2group[i])
        pairs.append(pair)
        pair = []
    print("Pairs is completly made!")
    return pairs

def readVocs(corpus, corpus_name, corpus_index, corpus_index_name):
    print("Reading index lines...")
    with open(corpus_index) as f:
        index = f.readlines()
    index_content = [x.split(' ')[6:] for x in index]
    index_line = []
    index_lines = []
    for line in index_content:
        for i in iter(line):
            index_line.append(i.strip('[').strip(']\n').strip(',').strip("'"))
        index_lines.append(index_line)
        index_line = []
    #The format of index_lines is [['L194', 'L195', 'L196', 'L197'],['L198', 'L199'],...]
    # combine every two lines into pairs and normalize
    print("Reading corpus lines...")
    with open(corpus, 'r', errors='ignore') as f:
        content = f.readlines()
    # import gzip
    # content = gzip.open(corpus, 'rt')
    lines = [x.strip() for x in content]
    pairs = makepairs(lines, index_lines)
    #it = iter(lines)
    #pairs = [[normalizeString(x), normalizeString(next(it))] for x in it]#assume the pairs is a diologue now
    #pairs = [[x, next(it)] for x in it]
    voc = Voc(corpus_name)
    return voc, pairs

def filterPair(p):
    # input sequences need to preserve the last word for EOS_token
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(corpus, corpus_name, corpus_index, corpus_index_name):
    voc, pairs = readVocs(corpus, corpus_name, corpus_index, corpus_index_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        for i in range(len(pair)):
            voc.addSentence(pair[i])
    print("Counted words:", voc.n_words)
    directory = os.path.join(save_dir, 'training_data', corpus_name) 
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(voc, os.path.join(directory, '{!s}.tar'.format('voc')))
    torch.save(pairs, os.path.join(directory, '{!s}.tar'.format('pairs')))
    return voc, pairs
def strip_low_frequency(voc, fb):
    original_n_words = voc.n_words
    for i in range(4, voc.n_words):
        word = voc.index2word[i]
        if voc.word2count[word] <= fb:
            del voc.word2index[word]
            del voc.word2count[word]
            voc.index2word[i] = 'UNK'
            voc.word2count['UNK'] += 1
            voc.n_words -= 1
    i_vacant = 4
    for i in range(4, original_n_words):
        word = voc.index2word[i]
        if word != 'UNK':
            voc.index2word[i_vacant] = word
            voc.word2index[word] = i_vacant
            if i != i_vacant:
                del voc.index2word[i]
            i_vacant += 1
        else:
            del voc.index2word[i]
    #print("original_n_words: ", original_n_words)
    print("Low frequency words are stripped off, {} words left...".format(voc.n_words))
    return voc
def loadPrepareData(corpus, corpus_index, strip):
    corpus_name = corpus.split('/')[-1].split('.')[0]
    corpus_index_name = corpus_index.split('/')[-1].split('.')[0]
    try:
        print("Start loading training data ...")
        voc = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc.tar'))
        pairs = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'pairs.tar'))
    except FileNotFoundError:
        print("Saved data not found, start preparing training data ...")
        voc, pairs = prepareData(corpus, corpus_name, corpus_index, corpus_index_name)
    print("n_words: ", voc.n_words)
    if strip:
        voc = strip_low_frequency(voc, strip)
    return voc, pairs
def voc_test(voc, pairs, original_n_words):
    for i in range(original_n_words):
        try:
            k = voc.index2word[i]
        except KeyError:
            print("Total number of words: ", i-1)
            print("last word:", k)
            break

