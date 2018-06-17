import torch
from torch.autograd import Variable
import random
from train import indexesFromSentence
from load import SOS_token, EOS_token
from load import MAX_LENGTH, loadPrepareData, Voc 
from model import *
from config import USE_CUDA
import sys
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from config import MAX_LENGTH, USE_CUDA, teacher_forcing_ratio, save_dir
import gensim
import logging
import pprint
class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        return sum(self.sentence_scores) / len(self.sentence_scores)
        # return mean of sentence_score

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][i] == EOS_token:
                terminates.append(([voc.index2word[idx] for idx in self.sentence_idxes] + ['<EOS>'], 
                                   self.avgScore())) # tuple(word_list, score_float 
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, topi[0][i], idxes, scores))
        return terminates, sentences

    def toWordScore(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<EOS>')
            else:
                words.append(voc.index2word[self.sentence_idxes[i]])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<EOS>')
        return (words, self.avgScore())

def beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size, w2v_model, max_length=MAX_LENGTH):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for t in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = Variable(torch.LongTensor([[sentence.last_idx]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs, w2v_model, voc
            )
            topv, topi = decoder_output.data.topk(beam_size)
            term, top = sentence.addTopk(topi, topv, decoder_hidden, beam_size, voc)
            terminal_sentences.extend(term)
            next_top_sentences.extend(top)

        next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
        prev_top_sentences = next_top_sentences[:beam_size]
        next_top_sentences = []

    terminal_sentences += [sentence.toWordScore(voc) for sentence in prev_top_sentences]
    terminal_sentences.sort(key=lambda x: x[1], reverse=True)

    n = min(len(terminal_sentences), 15)
    return terminal_sentences[:n]

def decode(decoder, decoder_hidden, encoder_outputs, voc, w2v_model, max_length=MAX_LENGTH):

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length) #TODO: or (MAX_LEN+1, MAX_LEN+1)
    #decoder_input, decoder_hidden, encoder_outputs, w2v_model, voc
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs, w2v_model, voc
        )
        topv, topi = decoder_output.data.topk(3)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(voc.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluate(encoder, decoder, voc, sentence, beam_size, w2v_model, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc, sentence)] #[1, seq_len]
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = Variable(torch.LongTensor(indexes_batch), volatile=True).transpose(0, 1)
    input_batch = input_batch.cuda() if USE_CUDA else input_batch

    encoder_outputs, encoder_hidden = encoder(input_batch, lengths, w2v_model, voc, None)
    #encoder(input_variable, lengths, w2v_model, voc, None)
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    if beam_size == 1:
        return decode(decoder, decoder_hidden, encoder_outputs, voc, w2v_model)
    else:
        return beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size, w2v_model)


def evaluateRandomly(encoder, decoder, voc, pairs, reverse, beam_size, w2v_model, n=10):
    for i in range(n):
        #pair = random.choice(pairs)
        pair = pairs[i]
        print("=============================================================")
        if reverse:
            print('>', " ".join(reversed(pair[0].split())))
        else:
            print('>', pair[0])
        if beam_size == 1:
            output_words, attentions = evaluate(encoder, decoder, voc, pair[0], beam_size, w2v_model)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
        else:
            output_words_list = evaluate(encoder, decoder, voc, pair[0], beam_size, w2v_model)
            for output_words, score in output_words_list:
                output_sentence = ' '.join(output_words)
                print("{:.3f} < {}".format(score, output_sentence))

def evaluateInput(encoder, decoder, voc, beam_size, w2v_model):
    pair = ''
    while(1):
        try:
            pair = input('> ')
            if pair == 'q': break
            if beam_size == 1:
                output_words, attentions = evaluate(encoder, decoder, voc, pair, beam_size, w2v_model)
                output_sentence = ' '.join(output_words)
                print('<', output_sentence)
            else:
                output_words_list = evaluate(encoder, decoder, voc, pair, beam_size, w2v_model)
                for output_words, score in output_words_list:
                    output_sentence = ' '.join(output_words)
                    print("{:.3f} < {}".format(score, output_sentence))
        except KeyError:
            print("Incorrect spelling.")


def runTest(n_layers, pre_modelFile, hidden_size, reverse, modelFile, beam_size, input, corpus, diff_corpus):

    voc, pairs = loadPrepareData(corpus)
    diff_voc, diff_pairs = loadPrepareData(diff_corpus)
    embedding = nn.Embedding(300, hidden_size)
    #-----------------------------------------------------------------
    #my code
    '''
    EMBEDDING_DIM = 300 #Should be the same as hidden_size!
    if EMBEDDING_DIM != hidden_size:
        sys.exit("EMBEDDING_DIM do not equal to hidden_size. Please correct it.")
    CONTEXT_SIZE = 2
    pre_checkpoint = torch.load(pre_modelFile)
    pretrained_model = NGramLanguageModeler(voc.n_words, EMBEDDING_DIM, CONTEXT_SIZE)
    pretrained_model.load_state_dict(pre_checkpoint['w2v'])
    pretrained_model.train(False)
    embedding = pretrained_model
    '''
    if USE_CUDA:
        embedding = embedding.cuda()
    #-----------------------------------------------------------------
    encoder = EncoderRNN(300, hidden_size, embedding, n_layers)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers)
    if USE_CUDA:
        checkpoint = torch.load(modelFile)
    else:
        checkpoint = torch.load(modelFile, map_location='cpu')
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])
    
    # train mode set to false, effect only on dropout, batchNorm
    encoder.train(False);
    decoder.train(False);

    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    print('Loading w2v_model ...')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(pre_modelFile, binary=True)
    print("Loading complete!")

    if input:
        evaluateInput(encoder, decoder, voc, beam_size, w2v_model)
    else:
        evaluateRandomly(encoder, decoder, voc, diff_pairs, reverse, beam_size, w2v_model, 20)

def loss_graph(modelFile, corpus, EMBEDDING_DIM):
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    if USE_CUDA:
        checkpoint = torch.load(modelFile)
    else:    
        checkpoint = torch.load(modelFile, map_location='cpu')
    losses = checkpoint['plt']
    it = checkpoint['iteration']
    plt.plot(losses)
    directory = os.path.join(save_dir, 'seq2seq', corpus_name, 'loss_graph')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.join(directory,'it{}d{}_loss_graph.png'.format(it, EMBEDDING_DIM))
    plt.savefig(directory, format='png')