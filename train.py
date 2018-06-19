# raise ValueError("deal with Variable requires_grad, and .cuda()")
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence

import itertools
import random
import math
import sys
import os
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_token, EOS_token, PAD_token
from model import EncoderRNN, LuongAttnDecoderRNN, Attn, NGramLanguageModeler
from config import MAX_LENGTH, USE_CUDA, teacher_forcing_ratio, save_dir
import sys
import gensim
import logging
import pprint
# from plot import plotPerplexity

cudnn.benchmark = True
#############################################
# generate file name for saving parameters
#############################################
def filename(reverse, obj):
	filename = ''
	if reverse:
		filename += 'reverse_'
	filename += obj
	return filename


#############################################
# Prepare Training Data
#############################################
def indexesFromSentence(voc, sentence):
    output = []
    for word in sentence.split(' '):
        if word in voc.word2index:
            output.append(voc.word2index[word])
        else:
            output.append(voc.word2index['UNK'])
    #chabge index to wordvector
    return output + [EOS_token]
    #return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# batch_first: true -> false, i.e. shape: seq_len * batch
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue)) 

def binaryMatrix(l, value=PAD_token):
    m = []
    for i in range(len(l)):
        m.append([])
        for j in range(len(l[i])):
            if l[i][j] == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# convert to index, add EOS
# return input pack_padded_sequence
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = [len(indexes) for indexes in indexes_batch]
    padList = zeroPadding(indexes_batch)
    padVar = Variable(torch.LongTensor(padList))
    return padVar, lengths

# convert to index, add EOS, zero padding
# return output variable, mask, max length of the sentences in batch
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = Variable(torch.ByteTensor(mask))
    padVar = Variable(torch.LongTensor(padList))
    return padVar, mask, max_target_len

# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by input length, reverse input
# return input, lengths for pack_padded_sequence, output_variable, mask
def batch2TrainData(voc, pair_batch, reverse):
    #print("--------------------original batch--------------------------")
    #pprint.pprint(pair_batch)
    out_seq2in_index = out_in_table(pair_batch)
    
    pair_batch = pair_batch_transform(pair_batch)#strip off the middle square brackets
    #[ [ [A1,A2], [A2,A3], [A3,A4] ] , [ [B1,B2] ],...] to [ [A1,A2], [A2,A3], [A3,A4], [B1,B2],...]
    #print("--------------------batch stripped off brackets--------------------------")
    #pprint.pprint(pair_batch)
    if reverse:
        pair_batch = [pair[::-1] for pair in pair_batch]
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    #print("--------------------sorted batch--------------------------")
    #pprint.pprint(pair_batch)
    old_index2new_index, pair_batch = index_table(pair_batch)#build index table, strip off the head(old index) of seq
    #print("--------------------head-cut batch--------------------------")
    #pprint.pprint(pair_batch)
    #print("--------------------out_seq2in_index--------------------------")
    #pprint.pprint(out_seq2in_index)
    #print("--------------------old_index2new_index--------------------------")
    #pprint.pprint(old_index2new_index)
    input_batch, output_batch = [], []
    for i in range(len(pair_batch)):
        input_batch.append(pair_batch[i][0])
        output_batch.append(pair_batch[i][1])
    input, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return input, lengths, output, mask, max_target_len, out_seq2in_index, old_index2new_index
def index_table(pair_batch):
    old_index2new_index = {}
    new_pair_batch = []
    for i in range(len(pair_batch)):
        pair_buf_in = pair_batch[i][0].split(' ')
        old_index2new_index[int(pair_buf_in[0])] = i # int(pair_buf_in[0]) = old_index, i = new_index
        #pair_buf = [line_combine(pair_buf_in[1:]).strip(), pair_batch[i][1]]
        new_pair_batch.append([line_combine(pair_buf_in[1:]).strip(), pair_batch[i][1]])
    return old_index2new_index, new_pair_batch
def line_combine(line_split):
    line_comb = ''
    for i in line_split:
        line_comb += i
        line_comb += ' '
    return line_comb
def out_in_table(pair_batch):#unsorted batch, and the middle square brackets have not been stripped off
    out_seq2in_index = {}
    count = 0
    start_of_group = 0
    for group in pair_batch:
        for i_pair in range(len(group)):
            out_seq2in_index[group[i_pair][1]] = [ int(group[i][0].split(" ")[0]) for i in range(0, i_pair+1)]
            count += 1
        start_of_group = count
    return out_seq2in_index
def pair_batch_transform(pair_batch):
    new_pair_batch = []
    for group in pair_batch:
        for pair in group:
            new_pair_batch.append(pair)
    return new_pair_batch
def pairs_transform(pairs):
    new_pairs = []
    new_pair = []
    output_seq = []
    #start_of_group = 0
    count = 0
    #out_seq2in_index= {}
    pair_buf = []
    for pair in pairs:
        for i in range(len(pair)-1):
            input_seq = str(count) + ' ' + pair[i]#add index of the seq at the top of the seq
            new_pair.append(input_seq.strip())
            new_pair.append(pair[i+1].strip())
            pair_buf.append(new_pair)
            #out_seq2in_index[new_pair[1]] = [i for i in range(start_of_group, count + 1)]# count = the index of pair(unsorted) now
            new_pair = []
            count += 1
        #start_of_group = count
        new_pairs.append(pair_buf)
        pair_buf = []
    #new_pairs= [ [ [A1,A2], [A2,A3], [A3,A4] ] , [ [B1,B2] ],...]
    return new_pairs #, out_seq2in_index


#############################################
# Training
#############################################

def maskNLLLoss(input, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(input, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.cuda() if USE_CUDA else loss
    return loss, nTotal.data[0]

def train(input_variable, lengths, target_variable, mask, max_target_len, out_seq2in_index, old_index2new_index, encoder, decoder, embedding, 
          encoder_optimizer, decoder_optimizer, batch_size, w2v_model, voc, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    if USE_CUDA:
        input_variable = input_variable.cuda()
        target_variable = target_variable.cuda()
        mask = mask.cuda()

    loss = 0
    print_losses = []
    n_totals = 0 

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths, w2v_model, voc, None)

    decoder_input = Variable(torch.LongTensor([[SOS_token for _ in range(batch_size)]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Run through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs, w2v_model, voc
            )
            decoder_input = target_variable[t].view(1, -1) # Next input is current target
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.data[0] * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs, w2v_model, voc
            )
            topv, topi = decoder_output.data.topk(1) # [64, 1]

            decoder_input = Variable(torch.LongTensor([[topi[i][0] for i in range(batch_size)]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.data[0] * nTotal)
            n_totals += nTotal

    loss.backward()

    clip = 50.0
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals 

def trainIters(corpus, corpus_index, strip, pre_modelFile, reverse, n_iteration, learning_rate, batch_size, n_layers, hidden_size, 
                print_every, save_every, loadFilename=None, attn_model='dot', decoder_learning_ratio=5.0):

    voc, pairs = loadPrepareData(corpus, corpus_index, strip)
    pairs = pairs_transform(pairs)#transform [seq1,seq2,...] to [concatenated_input_seqi,seqi+1]
    # training data
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    training_batches = None
    try:
        training_batches = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 
                                                   '{}_{}_{}.tar'.format(n_iteration, \
                                                                         filename(reverse, 'training_batches'), \
                                                                         batch_size)))
    except FileNotFoundError:
        print('Training pairs not found, generating ...')
        training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)], reverse)
                          for _ in range(n_iteration)]
        torch.save(training_batches, os.path.join(save_dir, 'training_data', corpus_name, 
                                                  '{}_{}_{}.tar'.format(n_iteration, \
                                                                        filename(reverse, 'training_batches'), \
                                                                        batch_size)))
    # model
    checkpoint = None 
    #print('Building pretrained word2vector model...')
    embedding = nn.Embedding(300, hidden_size) #The dimension of google's model is 300
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
    #replace embedding by pretrained_model
    print('Building encoder and decoder ...')
    encoder = EncoderRNN(300, hidden_size, embedding, n_layers)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers)
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
    # use cuda
    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # optimizer
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])

    # Load Google's pre-trained Word2Vec model.
    print('Loading w2v_model ...')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(pre_modelFile, binary=True)
    print("Loading complete!")

    # initialize
    print('Initializing ...')
    start_iteration = 1
    perplexity = []
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1
        perplexity = checkpoint['plt']

    for iteration in tqdm(range(start_iteration, n_iteration + 1)):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len, out_seq2in_index, old_index2new_index = training_batch

        loss = train(input_variable, lengths, target_variable, mask, max_target_len, out_seq2in_index, old_index2new_index, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, w2v_model, voc)
        print_loss += loss
        perplexity.append(loss)

        if iteration % print_every == 0:
            print_loss_avg = math.exp(print_loss / print_every)
            # perplexity.append(print_loss_avg)
            # plotPerplexity(perplexity, iteration)
            print('%d %d%% %.4f' % (iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, 'model', corpus_name, '{}-{}_{}'.format(n_layers, n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, filename(reverse, 'backup_bidir_model'))))
