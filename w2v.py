import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence

from load import readVocs, loadPrepareData
from load import SOS_token, EOS_token, PAD_token

import argparse
from config import MAX_LENGTH, USE_CUDA, teacher_forcing_ratio, save_dir
from tqdm import tqdm
import random
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import entropy
import time
def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-tr', '--train', help='Train the model with corpus')
    parser.add_argument('-te', '--test', help='Test the saved w2v model')
    parser.add_argument('-ter', '--test_vector_relation', help='Test the saved w2v model')
    parser.add_argument('-l', '--load', help='Load the model and train')
    parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')
    parser.add_argument('-r', '--reverse', action='store_true', help='Reverse the input sequence')
    parser.add_argument('-f', '--filter', action='store_true', help='Filter to small training data set')
    parser.add_argument('-i', '--input', action='store_true', help='Test the model by input the sentence')
    parser.add_argument('-it', '--iteration', type=int, default=10000, help='Train the model with it iterations')
    parser.add_argument('-p', '--print', type=int, default=5000, help='Print every p iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-la', '--layer', type=int, default=1, help='Number of layers in encoder and decoder')
    parser.add_argument('-hi', '--hidden', type=int, default=50, help='size of word vectors')
    parser.add_argument('-be', '--beam', type=int, default=1, help='Hidden size in encoder and decoder')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-s', '--save', type=float, default=10000, help='Save every s iterations')
    parser.add_argument('-co', '--context_size', type=int, default=2, help='The (n-1) of the n-gram')
    parser.add_argument('-d', '--draw', help='Draw 2D word vector with the word vector model')
    parser.add_argument('-dm', '--draw_manually', help='Draw 2D word vector manually')
    parser.add_argument('-fb', '--frequency_boundary', type=int, default=1500, help='The frequency_boundary of the 2D graph')
    parser.add_argument('-lo', '--loss', help='Draw the loss trend graph while the word vector model was being trained')
    parser.add_argument('-pr', '--predict', help='Predict the next word with previous 2 words.')
    args = parser.parse_args()
    return args
class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)
        #self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = torch.sum(embeds, 0)
        out = F.relu(self.linear1(embeds))
        #out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=0).view(1,self.vocab_size)
        return log_probs, embeds
def train_word_vector(corpus, n_iteration, hidden_size, context_size, learning_rate, batch_size, loadFilename=None):
    voc, pairs = loadPrepareData(corpus)
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    CONTEXT_SIZE = context_size
    EMBEDDING_DIM = hidden_size
    try:
        trigrams = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 
                                                   '{}_{}_{}.tar'.format(n_iteration, \
                                                                         'training_batches', \
                                                                         batch_size)))
    except FileNotFoundError:
        test_sentence = []
        for i in range(batch_size):
            pair = random.choice(pairs)
            test_sentence.append(pair[0].split())
            test_sentence[i].insert(0,"SOS")
            test_sentence[i].append("EOS")
        #print(test_sentence[:3])
        trigrams = []
        for j in range(len(test_sentence)):
            for i in range(len(test_sentence[j]) - 2):
                trigram = ([test_sentence[j][i], test_sentence[j][i + 1]], test_sentence[j][i + 2])
                trigrams.append(trigram)
        torch.save(trigrams, os.path.join(save_dir, 'training_data', corpus_name, 
                                                  '{}_{}_{}.tar'.format(n_iteration, \
                                                                        'training_batches', \
                                                                        batch_size)))
    #print the first 3, just so you can see what they look like
    #print(trigrams[:30])
    #print(voc.n_words())
    #vocab = set(test_sentence)
    #word_to_ix = {word: i for i, word in enumerate(vocab)}
    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(voc.n_words, EMBEDDING_DIM, CONTEXT_SIZE)

    if loadFilename:
        checkpoint = torch.load(loadFilename)
        model.load_state_dict(checkpoint['w2v'])

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    if loadFilename:
        optimizer.load_state_dict(checkpoint['w2v_opt'])
    print("There are {} trigrams.".format(len(trigrams)))
    print("Total {} iterations.".format(n_iteration))

    start_iteration = 1
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1
        losses = checkpoint['losses']
        print("{} iterations left...".format(n_iteration - start_iteration + 1))
    for iteration in tqdm(range(start_iteration, n_iteration + 1)):
        total_loss = torch.Tensor([0])
        for context, target in trigrams:
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in variables)
            context_idxs = []
            for w in context:
                if w not in voc.word2index:
                    context_idxs.append(voc.word2index['UNK'])
                else:
                    context_idxs.append(voc.word2index[w])
            #context_idxs = [voc.word2index[w] for w in context]
            context_var = Variable(torch.LongTensor(context_idxs))
            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()
            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs, embeds = model(context_var)
            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a variable)
            if target not in voc.word2index:
                target = 'UNK'

            loss = loss_function(log_probs, Variable(
                torch.LongTensor([voc.word2index[target]])))
            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        losses.append(total_loss)
        save_every = 500
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, 'model', corpus_name, 'hi{}_ba{}'.format(hidden_size, batch_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                    'iteration': iteration,
                    'w2v': model.state_dict(),
                    'w2v_opt': optimizer.state_dict(),
                    'loss': loss,
                    'losses': losses
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'backup_w2v_model')))
         
    print('\n')
    print("Training completed!")
    print('\n')
    print("Loss: {}".format(losses))  # The loss decreased every iteration over the training data!
    directory = os.path.join(save_dir, 'model', corpus_name, 'hi{}_ba{}'.format(hidden_size, batch_size))
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({
                'iteration': n_iteration,
                'w2v': model.state_dict(),
                'w2v_opt': optimizer.state_dict(),
                'loss': loss,
                'losses':losses
            }, os.path.join(directory, '{}_{}.tar'.format(n_iteration, 'backup_w2v_model')))
def test_word_vector(modelFile, corpus, EMBEDDING_DIM, CONTEXT_SIZE):
    checkpoint = torch.load(modelFile)
    voc, pairs = loadPrepareData(corpus)
    model = NGramLanguageModeler(voc.n_words, EMBEDDING_DIM, CONTEXT_SIZE)
    model.load_state_dict(checkpoint['w2v'])
    model.train(False)
    while(1):
        test_word = input('>')
        if test_word == 'q': break
        else:
            try:
                embeds = get_word_vector(model, voc.index2word[int(test_word)], voc, EMBEDDING_DIM)
                print("Word freauency of '{}': {}".format(voc.index2word[int(test_word)], \
                    voc.word2count[voc.index2word[int(test_word)]]))
            except KeyError:
                print("This index is vacant.")
            except ValueError:
                print("Please input an index.")
            #embeds = get_word_vector(model, test_word, voc, EMBEDDING_DIM)
            #print("The word vector of '{}': {}".format(test_word, embeds.data.view(1, EMBEDDING_DIM)))
def test_vector_relation(modelFile, corpus, EMBEDDING_DIM, CONTEXT_SIZE):
    checkpoint = torch.load(modelFile)
    voc, pairs = loadPrepareData(corpus)
    model = NGramLanguageModeler(voc.n_words, EMBEDDING_DIM, CONTEXT_SIZE)
    model.load_state_dict(checkpoint['w2v'])
    model.train(False)
    word1, word2, word3, word4 = "heaven", "hell", "good", "cat"
    word4 = input('>')
    test_word1 = np.array(get_word_vector(model, word1, voc, EMBEDDING_DIM).data)
    test_word2 = np.array(get_word_vector(model, word2, voc, EMBEDDING_DIM).data)
    test_word3 = np.array(get_word_vector(model, word3, voc, EMBEDDING_DIM).data)
    test_word4 = np.array(get_word_vector(model, word4, voc, EMBEDDING_DIM).data)
    #test_word4_like = test_word3 - (test_word1 - test_word2)
    test_word4_like = test_word4

    _1st, _2nd, _3rd, _4th = 99999999, 99999999, 99999999, 99999999
    i_1st, i_2nd, i_3rd, i_4th = -1, -1, -1, -1
    for i in tqdm(range(0, voc.n_words)):
        i_vector = np.array(get_word_vector(model, voc.index2word[i], voc, EMBEDDING_DIM).data)
        distance = ((i_vector - test_word4_like) ** 2).mean(axis=None)
        #print(distance)
        if distance < _1st:
            _4th, _3rd, _2nd, _1st = _3rd, _2nd, _1st, distance
            i_4th, i_3rd, i_2nd, i_1st = i_3rd, i_2nd, i_1st, i
            #print("1st index:", i)
        elif distance < _2nd:
            _4th, _3rd, _2nd = _3rd, _2nd, distance
            i_4th, i_3rd, i_2nd = i_3rd, i_2nd, i
        elif distance < _3rd:
            _4th, _3rd = _3rd, distance
            i_4th, i_3rd = i_3rd, i
        elif distance < _4th:
            _4th = distance
            i_4th = i
    _1st_word = voc.index2word[i_1st]
    _2nd_word = voc.index2word[i_2nd]
    _3rd_word = voc.index2word[i_3rd]
    _4th_word = voc.index2word[i_4th]
    print("Most likely words of {}: {} > {} > {} > {} > other_words".format(word4,
     _1st_word, _2nd_word, _3rd_word, _4th_word)) 
def get_word_vector(model, test_word, voc, EMBEDDING_DIM):
    try:
        if test_word not in voc.word2index:
            test_word = 'UNK'
        test_word_idxs = [voc.word2index[test_word]]
        test_word_var = Variable(torch.LongTensor(test_word_idxs))
        log_probs, embeds = model(test_word_var)
        return embeds
    except KeyError:
        print("Incorrect spelling or unseen word.")
def draw_manually(modelFile, corpus, EMBEDDING_DIM, CONTEXT_SIZE, frequency_boundary, batch_size):
    checkpoint = torch.load(modelFile)
    voc, pairs = loadPrepareData(corpus)
    model = NGramLanguageModeler(voc.n_words, EMBEDDING_DIM, CONTEXT_SIZE)
    model.load_state_dict(checkpoint['w2v'])
    model.train(False);
    words = input("Input space-separated words: ").split()
    labels = [words[0]]
    new_word = np.array(get_word_vector(model, words[0], voc, EMBEDDING_DIM).data)
    vectors2D = np.array([new_word])
    for w in words[1:]:
        labels.append(w)
        new_word = np.array(get_word_vector(model, w, voc, EMBEDDING_DIM).data)
        vectors2D = np.concatenate((vectors2D, [new_word]), axis = 0)
    print("Shape of vectors2D: {}".format(vectors2D.shape))
    file_name = 'manually_{}2{}.png'.format(words[0], words[-1])
    iteration = os.path.split(modelFile)[-1].split('_')[0]
    tsne(corpus, len(words), vectors2D, labels, file_name, iteration, batch_size, EMBEDDING_DIM)
def tsne(corpus, n_sne, vectors2D, labels, file_name, iteration, batch_size, EMBEDDING_DIM):
    print("t-SNE processing...")
    time_start = time.time()
    tsne = TSNE()
    tsne_results = tsne.fit_transform(vectors2D)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    tsne_results = tsne_results.reshape(2,vectors2D.shape[0])
    plt.scatter(tsne_results[0], tsne_results[1], marker=".")
    for label, x, y in zip(labels, tsne_results[0], tsne_results[1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(0, 0),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0', fc='yellow', alpha=0))
    hi_ba_it = "hi{}_ba{}_it{}".format(EMBEDDING_DIM, batch_size, iteration)
    directory = os.path.join(save_dir, 'w2v_image', corpus_name, "manual_graph", hi_ba_it)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.join(directory, file_name)
    plt.savefig(directory, format='png')
def draw_2D_word_vector(modelFile, corpus, EMBEDDING_DIM, CONTEXT_SIZE, frequency_boundary, batch_size):
    checkpoint = torch.load(modelFile)
    voc, pairs = loadPrepareData(corpus)
    model = NGramLanguageModeler(voc.n_words, EMBEDDING_DIM, CONTEXT_SIZE)
    model.load_state_dict(checkpoint['w2v'])
    model.train(False);
    new_word = voc.index2word[0]
    labels = [new_word]
    new_word = np.array(get_word_vector(model, new_word, voc, EMBEDDING_DIM).data)
    vectors2D = np.array([new_word])
    start_word = 0
    index2vector = {start_word:new_word}
    below1000_count = 0
    #frequency_boundary = 1600
    for i in range(voc.n_words):
        new_word = voc.index2word[i]
        if voc.word2count[new_word] <= frequency_boundary:
            below1000_count += 1
        else:
            labels.append(new_word)
            new_word = np.array(get_word_vector(model, new_word, voc, EMBEDDING_DIM).data)
            vectors2D = np.concatenate((vectors2D, [new_word]), axis = 0)
        index2vector[i] = [new_word]
    print("{} words out of {} words are in low frequency({} times).".format(\
        below1000_count, voc.n_words, frequency_boundary))
    print("{} words left".format(voc.n_words - below1000_count))
    print("Shape of vectors2D: {}".format(vectors2D.shape))
    file_name = 'b{}vectors2D.png'.format(frequency_boundary)
    iteration = os.path.split(modelFile)[-1].split('_')[0]
    tsne(corpus, voc.n_words, vectors2D, labels, file_name, iteration, batch_size, EMBEDDING_DIM)
def loss_graph(modelFile, corpus, EMBEDDING_DIM, CONTEXT_SIZE):
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    checkpoint = torch.load(modelFile)
    losses = checkpoint['losses']
    it = checkpoint['iteration']
    plt.plot(losses)
    directory = os.path.join(save_dir, 'w2v_image', corpus_name, 'loss_graph')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.join(directory,'it{}d{}_loss_graph.png'.format(it, EMBEDDING_DIM))
    plt.savefig(directory, format='png')
def predict_word(modelFile, corpus, EMBEDDING_DIM, CONTEXT_SIZE):
    checkpoint = torch.load(modelFile)
    voc, pairs = loadPrepareData(corpus)
    model = NGramLanguageModeler(voc.n_words, EMBEDDING_DIM, CONTEXT_SIZE)
    model.load_state_dict(checkpoint['w2v'])
    model.train(False)
    print("Please input 2 space-separated words(input 'q' to exit)")
    while(1):
        test_word = input('>')
        test_word = test_word.split()
        if test_word[0] == 'q': break
        if len(test_word) != 2:
            print("You should input 2 words!")
        else:
            try:
                test_word_idxs = [voc.word2index[w] for w in test_word]
                test_word_var = Variable(torch.LongTensor(test_word_idxs))
                log_probs, embeds = model(test_word_var)
                _, i_predicted_word = torch.max(log_probs, 1)
                print("The next word of '{} {}' is '{}'".format(test_word[0], 
                    test_word[1], voc.index2word[i_predicted_word.data[0]]))
            except KeyError:
                print("Incorrect spelling or unseen word.")

def run(args):
    reverse, fil, n_iteration, print_every, save_every, learning_rate, n_layers, hidden_size, batch_size, beam_size, input = \
        args.reverse, args.filter, args.iteration, args.print, args.save, args.learning_rate, \
        args.layer, args.hidden, args.batch_size, args.beam, args.input
    context_size = args.context_size
    if args.train and not args.load:
        train_word_vector(args.train, n_iteration, hidden_size, context_size, learning_rate, batch_size)
    elif args.load:
        train_word_vector(args.train, n_iteration, hidden_size, context_size, learning_rate, batch_size,
         loadFilename=args.load)
    elif args.test:
        test_word_vector(args.test, args.corpus, hidden_size, context_size)
    elif args.draw:
        draw_2D_word_vector(args.draw, args.corpus, hidden_size, context_size, args.frequency_boundary, batch_size)
    elif args.draw_manually:
        draw_manually(args.draw_manually, args.corpus, hidden_size, context_size, args.frequency_boundary, batch_size)
    elif args.test_vector_relation:
        test_vector_relation(args.test_vector_relation, args.corpus, hidden_size, context_size)
    elif args.loss:
        loss_graph(args.loss, args.corpus, hidden_size, context_size)
    elif args.predict:
        predict_word(args.predict, args.corpus, hidden_size, context_size)

if __name__ == '__main__':
    args = parse()
    run(args)
