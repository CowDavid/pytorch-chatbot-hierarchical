import argparse
from train import trainIters
from evaluate import runTest, loss_graph
from load import Voc

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
    parser.add_argument('-hi', '--hidden', type=int, default=256, help='Hidden size in encoder and decoder')
    parser.add_argument('-be', '--beam', type=int, default=1, help='Hidden size in encoder and decoder')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-s', '--save', type=float, default=10000, help='Save every s iterations')
    parser.add_argument('-pre', '--pretrained_model', help='Pretrained tri-gram model')
    parser.add_argument('-lo', '--loss', help='Draw the loss trend graph while the seq2seq model was being trained')
    parser.add_argument('-dc', '--diff_corpus', help='Different corpus for testing')
    parser.add_argument('-ic', '--corpus_index', help='The conversation index pointing to the sentenses of the corpus')
    parser.add_argument('-st', '--strip', type=int, default=0, help='Strip of the low frequency words with defined frequency boundary.')
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
    reverse, fil, n_iteration, print_every, save_every, learning_rate, n_layers, hidden_size, batch_size, beam_size, input = \
        args.reverse, args.filter, args.iteration, args.print, args.save, args.learning_rate, \
        args.layer, args.hidden, args.batch_size, args.beam, args.input
    if args.train and not args.load:
        trainIters(args.train, args.corpus_index, args.strip, args.pretrained_model, reverse, n_iteration, learning_rate, batch_size,
                    n_layers, hidden_size, print_every, save_every)
    elif args.load:
        n_layers, hidden_size, reverse = parseFilename(args.load)
        trainIters(args.train, args.corpus_index, args.strip, args.pretrained_model, reverse, n_iteration, learning_rate, batch_size,
                    n_layers, hidden_size, print_every, save_every, loadFilename=args.load)
    elif args.test:
        n_layers, hidden_size, reverse = parseFilename(args.test, True)
        runTest(n_layers, args.pretrained_model, hidden_size, reverse, args.test, beam_size, input, args.corpus, args.diff_corpus)
    elif args.loss:
        loss_graph(args.loss, args.corpus, hidden_size)



if __name__ == '__main__':
    args = parse()
    run(args)
