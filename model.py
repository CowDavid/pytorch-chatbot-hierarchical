import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys

from config import USE_CUDA, MAX_LENGTH
import numpy as np
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
def index_seq2vector_seq(input_seq, embedding):
    output_seq = []
    output_words = []
    #print(len(input_seq[0][0]), "x", len(input_seq[0]), "x", len(input_seq))
    for seq in input_seq.data:
        for w in seq:
            word_var = Variable(torch.LongTensor([w])).cuda() if USE_CUDA else Variable(torch.LongTensor([w]))
            log_probs, word_vector = embedding(word_var)
            word_vector = np.array(word_vector.data).tolist()
            output_words.append(word_vector)
        output_seq.append(output_words)
        output_words = []
    
    #print(len(output_seq[0][0]), "x", len(output_seq[0]), "x", len(output_seq))
    #print(type(output_seq))
    output_seq = Variable(torch.FloatTensor(output_seq)).cuda() if USE_CUDA else Variable(torch.FloatTensor(output_seq))
    return output_seq
def index_seq2vector_seq_google(input_seq, w2v_model, voc):
    line = []
    input_variable_words = []
    for sentence in input_seq.data:
        for word_index in sentence:
            word = voc.index2word[word_index]
            try:
                word_vec = w2v_model[word].tolist()
            except KeyError:
                word_vec = w2v_model['UNK'].tolist()
            line.append(word_vec)
            #print(voc.index2word[word_index]," ",end=' ')
        input_variable_words.append(line)
        line = []
    #print(input_variable_words)
    word_vectors = Variable(torch.FloatTensor(input_variable_words))
    return word_vectors
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seq, input_lengths, w2v_model, voc, hidden=None):
        #embedded = self.embedding(input_seq)
        embedded = index_seq2vector_seq_google(input_seq, w2v_model, voc)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs (1, batch, hidden)
        return outputs, hidden
class EncoderRNN_sentence(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        #self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seq, input_lengths, w2v_model, voc, hidden=None):
        #embedded = self.embedding(input_seq)
        #embedded = index_seq2vector_seq_google(input_seq, w2v_model, voc)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs (1, batch, hidden)
        return outputs, hidden
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden [1, 64, 512], encoder_outputs [14, 64, 512]
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(batch_size, max_len)) # B x S

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, 1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        # hidden [1, 512], encoder_output [1, 512]
        if self.method == 'dot':
            energy = hidden.squeeze(0).dot(encoder_output.squeeze(0))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.squeeze(0).dot(energy.squeeze(0))
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.squeeze(0).dot(energy.squeeze(0))
            return energy

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, w2v_model, voc):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        #log_probs, embedded = self.embedding(input_seq)
        #embedded = index_seq2vector_seq(input_seq, self.embedding)
        embedded = index_seq2vector_seq_google(input_seq, w2v_model, voc)
        embedded = self.embedding_dropout(embedded) #[1, 64, 512]
        if(embedded.size(0) != 1):
            raise ValueError('Decoder input sequence length should be 1')

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs) #[64, 1, 14]
        # encoder_outputs [14, 64, 512]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) #[64, 1, 512] 

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) #[64, 512]
        context = context.squeeze(1) #[64, 512]
        concat_input = torch.cat((rnn_output, context), 1) #[64, 1024]
        concat_output = F.tanh(self.concat(concat_input)) #[64, 512]

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output) #[64, output_size]
        output = F.softmax(output, 1)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights
