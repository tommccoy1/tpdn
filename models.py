from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from random import shuffle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import sys
import os

import time
import math

import argparse

from binding_operations import *
from role_assignment_functions import *

from collections import OrderedDict


# Definitions of all the seq2seq models and the TPDN
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Every encoder should take a list of integers as input, and return a vector encoding.
# Every decoder should take a vector encoding as input, and return a list of logits.


# Encoder RNN for the mystery vector generating network--unidirectional GRU
class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size # Hidden size
        self.embedding = nn.Embedding(input_size, emb_size) # Embedding layer
        self.rnn = nn.GRU(emb_size, hidden_size) # Recurrent layer
     
    # A forward pass of the encoder
    def forward(self, sequence, trees=None, return_hidden=False):
        hidden = self.init_hidden(len(sequence))
        hidden_states = [hidden]
        batch_size = len(sequence)

        sequence = Variable(torch.LongTensor([sequence])).transpose(0,2)
        sequence = sequence.to(device=device)

        for element in sequence:
            embedded = self.embedding(element).transpose(0,1)
            output, hidden = self.rnn(embedded, hidden)
            hidden_states.append(hidden)
            
        if return_hidden:
            return hidden, hidden_states
        else:
            return hidden
    
    # Initialize the hidden state as all zeroes
    def init_hidden(self, batch_size):
        result = Variable(torch.zeros(1,batch_size,self.hidden_size))

        return result.to(device=device)


# Encoder RNN for the mystery vector generating network--bidirectional GRU
class EncoderBiRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size):
        super(EncoderBiRNN, self).__init__()
        self.hidden_size = hidden_size # Hidden size
        self.embedding = nn.Embedding(input_size, emb_size) # Embedding layer
        self.rnn_fwd = nn.GRU(emb_size, int(hidden_size/2)) # Recurrent layer-forward
        self.rnn_rev = nn.GRU(emb_size, int(hidden_size/2)) # Recurrent layer-backward
     
    # A forward pass of the encoder
    def forward(self, sequence, trees=None, return_hidden=False):
        batch_size = len(sequence)

        sequence_rev = Variable(torch.LongTensor([sequence[::-1]])).transpose(0,2).to(device=device)
        sequence = Variable(torch.LongTensor([sequence])).transpose(0,2).to(device=device)

        # Forward pass
        hidden_fwd = self.init_hidden(batch_size)
        hidden_states_fwd = []
        
        for element in sequence:
            embedded = self.embedding(element).transpose(0,1)
            output, hidden_fwd = self.rnn_fwd(embedded, hidden_fwd)
            hidden_states_fwd.append(hidden_fwd)
            
        # Backward pass
        hidden_rev = self.init_hidden(batch_size)
        hidden_states_rev = []
        
        for element in sequence_rev:
            embedded = self.embedding(element).transpose(0,1)
            output, hidden_rev = self.rnn_rev(embedded, hidden_rev)
            hidden_states_rev.append(hidden_rev)
            
        # Concatenate the two hidden representations
        hidden = torch.cat((hidden_fwd, hidden_rev), 2)
        
        if return_hidden:
            hidden_states = []
            for index in range(len(hidden_states_fwd)):
                hidden_states.append(torch.cat((hidden_states_fwd[index], hidden_states_rev[index]), 2))

            return hidden, hidden_states
        else:
            return hidden
 
    # Initialize the hidden state as all zeroes
    def init_hidden(self, batch_size):
        result = Variable(torch.zeros(1,batch_size,int(self.hidden_size/2)))
        
        return result.to(device=device)

 
# Encoder RNN for the mystery vector generating network--Tree-GRU.
# Based on Chen et al. (2017): Improved neural machine translation
# with a syntax-aware encoder and decoder.
class EncoderTreeRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super(EncoderTreeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        
        self.w_z = nn.Linear(emb_size, hidden_size)
        self.u_zl = nn.Linear(hidden_size, hidden_size)
        self.u_zr = nn.Linear(hidden_size, hidden_size)
        self.w_r = nn.Linear(emb_size, hidden_size)
        self.u_rl = nn.Linear(hidden_size, hidden_size)
        self.u_rr = nn.Linear(hidden_size, hidden_size)
        self.w_h = nn.Linear(emb_size, hidden_size)
        self.u_hl = nn.Linear(hidden_size, hidden_size)
        self.u_hr = nn.Linear(hidden_size, hidden_size)
        
    def tree_gru(self, word, hidden_left, hidden_right):
        z_t = nn.Sigmoid()(self.w_z(word) + self.u_zl(hidden_left) + self.u_zr(hidden_right))
        r_t = nn.Sigmoid()(self.w_r(word) + self.u_rl(hidden_left) + self.u_rr(hidden_right))
        h_tilde = F.tanh(self.w_h(word) + self.u_hl(r_t * hidden_left) + self.u_hr(r_t * hidden_right))
        h_t = z_t * hidden_left + z_t * hidden_right + (1 - z_t) * h_tilde
        
        return h_t
    
 
    def forward(self, sequence, trees=None, return_hidden=False):
        trees = [parse_digits(elt) for elt in sequence]
        final_output = None
        all_hiddens = []
        for index, input_seq in enumerate(sequence):
            this_seq_hiddens = []
            tree = trees[index]
        
            embedded_seq = []
        
            for elt in input_seq:
                embedded_seq.append(self.embedding(Variable(torch.LongTensor([elt])).to(device=device)).unsqueeze(0))
            
            leaf_nodes = []
            for elt in embedded_seq:
                this_hidden = self.tree_gru(elt, self.init_hidden(), self.init_hidden())
                leaf_nodes.append(this_hidden)
                this_seq_hiddens.append(this_hidden)
            
            current_level = leaf_nodes
            for level in tree:
                next_level = []
            
                for node in level:
                
                    if len(node) == 1:
                        next_level.append(current_level[node[0]])
                        continue
                    left = node[0]
                    right = node[1]
                
                    hidden = self.tree_gru(self.init_word(), current_level[left], current_level[right])
                
                    next_level.append(hidden)
                    this_seq_hiddens.append(hidden)
                
                current_level = next_level
                all_hiddens.append(this_seq_hiddens)
            if final_output is None:
                final_output = current_level[0][0].unsqueeze(0)
            else: 
                final_output = torch.cat((final_output, current_level[0][0].unsqueeze(0)),0)

        if return_hidden:
            return final_output.transpose(0,1), all_hiddens
        else:
            return final_output.transpose(0,1)
                                
 
   # Initialize the hidden state as all zeroes
    def init_hidden(self):
        result = Variable(torch.zeros(1,1,int(self.hidden_size)))
             
        return result.to(device=device)


    # Initialize the word hidden state as all zeroes
    def init_word(self):
        result = Variable(torch.zeros(1,1,int(self.emb_size)))
         
        return result.to(device=device)


 
 
# Unidirectional decoder RNN for the mystery vector decoding network
# At each step of decoding, the decoder takes the encoding of the
# input (i.e. the final hidden state of the encoder) as well as
# the previous hidden state. It outputs a probability distribution
# over the possible output digits; the highest-probability digit is
# taken to be that time step's output
class DecoderRNN(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size # Size of the hidden state
        self.output_size = output_size # Size of the output
        self.emb_size = emb_size
        self.rnn = nn.GRU(emb_size, hidden_size) # Recurrent unit
        self.out = nn.Linear(hidden_size, output_size) # Linear layer giving the output
        self.softmax = nn.LogSoftmax() # Softmax layer
    
    # Forward pass
    def forward(self, hidden, output_len=None, tree=None, return_hidden=False, just_embs=False, role_list=None):
        outputs = []
        hidden = nn.ReLU()(hidden)
        hidden_states = [hidden]
        
        for item in range(output_len):
            output, hidden = self.rnn(Variable(torch.zeros(1,hidden.size()[1],int(self.emb_size))).to(device=device), hidden) # Pass the inputs through the hidden layer
            output = self.softmax(self.out(output[0])) # Pass the result through softmax to make it probabilities
            outputs.append(output)
            hidden_states.append(hidden)

        #print(outputs)
        #print(outputs[0].shape)
        #14/0

        if return_hidden:
            return outputs, hidden_states
        else:
            return outputs
            
# Bidirectional decoder RNN for the mystery vector decoding network
# At each step of decoding, the decoder takes the encoding of the
# input (i.e. the final hidden state of the encoder) as well as
# the previous hidden state. It outputs a probability distribution
# over the possible output digits; the highest-probability digit is
# taken to be that time step's output
class DecoderBiRNN(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size):
        super(DecoderBiRNN, self).__init__()
        self.hidden_size = hidden_size # Size of the hidden state
        self.output_size = output_size # Size of the output
        self.emb_size = emb_size
        self.rnn_fwd = nn.GRU(emb_size, int(hidden_size/2)) # Recurrent layer-forward
        self.rnn_rev = nn.GRU(emb_size, int(hidden_size/2)) # Recurrent layer-backward
        self.out = nn.Linear(hidden_size, output_size) # Linear layer giving the output
        self.softmax = nn.LogSoftmax() # Softmax layer
        self.squeeze = nn.Linear(hidden_size, int(hidden_size/2))

    # Forward pass
    def forward(self, hidden, output_len=None, tree=None, return_hidden=False, just_embs=False, role_list=None):
        hidden = nn.ReLU()(hidden)
        hidden_list = [hidden]
        outputs = []
        encoder_hidden = self.squeeze(hidden)
        fwd_hiddens = []
        rev_hiddens = []

        fwd_hidden = encoder_hidden       
        for item in range(output_len):
            output, fwd_hidden = self.rnn_fwd(Variable(torch.zeros(1,fwd_hidden.size()[1],int(self.emb_size))).to(device=device), fwd_hidden) # Pass the inputs through the hidden layer
            fwd_hiddens.append(fwd_hidden)

        rev_hidden = encoder_hidden       
        for item in range(output_len):
            output, rev_hidden = self.rnn_rev(Variable(torch.zeros(1,rev_hidden.size()[1],int(self.emb_size))).to(device=device), rev_hidden) # Pass the inputs through the hidden layer
            rev_hiddens.append(rev_hidden)

        all_hiddens = zip(fwd_hiddens, rev_hiddens[::-1])

        for hidden_pair in all_hiddens:
            output = torch.cat((hidden_pair[0], hidden_pair[1]), 2)
            hidden_list.append(output)
            output = self.softmax(self.out(output[0])) # Pass the result through softmax to make it probabilities
            outputs.append(output)
            
        return outputs
 

            
# Bidirectional decoder RNN for the mystery vector decoding network
# At each step of decoding, the decoder takes the encoding of the
# input (i.e. the final hidden state of the encoder) as well as
# the previous hidden state. It outputs a probability distribution
# over the possible output digits; the highest-probability digit is
# taken to be that time step's output
class DecoderBiDoubleRNN(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size):
        super(DecoderBiDoubleRNN, self).__init__()
        self.hidden_size = hidden_size # Size of the hidden state
        self.output_size = output_size # Size of the output
        self.emb_size = emb_size
        self.rnn_fwd = nn.GRU(emb_size, int(hidden_size/2)) # Recurrent layer-forward
        self.rnn_rev = nn.GRU(emb_size, int(hidden_size/2)) # Recurrent layer-backward
        self.rnn_fwd2 = nn.GRU(emb_size, int(hidden_size/2)) # Recurrent layer-forward
        self.rnn_rev2 = nn.GRU(emb_size, int(hidden_size/2)) # Recurrent layer-backward
        self.out = nn.Linear(hidden_size, output_size) # Linear layer giving the output
        self.softmax = nn.LogSoftmax() # Softmax layer
        self.squeeze = nn.Linear(hidden_size, int(hidden_size/2))

    # Forward pass
    def forward(self, hidden, output_len=None, tree=None, return_hidden=False, just_embs=False, role_list=None):
        hidden = nn.ReLU()(hidden)
        hidden_list = [hidden]
        outputs = []
        encoder_hidden = self.squeeze(hidden)
        fwd_hiddens = []
        rev_hiddens = []

        fwd_hidden = encoder_hidden       
        for item in range(output_len):
            output, fwd_hidden = self.rnn_fwd(Variable(torch.zeros(1,fwd_hidden.size()[1],int(self.emb_size))).to(device=device), fwd_hidden) # Pass the inputs through the hidden layer
            fwd_hiddens.append(fwd_hidden)

        rev_hidden = encoder_hidden       
        for item in range(output_len):
            output, rev_hidden = self.rnn_rev(Variable(torch.zeros(1,rev_hidden.size()[1],int(self.emb_size))).to(device=device), rev_hidden) # Pass the inputs through the hidden layer
            rev_hiddens.append(rev_hidden)

        all_hiddens = zip(fwd_hiddens, rev_hiddens[::-1])

        for hidden_pair in all_hiddens:
            output = torch.cat((hidden_pair[0], hidden_pair[1]), 2)
            hidden_list.append(output)
            
        fwd_hiddens = []
        rev_hiddens = []

        fwd_hidden = self.squeeze(hidden_list[0])
        for item in range(output_len):
            output, fwd_hidden = self.rnn_fwd2(Variable(torch.zeros(1,fwd_hidden.size()[1],int(self.emb_size))).to(device=device), fwd_hidden) # Pass the inputs through the hidden layer 
            fwd_hiddens.append(fwd_hidden)

        rev_hidden = self.squeeze(hidden_list[-1])
        for item in range(output_len):
            output, rev_hidden = self.rnn_rev2(Variable(torch.zeros(1,rev_hidden.size()[1],int(self.emb_size))).to(device=device), rev_hidden) # Pass the inputs through the hidden layer
            rev_hiddens.append(rev_hidden)

        all_hiddens = zip(fwd_hiddens, rev_hiddens[::-1])

        for hidden_pair in all_hiddens:
            output = torch.cat((hidden_pair[0], hidden_pair[1]), 2)
            hidden_list.append(output)
            output = self.softmax(self.out(output[0])) # Pass the result through softmax to make it probabilities
            outputs.append(output)
 


        return outputs
 

# Tree-based seq2seq decoder.
# Based on Chen et al. (2018): Tree-to-tree neural networks for program translation.
class DecoderTreeRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super(DecoderTreeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_out = nn.Linear(hidden_size, vocab_size)
        self.left_child = nn.GRU(hidden_size, hidden_size)
        self.right_child = nn.GRU(hidden_size, hidden_size)
        
    def forward(self, encoding_list, output_len=None, tree=None, return_hidden=False, just_embs=False, role_list=None):
        words_out = []
        all_hiddens = []
        tree_list = tree
        #print(tree_list)
        for encoding_mini, tree in zip(encoding_list.transpose(0,1), tree_list):
            encoding = encoding_mini.unsqueeze(0)
            encoding = nn.ReLU()(encoding)
            this_hidden_list = [encoding]
            tree_to_use = tree[::-1][1:]
        
            current_layer = [encoding]
        
            for layer in tree_to_use:
                next_layer = []
                for index, node in enumerate(layer):
                    if len(node) == 1:
                        next_layer.append(current_layer[index])
                    else:
                        output, left = self.left_child(Variable(torch.zeros(1,1,self.hidden_size)).to(device=device), current_layer[index])
                        output, right = self.right_child(Variable(torch.zeros(1,1,self.hidden_size)).to(device=device), current_layer[index])

                        next_layer.append(left)
                        next_layer.append(right)
                        this_hidden_list.append(left)
                        this_hidden_list.append(right)
                current_layer = next_layer
                all_hiddens.append(this_hidden_list)
            
            if words_out == []:
                for elt in current_layer:
                    words_out.append(nn.LogSoftmax()(self.word_out(elt).view(-1).unsqueeze(0)))
            else:
                index = 0
                for elt in current_layer:
                    words_out[index] = torch.cat((words_out[index], nn.LogSoftmax()(self.word_out(elt).view(-1).unsqueeze(0))), 0)
                    index += 1
        
        if return_hidden:
            return words_out, all_hiddens
        else:
            return words_out

# INCOMPLETE - make handle batches, and convert input to one-hot, and convert output to sequence of hiddens (?)                     
class MLP(nn.Module):
    def __init__(self, inp_dim, outp_dim, n_hidden, hidden_size):
        super(MLP, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.lin1 = nn.Linear(inp_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.lin4 = nn.Linear(hidden_size, hidden_size)
        self.lin5 = nn.Linear(hidden_size, hidden_size)
        self.lin6 = nn.Linear(hidden_size, hidden_size)
        self.lin7 = nn.Linear(hidden_size, hidden_size)
        self.lin8 = nn.Linear(hidden_size, hidden_size)
        self.lin9 = nn.Linear(hidden_size, hidden_size)
        self.lin10 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, outp_dim)
        
        self.lin_list = [self.lin1, self.lin2, self.lin3, self.lin4, self.lin5, 
                         self.lin6, self.lin7, self.lin8, self.lin9, self.lin10]
        
        self.n_hidden = n_hidden
        
    def forward(self, inp, start=None, end=None):
        
        h = inp
        
        if start is None:
            for i in range(self.n_hidden):
                h = nn.Sigmoid()(self.lin_list[i](h))
        
            h = self.out(h)
        else:
            if end > self.n_hidden:
                for i in range(start, self.n_hidden):
                    h = nn.Sigmoid()(self.lin_list[i](h))

                h = self.out(h)
            else:
                for i in range(start, end):
                    h = nn.Sigmoid()(self.lin_list[i](h))

        return h
        
def pad_list(lst, length):
    if len(lst) == length:
        return lst
    else:
        return pad_list(lst + [9], length)

def onehot(length, position):
    z = torch.zeros(length)

    z[position] = 1

    return torch.FloatTensor(z)

def plus_one(lst):
    new = []

    for elt in lst:
        new.append(elt + 0)

    return new

def make_one_hot_inp(inp):
    return torch.cat([onehot(10, x) for x in plus_one(pad_list(list(inp), 6))], 0)

def make_one_hot_outp(inp):
    return torch.cat([onehot(10, x) for x in plus_one(pad_list(list(inp)[::-1], 6))], 0)
          

# INCOMPLETE - make handle batches, and convert input to one-hot, and convert output to sequence of hiddens (?)                     
class MLPEncoder(nn.Module):
    def __init__(self, inp_dim, outp_dim, n_hidden, hidden_size):
        super(MLPEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.lin1 = nn.Linear(inp_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.lin4 = nn.Linear(hidden_size, hidden_size)
        self.lin5 = nn.Linear(hidden_size, hidden_size)
        self.lin6 = nn.Linear(hidden_size, hidden_size)
        self.lin7 = nn.Linear(hidden_size, hidden_size)
        self.lin8 = nn.Linear(hidden_size, hidden_size)
        self.lin9 = nn.Linear(hidden_size, hidden_size)
        self.lin10 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, outp_dim)
        
        self.lin_list = [self.lin1, self.lin2, self.lin3, self.lin4, self.lin5, 
                         self.lin6, self.lin7, self.lin8, self.lin9, self.lin10]
        
        self.n_hidden = n_hidden
        
    def forward(self, sequence):
        #print(sequence)
        
        padded = False
        len_seq = len(sequence[0])
        if len_seq == 6:
            padded = True

        while not padded:
            sequence = [(x + [-1]) for x in sequence]

            len_seq = len(sequence[0])
            if len_seq == 6:
                padded = True

        sequence = [make_one_hot_inp(x).unsqueeze(0) for x in sequence]
        #print(sequence)
        sequence = torch.cat(sequence, 0).unsqueeze(0)
        #print(sequence)
        #print(sequence.shape)
        h = sequence.to(device=device)

        #14/0
        
        for i in range(self.n_hidden):
            #print("hshape", h.shape)
            h = nn.ReLU()(self.lin_list[i](h))
            #print("hshape", h.shape)
        
        #h = self.out(h)
        #print(h)
        #14/0

        return h
 


# INCOMPLETE - make handle batches, and convert input to one-hot, and convert output to sequence of hiddens (?)                     
class MLPDecoder(nn.Module):
    def __init__(self, inp_dim, outp_dim, n_hidden, hidden_size):
        super(MLPDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.lin1 = nn.Linear(inp_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.lin4 = nn.Linear(hidden_size, hidden_size)
        self.lin5 = nn.Linear(hidden_size, hidden_size)
        self.lin6 = nn.Linear(hidden_size, hidden_size)
        self.lin7 = nn.Linear(hidden_size, hidden_size)
        self.lin8 = nn.Linear(hidden_size, hidden_size)
        self.lin9 = nn.Linear(hidden_size, hidden_size)
        self.lin10 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, outp_dim)
        
        self.lin_list = [self.lin1, self.lin2, self.lin3, self.lin4, self.lin5, 
                         self.lin6, self.lin7, self.lin8, self.lin9, self.lin10]
        
        self.n_hidden = n_hidden
        
    def forward(self, encoding, output_len=None, tree=None, return_hidden=False, just_embs=False, role_list=None):
        h = encoding

        
        for i in range(self.n_hidden):
            h = nn.ReLU()(self.lin_list[i](h))
        
        h = self.out(h)
        h_pieces = []
        for i in range(6):
            h_pieces.append(nn.LogSoftmax(dim=1)(h.transpose(1,2)[0][10*i:10*(i+1)].transpose(0,1)))


        #print(h.transpose(1,2)[0][0:6].shape)
        #print(h_pieces[0])
        #14/0

        #print(h_pieces)
        #print(h_pieces[0])
        #14/0

        return h_pieces[:output_len]
 








# INCOMPLETE: Make its I/O compatible with other encoders       
# A tensor product encoder layer 
# Takes a list of fillers and a list of roles and returns an encoding
class TensorProductEncoder(nn.Module):
    def __init__(self, n_fillers=2, filler_dim=3, role_dim=4, 
                 final_layer_width=None, pretrained_embeddings=None, embedder_squeeze=None, binder="tpr", role_scheme="ltr", max_length=10, role_filea=None, filler_filea=None, role_fileb=None, filler_fileb=None, role_filec=None, filler_filec=None, role_filed=None, filler_filed=None):

        super(TensorProductEncoder, self).__init__()
        
        self.n_fillers = n_fillers # number of fillers
        
        # Set the dimension for the filler embeddings
        self.filler_dim = filler_dim
           
        # Set the dimension for the role embeddings
        self.role_dim = role_dim
        
        # Create an embedding layer for the fillers
        if embedder_squeeze is None:
                self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim)
                self.embed_squeeze = False
                print("no squeeze")
        else:
                self.embed_squeeze = True
                self.filler_embedding = nn.Embedding(self.n_fillers, embedder_squeeze)
                self.embedding_squeeze_layer = nn.Linear(embedder_squeeze, self.filler_dim)                
                print("squeeze")

        if pretrained_embeddings is not None:
                self.filler_embedding.load_state_dict({'weight': torch.FloatTensor(pretrained_embeddings).to(device=device)})
                self.filler_embedding.weight.requires_grad = False       

        # Create a SumFlattenedOuterProduct layer that will
        # take the sum flattened outer product of the filler
        # and role embeddings (or a different type of role-filler
        # binding function, such as circular convolution)
        if binder == "tpr":
            self.sum_layer = SumFlattenedOuterProduct()
        elif binder == "hrr":
            self.sum_layer = CircularConvolution(self.filler_dim)
        elif binder == "eltwise" or binder == "elt":
            self.sum_layer = EltWise()
        else:
            print("Invalid binder")
        
        # This final part is for including a final linear layer that compresses
        # the sum flattened outer product into the dimensionality you desire
        # But if self.final_layer_width is None, then no such layer is used
        self.final_layer_width = final_layer_width
        if self.final_layer_width is None:
            self.has_last = 0
        else:
            self.has_last = 1
            if binder == "tpr":
                self.last_layer = nn.Linear(self.filler_dim * self.role_dim, self.final_layer_width)
            else:
                self.last_layer = nn.Linear(self.filler_dim, self.final_layer_width)
     
        self.n_roles, self.seq2roles = create_role_scheme(role_scheme, max_length, n_fillers, role_filea=role_filea, filler_filea=filler_filea, role_fileb=role_fileb, filler_fileb=filler_fileb, role_filec=role_filec, filler_filec=filler_filec, role_filed=role_filed, filler_filed=filler_filed)

 
        # Create an embedding layer for the roles
        self.role_embedding = nn.Embedding(self.n_roles, self.role_dim)
        

    # Function for a forward pass through this layer. Takes a list of fillers and 
    # a list of roles and returns an single vector encoding it.
    def forward(self, filler_list, trees=None, return_hidden=False):
        role_list = [self.seq2roles(x) for x in filler_list]

        filler_list = torch.LongTensor(filler_list).to(device=device)
        role_list = torch.LongTensor(role_list).to(device=device)

        # Embed the fillers
        fillers_embedded = self.filler_embedding(filler_list)
        if self.embed_squeeze:
            fillers_embedded = self.embedding_squeeze_layer(fillers_embedded)

            
        # Embed the roles
        roles_embedded = self.role_embedding(role_list)
        
        # Create the sum of the flattened tensor products of the 
        # filler and role embeddings
        output = self.sum_layer(fillers_embedded, roles_embedded)
        
        # If there is a final linear layer to change the output's dimensionality, apply it
        if self.has_last:
            output = self.last_layer(output)
            
        return output
     


 
                 
# INCOMPLETE: Make its I/O compatible with other decoders
# A tensor product encoder layer 
# Takes a list of fillers and a list of roles and returns an encoding
class TensorProductDecoder(nn.Module):
    def __init__(self, n_roles=2, n_fillers=2, filler_dim=3, role_dim=4,
                 final_layer_width=None, pretrained_embeddings=None, embedder_squeeze=None, binder="tpr"):

        super(TensorProductDecoder, self).__init__()

        self.n_roles = n_roles # number of roles
        self.n_fillers = n_fillers # number of fillers

        # Set the dimension for the filler embeddings
        self.filler_dim = filler_dim

        # Set the dimension for the role embeddings
        self.role_dim = role_dim

        # Create an embedding layer for the fillers
        if embedder_squeeze is None:
                self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim)
                self.embed_squeeze = False
                print("no squeeze")
        else:
                self.embed_squeeze = True
                self.filler_embedding = nn.Embedding(self.n_fillers, embedder_squeeze)
                self.embedding_squeeze_layer = nn.Linear(embedder_squeeze, self.filler_dim)
                print("squeeze")

        if pretrained_embeddings is not None:
                self.filler_embedding.load_state_dict({'weight': torch.FloatTensor(pretrained_embeddings).cuda()})
                self.filler_embedding.weight.requires_grad = False


        # Create an embedding layer for the roles
        self.role_embedding = nn.Embedding(self.n_roles, self.role_dim)

        # Create a SumFlattenedOuterProduct layer that will
        # take the sum flattened outer product of the filler
        # and role embeddings (or a different type of role-filler
        # binding function, such as circular convolution)
        if binder == "tpr":
            self.sum_layer = SumFlattenedOuterProduct()
        elif binder == "hrr":
            self.sum_layer = CircularConvolution(self.filler_dim)
        elif binder == "eltwise" or binder == "elt":
            self.sum_layer = EltWise()
        else:
            print("Invalid binder")

        # This final part if for including a final linear layer that compresses
        # the sum flattened outer product into the dimensionality you desire
        # But if self.final_layer_width is None, then no such layer is used
        self.final_layer_width = final_layer_width
        if self.final_layer_width is None:
            self.has_last = 0
        else:
            self.has_last = 1
            if binder == "tpr":
                self.last_layer = nn.Linear(self.final_layer_width, self.filler_dim * self.role_dim)
            else:
                self.last_layer = nn.Linear(self.final_layer_width, self.filler_dim)

                
    # Function for a forward pass through this layer. Takes a list of fillers and 
    # a list of roles and returns an single vector encoding it.
    def forward(self, encoding, role_list=None, just_embs=False, output_len=None, tree=None, return_hidden=False):
        
        if self.has_last:
            encoding = self.last_layer(encoding) 
            
        encoding = encoding.transpose(0,1)
        #print(encoding.shape)
        #print(encoding)
        encoding = encoding.view(-1,self.role_dim,self.filler_dim)
        #print(encoding.shape)
        #print(encoding)
       
        role_list = torch.LongTensor(role_list).to(device=device)
        #print(role_list) 
        roles_embedded = self.role_embedding(role_list)
        #print(roles_embedded.shape)
        
        filler_guess = torch.bmm(roles_embedded, encoding)
        
        if just_embs:
            return filler_guess
        else:
            listembs = [self.filler_embedding(torch.LongTensor([[i]]).to(device=device)) for i in range(self.n_fillers)]
            #print(listembs)
            embmat = torch.cat(listembs, 1)
        
            fillers_exp = filler_guess.unsqueeze(3).expand(filler_guess.shape[0],-1,self.filler_dim,self.n_fillers).transpose(2,3)
            embmat_exp = embmat.expand(filler_guess.shape[0],-1,self.n_fillers,self.filler_dim)

            logs = nn.LogSoftmax(dim=2)(torch.sum(torch.pow(fillers_exp - embmat_exp, 2), dim=3)).transpose(0,2).transpose(0,1)
            #print(logs)
            #print(logs.shape)
            return logs.transpose(1,2)


            #logs, preds = torch.sum(torch.pow(fillers_exp - embmat_exp, 2), dim=3).data.topk(1, largest=False)
            #preds = preds.squeeze(2)
            #print(logs)
            #print(logs.shape) 
            #return logs # Preds would be the predicted words

 
# Unidirectional decoder RNN for the mystery vector decoding network
# At each step of decoding, the decoder takes the encoding of the
# input (i.e. the final hidden state of the encoder) as well as
# the previous hidden state. It outputs a probability distribution
# over the possible output digits; the highest-probability digit is
# taken to be that time step's output
class DecoderTPRRNN(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size, n_roles=1, n_fillers=10, filler_dim=10, role_dim=6, final_layer_width=None):
        super(DecoderTPRRNN, self).__init__()
        self.hidden_size = hidden_size # Size of the hidden state
        self.output_size = output_size # Size of the output
        self.emb_size = emb_size
        self.rnn = nn.GRU(emb_size, hidden_size) # Recurrent unit
        #self.out = nn.Linear(hidden_size, output_size) # Linear layer giving the output
        self.softmax = nn.LogSoftmax() # Softmax layer
    
        self.n_roles = n_roles # number of roles
        self.n_fillers = n_fillers # number of fillers

        self.filler_dim = filler_dim
        self.role_dim = role_dim

        self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim)
        self.role_embedding = nn.Embedding(self.n_roles, self.role_dim)

        self.final_layer_width = final_layer_width
        if self.final_layer_width is None:
            self.has_last = 0
        else:
            self.has_last = 1
            self.last_layer = nn.Linear(self.final_layer_width, self.filler_dim * self.role_dim)


    # Forward pass
    def forward(self, hidden, output_len=None, tree=None, return_hidden=False, just_embs=False, role_list=None):
        outputs = []
        hidden = nn.ReLU()(hidden)
        hidden_states = [hidden]
        
        for item in range(output_len):
            output, hidden = self.rnn(Variable(torch.zeros(1,hidden.size()[1],int(self.emb_size))).to(device=device), hidden) # Pass the inputs through the hidden layer

            encoding = output
            if self.has_last:
                encoding = self.last_layer(encoding)
                
            encoding = encoding.transpose(0,1)
            encoding = encoding.view(-1,self.role_dim,self.filler_dim)

            role_list = torch.LongTensor([[0] for _ in range(len(encoding))]).to(device=device)
            roles_embedded = self.role_embedding(role_list)

            #print(roles_embedded.shape)
            #print(encoding.shape)
            #14/0
            filler_guess = torch.bmm(roles_embedded, encoding)
            listembs = [self.filler_embedding(torch.LongTensor([[i]]).to(device=device)) for i in range(self.n_fillers)]

            embmat = torch.cat(listembs, 1)
            fillers_exp = filler_guess.unsqueeze(3).expand(filler_guess.shape[0],-1,self.filler_dim,self.n_fillers).transpose(2,3)
            embmat_exp = embmat.expand(filler_guess.shape[0],-1,self.n_fillers,self.filler_dim)
            logs = nn.LogSoftmax(dim=2)(torch.sum(torch.pow(fillers_exp - embmat_exp, 2), dim=3)).transpose(0,2).transpose(0,1)
            #return logs.transpose(1,2)



      
#            output = self.softmax(self.out(output[0])) # Pass the result through softmax to make it probabilities


            outputs.append(logs.transpose(1,2).squeeze(0))
            hidden_states.append(hidden)

        #print(outputs)
        #print(outputs[0].shape)
        #14/0

        if return_hidden:
            return outputs, hidden_states
        else:
            return outputs
  
# Unidirectional decoder RNN for the mystery vector decoding network
# At each step of decoding, the decoder takes the encoding of the
# input (i.e. the final hidden state of the encoder) as well as
# the previous hidden state. It outputs a probability distribution
# over the possible output digits; the highest-probability digit is
# taken to be that time step's output
class DecoderTPRRNNB(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size, n_roles=2, n_fillers=10, filler_dim=10, role_dim=6, final_layer_width=None):
        super(DecoderTPRRNNB, self).__init__()
        self.hidden_size = hidden_size # Size of the hidden state
        self.output_size = output_size # Size of the output
        self.emb_size = emb_size
        self.rnn = nn.GRU(emb_size, hidden_size) # Recurrent unit
        #self.out = nn.Linear(hidden_size, output_size) # Linear layer giving the output
        self.softmax = nn.LogSoftmax() # Softmax layer
    
        self.n_roles = n_roles # number of roles
        self.n_fillers = n_fillers # number of fillers

        self.filler_dim = filler_dim
        self.role_dim = role_dim

        self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim)
        self.role_embedding = nn.Embedding(self.n_roles, self.role_dim)

        self.final_layer_width = final_layer_width
        if self.final_layer_width is None:
            self.has_last = 0
        else:
            self.has_last = 1
            self.last_layer = nn.Linear(self.final_layer_width, self.filler_dim * self.role_dim)

        self.expander = nn.Linear(self.filler_dim, self.hidden_size)

    # Forward pass
    def forward(self, hidden, output_len=None, tree=None, return_hidden=False, just_embs=False, role_list=None):
        outputs = []
        hidden = nn.ReLU()(hidden)
        hidden_states = [hidden]
        
        for item in range(output_len):
            #output, hidden = self.rnn(Variable(torch.zeros(1,hidden.size()[1],int(self.emb_size))).to(device=device), hidden) # Pass the inputs through the hidden layer
            output = hidden


            encoding = output
            if self.has_last:
                encoding = self.last_layer(encoding)
                
            encoding = encoding.transpose(0,1)
            encoding = encoding.view(-1,self.role_dim,self.filler_dim)

            role_list = torch.LongTensor([[0] for _ in range(len(encoding))]).to(device=device)
            roles_embedded = self.role_embedding(role_list)

            #print(roles_embedded.shape)
            #print(encoding.shape)
            #14/0
            filler_guess = torch.bmm(roles_embedded, encoding)
            listembs = [self.filler_embedding(torch.LongTensor([[i]]).to(device=device)) for i in range(self.n_fillers)]

            embmat = torch.cat(listembs, 1)

            fillers_exp = filler_guess.unsqueeze(3).expand(filler_guess.shape[0],-1,self.filler_dim,self.n_fillers).transpose(2,3)
            embmat_exp = embmat.expand(filler_guess.shape[0],-1,self.n_fillers,self.filler_dim)
            logs = nn.LogSoftmax(dim=2)(torch.sum(torch.pow(fillers_exp - embmat_exp, 2), dim=3)).transpose(0,2).transpose(0,1)
            #return logs.transpose(1,2)

            role_list = torch.LongTensor([[1] for _ in range(len(encoding))]).to(device=device)
            roles_embedded = self.role_embedding(role_list)

            #print(roles_embedded.shape)
            #print(encoding.shape)
            #14/0
            filler_guess = torch.bmm(roles_embedded, encoding)
            hidden = self.expander(filler_guess)

            #invembmat = torch.pinverse(embmat)
            
      
#            output = self.softmax(self.out(output[0])) # Pass the result through softmax to make it probabilities


            outputs.append(logs.transpose(1,2).squeeze(0))
            hidden_states.append(hidden)



        #print(outputs)
        #print(outputs[0].shape)
        #14/0

        if return_hidden:
            return outputs, hidden_states
        else:
            return outputs
             
