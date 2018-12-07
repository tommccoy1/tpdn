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

import pickle

from binding_operations import *
from role_assignment_functions import *

# Definitions of all the seq2seq models and the TPDN

use_cuda = torch.cuda.is_available()

# Encoder RNN for the mystery vector generating network--unidirectional GRU
class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size # Hidden size
        self.embedding = nn.Embedding(input_size, emb_size) # Embedding layer
        self.rnn = nn.GRU(emb_size, hidden_size) # Recurrent layer
     
    # A forward pass of the encoder
    def forward(self, sequence):
        hidden = self.init_hidden(len(sequence))
        batch_size = len(sequence)

 
        sequence = Variable(torch.LongTensor([sequence])).transpose(0,2)#.cuda()
        if use_cuda:
            sequence = sequence.cuda()

        for element in sequence:
            if use_cuda:
                embedded = self.embedding(element).transpose(0,1)
            else:
                embedded = self.embedding(element).transpose(0,1)
            output, hidden = self.rnn(embedded, hidden)
            
        return hidden
    
    # Initialize the hidden state as all zeroes
    def init_hidden(self, batch_size):
        result = Variable(torch.zeros(1,batch_size,self.hidden_size))

        if use_cuda:
            return result.cuda()
        else:
            return result

# Encoder RNN for the mystery vector generating network--bidirectional GRU
class EncoderBiRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size):
        super(EncoderBiRNN, self).__init__()
        self.hidden_size = hidden_size # Hidden size
        self.embedding = nn.Embedding(input_size, emb_size) # Embedding layer
        self.rnn_fwd = nn.GRU(emb_size, int(hidden_size/2)) # Recurrent layer-forward
        self.rnn_rev = nn.GRU(emb_size, int(hidden_size/2)) # Recurrent layer-backward
     
    # A forward pass of the encoder
    def forward(self, sequence):
        batch_size = len(sequence)

        sequence_rev = Variable(torch.LongTensor([sequence[::-1]])).transpose(0,2)
        if use_cuda:
            sequence_rev = sequence_rev.cuda()
        

        sequence = Variable(torch.LongTensor([sequence])).transpose(0,2)
        if use_cuda:
            sequence = sequence.cuda()

        # Forward pass
        hidden_fwd = self.init_hidden(batch_size)
        
        for element in sequence:
            embedded = self.embedding(element).transpose(0,1)
            output, hidden_fwd = self.rnn_fwd(embedded, hidden_fwd)
            
        # Backward pass
        hidden_rev = self.init_hidden(batch_size)
        
        for element in sequence_rev:
            embedded = self.embedding(element).transpose(0,1)
            output, hidden_rev = self.rnn_rev(embedded, hidden_rev)
            
        # Concatenate the two hidden representations
        hidden = torch.cat((hidden_fwd, hidden_rev), 2)
            
        return hidden
    
    # Initialize the hidden state as all zeroes
    def init_hidden(self, batch_size):
        result = Variable(torch.zeros(1,batch_size,int(self.hidden_size/2)))
        
        if use_cuda:
            return result.cuda()
        else:
            return result

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
    
    def forward(self, input_batch):
        final_output = None
        for input_seq in input_batch:
            tree = parse_digits(input_seq)
        
            embedded_seq = []
        
            for elt in input_seq:
                embedded_seq.append(self.embedding(Variable(torch.LongTensor([elt])).cuda()).unsqueeze(0))
            
            leaf_nodes = []
            for elt in embedded_seq:
                this_hidden = self.tree_gru(elt, self.init_hidden(), self.init_hidden())
                leaf_nodes.append(this_hidden)
            
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
                
                current_level = next_level
            if final_output is None:
                final_output = current_level[0][0].unsqueeze(0)
            else: 
                final_output = torch.cat((final_output, current_level[0][0].unsqueeze(0)),0)

        return final_output.transpose(0,1)
                                
    # Initialize the hidden state as all zeroes
    def init_hidden(self):
        result = Variable(torch.zeros(1,1,int(self.hidden_size)))
             
        if use_cuda:
            return result.cuda()
        else:
            return result


    # Initialize the word hidden state as all zeroes
    def init_word(self):
        result = Variable(torch.zeros(1,1,int(self.emb_size)))
         
        if use_cuda:
            return result.cuda()
        else:
            return result


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
    def forward(self, hidden, output_len, tree):
        outputs = []
        encoder_hidden = self.squeeze(F.relu(hidden))
        fwd_hiddens = []
        rev_hiddens = []

        fwd_hidden = encoder_hidden       
        for item in range(output_len):
            if use_cuda:
                output, fwd_hidden = self.rnn_fwd(Variable(torch.zeros(1,fwd_hidden.size()[1],int(self.emb_size))).cuda(), fwd_hidden) # Pass the inputs through the hidden layer
            else:
                output, fwd_hidden = self.rnn_fwd(Variable(torch.zeros(1,fwd_hidden.size()[1],int(self.emb_size))), fwd_hidden)
            fwd_hiddens.append(fwd_hidden)

        rev_hidden = encoder_hidden       
        for item in range(output_len):
            if use_cuda:
                output, rev_hidden = self.rnn_rev(Variable(torch.zeros(1,rev_hidden.size()[1],int(self.emb_size))).cuda(), rev_hidden) # Pass the inputs through the hidden layer
            else:
                output, rev_hidden = self.rnn_rev(Variable(torch.zeros(1,rev_hidden.size()[1],int(self.emb_size))), rev_hidden)
            rev_hiddens.append(rev_hidden)

        all_hiddens = zip(fwd_hiddens, rev_hiddens[::-1])

        for hidden_pair in all_hiddens:
            output = torch.cat((hidden_pair[0], hidden_pair[1]), 2)
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
        
    def forward(self, encoding_list, output_len, tree_list):
        words_out = []
        for encoding_mini, tree in zip(encoding_list.transpose(0,1), tree_list):
           
            encoding = encoding_mini.unsqueeze(0)
            tree_to_use = tree[::-1][1:]
        
            current_layer = [encoding]
        
            for layer in tree_to_use:
                next_layer = []
                for index, node in enumerate(layer):
                    if len(node) == 1:
                        next_layer.append(current_layer[index])
                    else:
                        output, left = self.left_child(Variable(torch.zeros(1,1,self.hidden_size)).cuda(), current_layer[index])
                        output, right = self.right_child(Variable(torch.zeros(1,1,self.hidden_size)).cuda(), current_layer[index])
                        next_layer.append(left)
                        next_layer.append(right)
                current_layer = next_layer
            
            
            if words_out == []:
                for elt in current_layer:
                    words_out.append(nn.LogSoftmax()(self.word_out(elt).view(-1).unsqueeze(0)))
            else:
                index = 0
                for elt in current_layer:
                    words_out[index] = torch.cat((words_out[index], nn.LogSoftmax()(self.word_out(elt).view(-1).unsqueeze(0))), 0)
                    index += 1

        return words_out

                    
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
    def forward(self, hidden, output_len, tree):
        outputs = []
        hidden = F.relu(hidden)
        
        for item in range(output_len):
            if use_cuda:
                output, hidden = self.rnn(Variable(torch.zeros(1,hidden.size()[1],int(self.emb_size))).cuda(), hidden) # Pass the inputs through the hidden layer
            else:
                output, hidden = self.rnn(Variable(torch.zeros(1,hidden.size()[1],int(self.emb_size))), hidden)
            output = self.softmax(self.out(output[0])) # Pass the result through softmax to make it probabilities
            outputs.append(output)
            
        return outputs
                    
# A tensor product encoder layer 
# Takes a list of fillers and a list of roles and returns an encoding
class TensorProductEncoder(nn.Module):
    def __init__(self, n_roles=2, n_fillers=2, filler_dim=3, role_dim=4, 
                 final_layer_width=None, pretrained_embeddings=None, embedder_squeeze=None, binder="tpr"):

        super(TensorProductEncoder, self).__init__()
        
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
                self.last_layer = nn.Linear(self.filler_dim * self.role_dim, self.final_layer_width)
            else:
                self.last_layer = nn.Linear(self.filler_dim, self.final_layer_width)
      
    # Function for a forward pass through this layer. Takes a list of fillers and 
    # a list of roles and returns an single vector encoding it.
    def forward(self, filler_list, role_list):
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
                  

                    
 
