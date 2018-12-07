# Functions needed for training models

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

from role_assignment_functions import *
from evaluation import *

use_cuda = torch.cuda.is_available()

# Train for a single batch
# Inputs: 
#   training_set: the batch
#   encoder: the encoder
#   decoder: the decoder
#   encoder_optimizer: optimizer for the encoder
#   decoder_optimizer: optimizer for the decoder
#   criterion: the loss function
#   input_to_output: function that maps input sequences to correct outputs
def train(training_set, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, input_to_output):
    loss = 0
    
    # Get the decoder's outputs outputs for these inputs
    logits = decoder(encoder(training_set), 
                     len(training_set[0]), 
                     [parse_digits(elt) for elt in training_set])
        
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
        
    # Compute the loss over each index in the output
    for index, logit in enumerate(logits):
        if use_cuda:
            loss += criterion(logit, Variable(torch.LongTensor([[input_to_output(x) for x in training_set]])).cuda().transpose(0,2)[index].view(-1))
        else:
            loss += criterion(logit, Variable(torch.LongTensor([[input_to_output(x) for x in training_set]])).transpose(0,2)[index].view(-1))

    # Backpropagate the loss
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
        
    return loss / len(training_sets)
        
# Compute the loss on the development set
# Inputs:
#    encoder: the encoder
#    decoder: the decoder
#    criterion: the loss function
#    dev_set: the development set
#    input_to_output: function that maps input sequences to correct outputs
def dev_loss(encoder, decoder, criterion, dev_set, input_to_output):
    dev_loss_val = 0
   
    for dev_elt in dev_set:
        logits = decoder(encoder(dev_elt), 
                       len(dev_elt[0]),
                         [parse_digits(elt) for elt in dev_elt])
        
        for index, logit in enumerate(logits):
                    if use_cuda: 
                        dev_loss_val += criterion(logit, Variable(torch.LongTensor([[input_to_output(x) for x in dev_elt]])).cuda().transpose(0,2)[index].view(-1))
                    else:
                        dev_loss_val += criterion(logit, Variable(torch.LongTensor([[input_to_output(x) for x in dev_elt]])).transpose(0,2)[index].view(-1))
                
    return dev_loss_val / len(dev_set)

# Generate batches from a data set
def batchify(data, batch_size):
	length_sorted_dict = {}
	max_length = 0
	
	for item in data:
		if len(item) not in length_sorted_dict:
			length_sorted_dict[len(item)] = []
		length_sorted_dict[len(item)].append(item)
		if len(item) > max_length:
			max_length = len(item)

	batches = []
	
	for seq_len in range(max_length + 1):
		if seq_len in length_sorted_dict:
			for batch_num in range(len(length_sorted_dict[seq_len])//batch_size):
				this_batch = length_sorted_dict[seq_len][batch_num*batch_size:(batch_num+1)*batch_size]
				batches.append(this_batch)

	shuffle(batches)
	return batches

# Generate batches suitable for a TPDN from some dataset
def batchify_tpr(data, batch_size):
	length_sorted_dict = {}
	max_length = 0
	
	for item in data:
		if len(item[0]) not in length_sorted_dict:
			length_sorted_dict[len(item[0])] = []
		length_sorted_dict[len(item[0])].append(item)
		if len(item[0]) > max_length:
			max_length = len(item[0])

	batches = []
	
	for seq_len in range(max_length + 1):
		if seq_len in length_sorted_dict:
			for batch_num in range(len(length_sorted_dict[seq_len])//batch_size):
				this_batch = length_sorted_dict[seq_len][batch_num*batch_size:(batch_num+1)*batch_size]
				batches.append(this_batch)


	shuffle(batches)
	return batches


# Perform a full training run for a digit
# sequence task. Inputs:
#    encoder: the encoder
#    decoder: the decoder
#    train_data: the training set
#    dev_data: the development set
#    file_prefix: file identifier to use when saving the weights
#    input_to_output: function for mapping input sequences to the correct outputs
#    max_epochs: maximum number of epochs to train for before halting
#    patience: maximum number of epochs to train without dev set improvement before halting
#    print_every: number of batches to go through before printing the current status
#    learning_rate: learning rate
#    batch_size: batch_size
def train_iters(encoder, decoder, train_data, dev_data, file_prefix, input_to_output, max_epochs=100, patience=1, print_every=1000, learning_rate=0.001, batch_size=32):
    print_loss_total = 0
    
    # Train using Adam
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    # Negative log likelihood loss
    criterion = nn.NLLLoss()
    best_loss = 1000000
    epochs_since_improved = 0

    # Group the data into batches
    training_sets = batchify(train_data, batch_size)
    dev_data = batchify(dev_data, batch_size)
    loss_total = 0

    # File for printing updates
    progress_file = open("models/progress_" + file_prefix, "w")

    # Iterate over epocjs
    for epoch in range(max_epochs):
        improved_this_epoch = 0
        shuffle(training_sets)

        # Iterate over batches
        for batch, training_set in enumerate(training_sets):
            
            # Train for this batch
            loss = train(training_set, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, input_to_output)            
            # Print an update and save the weights every print_every iterations
            if batch % print_every == 0:
                this_loss = dev_loss(encoder, decoder, criterion, dev_data, input_to_output)
                progress_file.write(str(epoch) + "\t" + str(batch) + "\t" + str(this_loss.item()) + "\n")
                if this_loss.data[0] < best_loss:
                    improved_this_epoch = 1
                    best_loss = this_loss
                    torch.save(encoder.state_dict(), "models/encoder_" + file_prefix + ".weights")
                    torch.save(decoder.state_dict(), "models/decoder_" + file_prefix + ".weights") 
                             
        # Early stopping
        if not improved_this_epoch:
            epochs_since_improved += 1
            if epochs_since_improved == patience:
                break 

        else:
            epochs_since_improved = 0        
            
    
            
# Training a TPDN for a single training example
def train_tpr(batch_set, tpr_encoder, tpr_optimizer, criterion):

    # Initialize the loss for this example at 0
    loss = 0
        
    # Zero the gradient 
    tpr_optimizer.zero_grad()
 
    # Iterate over this batch
    for training_set in batch_set:
        input_fillers = training_set[0] # The list of fillers for the input
        input_roles = training_set[1] # The list of roles hypothesized for the input
        target_variable = training_set[2] # The mystery vector associated with this input
     
        # Find the output for this input
        tpr_encoder_output = tpr_encoder(input_fillers, input_roles)

        # Find the loss associated with this output
        #loss += criterion(tpr_encoder_output.unsqueeze(0), target_variable)
        loss += criterion(tpr_encoder_output, target_variable)
      
    # Backpropagate the loss
    loss.backward()
    tpr_optimizer.step()
    tpr_encoder_output = tpr_encoder_output.detach()
    
    # Return the loss
    return loss.data.item()

# Training a TPDN for multiple iterations
def trainIters_tpr(train_data, dev_data, tpr_encoder, n_epochs, print_every=1000, learning_rate=0.001, batch_size=5, weight_file=None, patience=10):
    # The amount of loss accumulated between print updates
    print_loss_total = 0
    
    # The optimization algorithm; could use SGD instead of Adam
    tpr_optimizer = optim.Adam(tpr_encoder.parameters(), lr=learning_rate)
   
    # Using mean squared error as the loss
    criterion = nn.MSELoss()
    prev_loss = 1000000
    count_not_improved = 0
    count_unhelpful_cuts = 0
    training_done = 0
    best_loss = prev_loss    

    # Format the data
    train_data = batchify_tpr(train_data, batch_size)

    train_ex = train_data[0]
    
    training_sets = [(Variable(torch.LongTensor([item[0] for item in batch])).cuda(), 
                     Variable(torch.LongTensor([item[1] for item in batch])).cuda(),
                     torch.cat([item[2].unsqueeze(0).unsqueeze(0) for item in batch], 1)) for batch in train_data]

    dev_data = batchify_tpr(dev_data, batch_size)
    dev_data_sets = [(Variable(torch.LongTensor([item[0] for item in batch])).cuda(),
                     Variable(torch.LongTensor([item[1] for item in batch])).cuda(),
                     torch.cat([item[2].unsqueeze(0).unsqueeze(0) for item in batch], 1)) for batch in dev_data]
    
    # Conduct the desired number of training examples
    for epoch in range(n_epochs):
        improved_this_epoch = 0

        shuffle(training_sets)

        for batch in range(len(training_sets)//batch_size):
            batch_set = training_sets[batch * batch_size:(batch + 1)*batch_size]

            loss = train_tpr(batch_set, tpr_encoder, tpr_optimizer, criterion)       
        
            # If relevant, print the average loss over the last print_every iterations
            if batch % print_every == 0:
                total_mse = 0
            
                for i in range(len(dev_data_sets)):  
                    total_mse += torch.mean(torch.pow(tpr_encoder(dev_data_sets[i][0], dev_data_sets[i][1]).data - dev_data_sets[i][2].data, 2))
    
                print_loss = total_mse / len(dev_data)
                print(print_loss.item())
                if print_loss < best_loss:
                    improved_this_epoch = 1        
                    count_not_improved = 0
                    best_loss = print_loss
                    torch.save(tpr_encoder.state_dict(), weight_file)
                else:
                    count_not_improved += 1
                    if count_not_improved == patience: 
                        training_done = 1
                        break
      
        if training_done:
            break        
        if not improved_this_epoch:
            break 

           
    return best_loss



