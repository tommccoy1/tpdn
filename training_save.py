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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Train for a single batch
# Inputs: 
#   training_set: the batch
#   encoder: the encoder
#   decoder: the decoder
#   encoder_optimizer: optimizer for the encoder
#   decoder_optimizer: optimizer for the decoder
#   criterion: the loss function
#   input_to_output: function that maps input sequences to correct outputs
def train(training_set, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, input_to_output, parsing_fn, role_fn):
    loss = 0
   
    #print(training_set)
    #14/0
 
    # Get the decoder's outputs outputs for these inputs
    logits = decoder(encoder(training_set), 
                     output_len=len(training_set[0]), 
                     tree=parsing_fn(training_set),
                     role_list=role_fn(training_set))
        
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
        
    # Compute the loss over each index in the output
    for index, logit in enumerate(logits):
        if use_cuda:
            loss += criterion(logit, Variable(torch.LongTensor([[input_to_output(x) for x in training_set]])).cuda().transpose(0,2)[index].view(-1))
        else:
            #print(logit.shape)
            #print(Variable(torch.LongTensor([[input_to_output(x) for x in training_set]])).transpose(0,2)[index].view(-1).shape)
            #print("")
            loss += criterion(logit, Variable(torch.LongTensor([[input_to_output(x) for x in training_set]])).transpose(0,2)[index].view(-1))

    # Backpropagate the loss
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
        
    return loss / len(training_set)
        

# Compute the loss on the development set
# Inputs:
#    encoder: the encoder
#    decoder: the decoder
#    criterion: the loss function
#    dev_set: the development set
#    input_to_output: function that maps input sequences to correct outputs
def dev_loss(encoder, decoder, criterion, dev_set, input_to_output, parsing_fn, role_fn):
    dev_loss_val = 0
   
    for dev_elt in dev_set:
        logits = decoder(encoder(dev_elt), 
                         output_len=len(dev_elt[0]),
                         tree=parsing_fn(dev_elt), 
                         role_list=role_fn(dev_elt))
        
        for index, logit in enumerate(logits):
                    if use_cuda: 
                        dev_loss_val += criterion(logit, Variable(torch.LongTensor([[input_to_output(x) for x in dev_elt]])).cuda().transpose(0,2)[index].view(-1))
                    else:
                        dev_loss_val += criterion(logit, Variable(torch.LongTensor([[input_to_output(x) for x in dev_elt]])).transpose(0,2)[index].view(-1))
                
    return dev_loss_val / len(dev_set)


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
def train_iters(encoder, decoder, train_data, dev_data, file_prefix, input_to_output, max_epochs=100, patience=1, print_every=1000, learning_rate=0.001, batch_size=32, parsing_fn=None, role_fn=None, train_enc=True, train_dec=True, save_intermediate=False):
    print_loss_total = 0
    
    # Train using Adam
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    if not train_enc:
        for param in encoder.parameters():
            param.requires_grad = False
    
    if not train_dec:
        for param in decoder.parameters():
            param.requires_grad = False 


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
            loss = train(training_set, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, input_to_output, parsing_fn, role_fn)            
            # Print an update and save the weights every print_every iterations
            if batch % print_every == 0:
                this_loss = dev_loss(encoder, decoder, criterion, dev_data, input_to_output, parsing_fn, role_fn)
                progress_file.write(str(epoch) + "\t" + str(batch) + "\t" + str(this_loss.item()) + "\n")
                if this_loss.data[0] < best_loss:
                    improved_this_epoch = 1
                    best_loss = this_loss
                    if train_enc:
                        torch.save(encoder.state_dict(), "models/encoder_" + file_prefix + ".weights")
                        if save_intermediate:
                            torch.save(encoder.state_dict(), "models/encoder_" + file_prefix + "_" + str(epoch) + "_" + str(batch) + ".weights") 
                    if train_dec:
                        torch.save(decoder.state_dict(), "models/decoder_" + file_prefix + ".weights") 
                        if save_intermediate:
                            torch.save(decoder.state_dict(), "models/decoder_" + file_prefix + "_" + str(epoch) + "_" + str(batch) + ".weights")
                             
        # Early stopping
        if not improved_this_epoch:
            epochs_since_improved += 1
            if epochs_since_improved == patience:
                break 

        else:
            epochs_since_improved = 0        
            
 



  






# Generate batches from a data set
def batchify(data, batch_size, shuff=True):
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

	if shuff:
		shuffle(batches)
	else:
		random.seed(4)
		shuffle(batches)
	return batches

# Generate batches suitable for a TPDN from some dataset
def batchify_tpr(data, batch_size, shuff=True):
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
				this_batch = [[item[0] for item in this_batch], torch.cat([torch.FloatTensor(item[1]).unsqueeze(0).unsqueeze(0) for item in this_batch], 1).to(device=device)]
				#this_batch = [Variable(torch.LongTensor([item[0] for item in this_batch])), torch.cat([torch.FloatTensor(item[1]).unsqueeze(0).unsqueeze(0) for item in this_batch], 1).to(device=device)]
				batches.append(this_batch)

	if shuff:
		shuffle(batches)
	else:
		random.seed(4)
		shuffle(batches)
	return batches
   
            
# Training a TPDN for a single training example
def train_tpr(batch_set, tpr_encoder, tpr_optimizer, criterion):

    # Initialize the loss for this example at 0
    loss = 0
        
    # Zero the gradient 
    tpr_optimizer.zero_grad()
 
    # Iterate over this batch
    for training_set in batch_set:
        input_fillers = training_set[0] # The list of fillers for the input
        target_variable = training_set[1] # The mystery vector associated with this input

        #print(input_fillers)
        #print(input_roles)    
        #14/0  
        # Find the output for this input
        tpr_encoder_output = tpr_encoder(input_fillers)

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
def train_iters_tpr(train_data, dev_data, tpr_encoder, n_epochs, print_every=1000, learning_rate=0.001, batch_size=5, weight_file=None, patience=10):
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

    dev_data = batchify_tpr(dev_data, batch_size)
   
    
    # Conduct the desired number of training examples
    for epoch in range(n_epochs):
        improved_this_epoch = 0

        shuffle(train_data)

        #for batch in range(len(train_data)): #range(len(train_data)//batch_size):
        for batch in range(len(train_data)//batch_size):
            #batch_set = #train_data[batch] #train_data[batch * batch_size:(batch + 1)*batch_size]
            batch_set = train_data[batch * batch_size:(batch + 1)*batch_size]

            loss = train_tpr(batch_set, tpr_encoder, tpr_optimizer, criterion)       
        
            # If relevant, print the average loss over the last print_every iterations
            if batch % print_every == 0:
                total_mse = 0
            
                for i in range(len(dev_data)):  
                    total_mse += torch.mean(torch.pow(tpr_encoder(dev_data[i][0]).data - dev_data[i][1].data, 2))
    
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

 
# Training a TPDN for a single training example
def train_tpr_unbind(training_set, input_to_output, tpr_decoder, tpr_optimizer, criterion, role_fn=None):

    # Initialize the loss for this example at 0
    loss = 0

    # Zero the gradient 
    tpr_optimizer.zero_grad()

    # Iterate over this batch
    input_fillers = [list(x.data.cpu().numpy()) for x in training_set[0]] # The list of fillers for the input
    enc = training_set[1] # The mystery vector associated with this input


        
    logits = tpr_decoder(enc, role_list=role_fn(input_fillers))

    # Compute the loss over each index in the output
    for index, logit in enumerate(logits):
        if use_cuda:
            loss += criterion(logit, Variable(torch.LongTensor([[input_to_output(x) for x in training_set[0]]])).cuda().transpose(0,2)[index].view(-1))
        else:
            #print(logit.shape)
            #print(Variable(torch.LongTensor([[input_to_output(x) for x in training_set]])).transpose(0,2)[index].view(-1).shape)
            #print("")
            #print([input_to_output(x) for x in input_fillers])
            #print(torch.LongTensor([[input_to_output(x) for x in input_fillers]])) 
            loss += criterion(logit, Variable(torch.LongTensor([[input_to_output(x) for x in input_fillers]])).transpose(0,2)[index].view(-1))


    # Backpropagate the loss
    loss.backward()
    tpr_optimizer.step()
    #tpr_decoder_output = tpr_decoder_output.detach()

    # Return the loss
    return loss.data.item()

# Training a TPDN for multiple iterations
def train_iters_tpr_unbind(train_data, dev_data, tpr_decoder, n_epochs, input_to_output, print_every=1000, learning_rate=0.001, batch_size=5, weight_file=None, patience=10, role_fn=None):
    # The amount of loss accumulated between print updates
    print_loss_total = 0

    # The optimization algorithm; could use SGD instead of Adam
    tpr_optimizer = optim.Adam(tpr_decoder.parameters(), lr=learning_rate)

    # Using mean squared error as the loss
    criterion = nn.NLLLoss()
    prev_loss = 1000000
    count_not_improved = 0
    count_unhelpful_cuts = 0
    training_done = 0
    best_loss = prev_loss

    train_data = batchify_tpr(train_data, batch_size)

    dev_data = batchify_tpr(dev_data, batch_size)

    # Conduct the desired number of training examples
    for epoch in range(n_epochs):
        improved_this_epoch = 0

        shuffle(train_data)

        for batch in range(len(train_data)): #//batch_size):
            batch_set = train_data[batch]

            loss = train_tpr_unbind(batch_set, input_to_output, tpr_decoder, tpr_optimizer, criterion, role_fn=role_fn)

            # If relevant, print the average loss over the last print_every iterations
            if batch % print_every == 0:
                total_mse = 0

                #fo_probe = open("fo" + str(batch) + "batch_chinchilla.txt", "w")

                this_loss = dev_loss_unbind(tpr_decoder, criterion, dev_data, input_to_output, None, role_fn)


                #fo_probe = open("fo_" + str(total_mse) + "mse_chinchilla.txt", "w")


                print_loss = this_loss / len(dev_data)
                print(print_loss.item())
                if print_loss < best_loss:
                    improved_this_epoch = 1
                    count_not_improved = 0
                    best_loss = print_loss
                    torch.save(tpr_decoder.state_dict(), weight_file)
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


# Compute the loss on the development set
# Inputs:
#    encoder: the encoder
#    decoder: the decoder
#    criterion: the loss function
#    dev_set: the development set
#    input_to_output: function that maps input sequences to correct outputs
def dev_loss_unbind(decoder, criterion, dev_set, input_to_output, parsing_fn, role_fn):
    dev_loss_val = 0
   
    for dev_elt in dev_set:
        logits = decoder(dev_elt[1], 
                         output_len=len(dev_elt[0]),
                         tree=None, 
                         role_list=role_fn(dev_elt[0]))

        input_fillers = [list(x.data.cpu().numpy()) for x in dev_elt[0]]
        
        for index, logit in enumerate(logits):
                    if use_cuda: 
                        dev_loss_val += criterion(logit, Variable(torch.LongTensor([[input_to_output(x) for x in input_fillers]])).cuda().transpose(0,2)[index].view(-1))
                    else:
                        dev_loss_val += criterion(logit, Variable(torch.LongTensor([[input_to_output(x) for x in input_fillers]])).transpose(0,2)[index].view(-1))
                
    return dev_loss_val / len(dev_set)






