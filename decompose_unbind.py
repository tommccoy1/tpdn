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

import argparse

from tasks import *
from training import *
from models import *
from evaluation import *
from role_assignment_functions import *
from print_read_functions import *

import numpy as np

# Code for performing a tensor product decomposition on an
# existing set of vectors

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument("--data_prefix", help="prefix for the vectors", type=str, default=None)
parser.add_argument("--role_prefix", help="prefix for a file of roles (if used)", type=str, default=None)
parser.add_argument("--role_scheme", help="pre-coded role scheme to use", type=str, default=None)
parser.add_argument("--test_decoder", help="whether to test the decoder (in addition to MSE", type=str, default="False")
parser.add_argument("--decoder", help="decoder type", type=str, default="ltr")
parser.add_argument("--decoder_prefix", help="prefix for the decoder to test", type=str, default=None)
parser.add_argument("--decoder_embedding_size", help="embedding size for decoder", type=int, default=20)
parser.add_argument("--decoder_task", help="task performed by the decoder", type=str, default="auto")
parser.add_argument("--filler_dim", help="embedding dimension for fillers", type=int, default=10)
parser.add_argument("--role_dim", help="embedding dimension for roles", type=int, default=6)
parser.add_argument("--vocab_size", help="vocab size for the training language", type=int, default=10)
parser.add_argument("--hidden_size", help="size of the encodings", type=int, default=60)
parser.add_argument("--embedding_file", help="file containing pretrained embeddings", type=str, default=None)
parser.add_argument("--unseen_words", help="if using pretrained embeddings: whether to use all zeroes for unseen words' embeddings, or to give them random vectors", type=str, default="random")
parser.add_argument("--extra_test_set", help="additional file to print predictions for", type=str, default=None)
parser.add_argument("--train", help="whether or not to train the model", type=str, default="True")
parser.add_argument("--digits", help="whether this is one of the digit task", type=str, default="True")
parser.add_argument("--final_linear", help="whether to have a final linear layer", type=str, default="True")
parser.add_argument("--embed_squeeze", help="original dimension to be squeezed to filler_dim", type=int, default=None)
args = parser.parse_args()

# Create the logfile
if args.final_linear != "True":
	results_page = open("logs/" + args.data_prefix.split("/")[-1] + str(args.role_prefix).split("/")[-1] + str(args.role_scheme) + ".filler" + str(args.filler_dim) + ".role" + str(args.role_dim) + ".tpr_decomp.nf", "w")
else:
	results_page = open("logs/" + args.data_prefix.split("/")[-1] + str(args.role_prefix).split("/")[-1] + str(args.role_scheme) + ".filler" + str(args.filler_dim) + ".role" + str(args.role_dim) + "." + str(args.embed_squeeze) + ".tpr_decomp", "w")


# Prepare the train, dev, and test data
train_file = open("data/" + args.data_prefix + ".data_from_train", "r")
dev_file = open("data/" + args.data_prefix + ".data_from_dev", "r")
test_file = open("data/" + args.data_prefix + ".data_from_test", "r")

filler2index = {}
index2filler = {}
vocab_size = 0
max_length = 0

if args.digits == "True":
        for i in range(10):
                filler2index[str(i)] = i
                index2filler[i] = str(i)
                vocab_size = 10

train_set, filler2index, index2filler, vocab_size, max_length = file_to_integer_vector_lists(train_file, filler2index, index2filler, vocab_size, max_length)
dev_set, filler2index, index2filler, vocab_size, max_length = file_to_integer_vector_lists(dev_file, filler2index, index2filler, vocab_size, max_length)
test_set, filler2index, index2filler, vocab_size, max_length = file_to_integer_vector_lists(test_file, filler2index, index2filler, vocab_size, max_length)

train_file.close()
dev_file.close()
test_file.close()

if args.extra_test_set is not None:
        extra_file = open("data/" + args.extra_test_set, "r")
        extra_set, filler2index, index2filler, vocab_size, max_length = file_to_integer_vector_lists(extra_file, filler2index, index2filler, vocab_size, max_length)
        extra_file.close()

if args.digits == "True":
        for i in range(10):
                filler2index[str(i)] = i
                index2filler[i] = str(i)



# If there is a file of roles for the fillers, load those roles
filler_filea = None
filler_fileb = None
filler_filec = None
filler_filed = None
role_filea = None
role_fileb = None
role_filec = None
role_filed  = None

if args.role_prefix is not None:
        filler_filea = open("data/" + args.data_prefix + ".data_from_train", "r")
        filler_fileb = open("data/" + args.data_prefix + ".data_from_dev", "r")
        filler_filec = open("data/" + args.data_prefix + ".data_from_test", "r")

        role_filea = open("data/" + args.role_prefix + ".data_from_train.roles", "r")
        role_fileb = open("data/" + args.role_prefix + ".data_from_dev.roles", "r")
        role_filec = open("data/" + args.role_prefix + ".data_from_test.roles", "r")

        if args.extra_test_set is not None:
                filler_filed = open("data/" + args.extra_test_set, "r")
                role_filed = open("data/" + args.extra_test_set + ".roles", "r")

train_batches = batchify_tpr(all_train_data, 32)
dev_batches = batchify_tpr(all_dev_data, 32)
test_batches = batchify_tpr(all_test_data, 1)


weights_matrix = None

# Prepare the embeddings
# If a file of embeddings was provided, use those.
embedding_dict = None
if args.embedding_file is not None:
        weights_matrix = create_embedding_dictionary(args.embedding_file, args.filler_dim, filler_to_index, index_to_filler, emb_squeeze=args.embed_squeeze, unseen_words=args.unseen_words)


# Initialize the TPDN
if n_r != -1:
	role_counter = n_r

if args.final_linear == "True":
	tpr_decoder = TensorProductDecoder(n_roles=role_counter, n_fillers=filler_counter, filler_dim=args.filler_dim, role_dim=args.role_dim, final_layer_width=args.hidden_size)
else:
	tpr_decoder = TensorProductDecoder(n_roles=role_counter, n_fillers=filler_counter, filler_dim=args.filler_dim, role_dim=args.role_dim, final_layer_width=args.hidden_size)

if use_cuda:
	tpr_decoder = tpr_decoder.cuda()



# Define the architecture
if args.decoder == "ltr":
        encoder = EncoderRNN(filler_counter, args.filler_dim, args.hidden_size)
elif args.decoder == "bi":
        encoder = EncoderBiRNN(filler_counter, args.filler_dim, args.hidden_size)
elif args.decoder == "tree":
        encoder = EncoderTreeRNN(filler_counter, args.filler_dim, args.hidden_size)
else:
        print("Invalid encoder type")

if args.decoder == "ltr":
        decoder = DecoderRNN(filler_counter, args.filler_dim, args.hidden_size)
elif args.decoder == "bi":
        decoder = DecoderBiRNN(filler_counter, args.filler_dim, args.hidden_size)
elif args.decoder == "tree":
        decoder = DecoderTreeRNN(filler_counter, args.filler_dim, args.hidden_size)
else:
        print("Invalid decoder type")


encoder.load_state_dict(torch.load("models/encoder_" + args.decoder_prefix + ".weights"))
#CHINCHILLA
decoder.load_state_dict(torch.load("models/decoder_" + args.decoder_prefix + ".weights"))

if use_cuda:
	encoder = encoder.cuda()
	decoder = decoder.cuda()

args.data_prefix = args.data_prefix.split("/")[-1] + ".filler" + str(args.filler_dim) + ".role" + str(args.role_dim)
if args.final_linear != "True":
	args.data_prefix += ".no_final"

#fo_probe = open("fo6_chinchilla.txt", "w")

# Train the TPDN
args.role_prefix = str(args.role_prefix).split("/")[-1]
if args.train == "True":
	end_loss = trainIters_tpr_unbind(training_sets, dev_data_sets, tpr_decoder, encoder, 100, input_to_output, print_every=1000//32, learning_rate = 0.001, weight_file="models/" + args.data_prefix + str(args.role_prefix) + str(args.role_scheme) + ".tpr")

# Load the trained TPDn
tpr_decoder.load_state_dict(torch.load("models/" + args.data_prefix + str(args.role_prefix) + str(args.role_scheme) + ".tpr"))

#fo_probe = open("fo7_chinchilla.txt", "w")


correct = 0
total = 0

for elt in test_data_sets:
    
    preds = tpr_decoder.predict(encoder(list(list(x) for x in elt[0].cpu().numpy())), elt[1], encoder).detach()
    real = elt[0]
    
    #print(preds[0])
    
    for i in range(len(preds)):
        paira = list(preds[i].cpu().numpy())
        pairb = list(real[i].cpu().numpy())
        
        if paira == pairb:
            correct += 1
        else:
            pass
            #print(paira, pairb)
            
        total += 1
        
print("ACC:", correct * 1.0/total)

#fo_probe = open("fo8_chinchilla.txt", "w")
results_page.write(args.data_prefix + str(args.role_prefix) + str(args.role_scheme) + ".tpr" + " Swapping encoder performance: " + str(correct) + " " +  str(total) + "\n")





