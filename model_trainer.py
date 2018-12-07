from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

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

# Code for training a seq2seq RNN on a digit transformation task

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", help="prefix for your training/dev data", type=str, default="digits")
parser.add_argument("--encoder", help="encoder type", type=str, default="ltr")
parser.add_argument("--decoder", help="decoder type", type=str, default="ltr")
parser.add_argument("--task", help="training task", type=str, default="auto")
parser.add_argument("--vocab_size", help="vocab size for the training language", type=int, default=10)
parser.add_argument("--emb_size", help="embedding size", type=int, default=10)
parser.add_argument("--hidden_size", help="hidden size", type=int, default=60)
parser.add_argument("--generalization_prefix", help="prefix for generalization test set", type=str, default=None)
parser.add_argument("--initial_lr", help="initial learning rate", type=float, default=0.001)
parser.add_argument("--batch_size", help="batch size", type=int, default=32)
parser.add_argument("--train", help="whether to train the model or not", type=str, default="True")
parser.add_argument("--file_prefix", help="prefix of file to load and evaluate on", type=str, default=None)
args = parser.parse_args()


use_cuda = torch.cuda.is_available()

# Load the data sets
with open('data/' + args.prefix + '.train.pkl', 'rb') as handle:
    train_set = pickle.load(handle)

with open('data/' + args.prefix + '.dev.pkl', 'rb') as handle:
    dev_set = pickle.load(handle)

with open('data/' + args.prefix + '.test.pkl', 'rb') as handle:
    test_set = pickle.load(handle)

if args.generalization_prefix is not None:
	with open('data/' + args.generalization_prefix + '.test.pkl', 'rb') as handle:
		generalization_set = pickle.load(handle)

# Define the training function
input_to_output = lambda sequence: transform(sequence, args.task)


# Define the architecture
if args.encoder == "ltr":
	encoder = EncoderRNN(args.vocab_size, args.emb_size, args.hidden_size)
elif args.encoder == "bi":
	encoder = EncoderBiRNN(args.vocab_size, args.emb_size, args.hidden_size)
elif args.encoder == "tree":
	encoder = EncoderTreeRNN(args.vocab_size, args.emb_size, args.hidden_size)
else:
	print("Invalid encoder type") 

if args.decoder == "ltr":
	decoder = DecoderRNN(args.vocab_size, args.emb_size, args.hidden_size)
elif args.decoder == "bi":
	decoder = DecoderBiRNN(args.vocab_size, args.emb_size, args.hidden_size)
elif args.decoder == "tree":
	decoder = DecoderTreeRNN(args.vocab_size, args.emb_size, args.hidden_size)
else:
	print("Invalid decoder type")

if use_cuda:
	encoder = encoder.cuda()
	decoder = decoder.cuda()

# Set the prefix for saving weights
file_prefix = args.encoder + "_" + args.decoder + "_" + args.task + "_"
directories = os.listdir("./models")
found = False
suffix = 0
while not found:
	if "encoder_" + file_prefix + str(suffix) + ".weights" not in directories:
		found = 1
	else:
		suffix += 1

suffix = str(suffix)

# Train the model
if args.train == "True":
	file_prefix = file_prefix + suffix

	train_iters(encoder, decoder, train_set, dev_set, file_prefix, input_to_output, max_epochs=100, patience=1, print_every=10000//32, learning_rate=0.001, batch_size=args.batch_size)            
else:
	file_prefix = args.file_prefix

# Evaluate the trained model
encoder.load_state_dict(torch.load("models/encoder_" + file_prefix + ".weights"))
decoder.load_state_dict(torch.load("models/decoder_" + file_prefix + ".weights"))


report_file = open("models/results_" + file_prefix + ".txt", "w")            
correct, total = score(encoder, decoder, batchify(test_set, 1), input_to_output)
report_file.write("Test set results:\nCorrect:\t" + str(correct) + "\nTotal:\t" + str(total) + "\nAccuracy:\t" + str(correct * 1.0 / total) + "\n\n")


if args.generalization_prefix is not None:
	correct, total = score(encoder, decoder, batchify(generalization_set, 1), input_to_output)
	report_file.write("Generalization set results:\nCorrect:\t" + str(correct) + "\nTotal:\t" + str(total) + "\nAccuracy:\t" + str(correct * 1.0 / total) + "\n\n")






