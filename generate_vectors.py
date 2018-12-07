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

import argparse


import sys
import os

import time
import math

import pickle

from tasks import *
from training import *
from models import *
from evaluation import *
from role_assignment_functions import *


# Given a trained model, generate encodings for sequences from that model

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", help="prefix for your training/dev data", type=str, default="digits")
parser.add_argument("--encoder", help="encoder type", type=str, default="ltr")
parser.add_argument("--decoder", help="decoder type", type=str, default="ltr")
parser.add_argument("--task", help="training task", type=str, default="auto")
parser.add_argument("--vocab_size", help="vocab size for the training language", type=int, default=10)
parser.add_argument("--emb_size", help="embedding size", type=int, default=10)
parser.add_argument("--hidden_size", help="hidden size", type=int, default=60)
parser.add_argument("--model_prefix", help="prefix for the trained model", type=str, default=None)
args = parser.parse_args()


use_cuda = torch.cuda.is_available()


# Load data
with open('data/' + args.prefix + '.train.pkl', 'rb') as handle:
    train_set = pickle.load(handle)

with open('data/' + args.prefix + '.dev.pkl', 'rb') as handle:
    dev_set = pickle.load(handle)

with open('data/' + args.prefix + '.test.pkl', 'rb') as handle:
    test_set = pickle.load(handle)

input_to_output = lambda sequence: transform(sequence, args.task)

# Load the models
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

encoder.load_state_dict(torch.load("models/encoder_" + args.model_prefix + ".weights"))
decoder.load_state_dict(torch.load("models/decoder_" + args.model_prefix + ".weights"))


# To be populated with 2-tuples of the form (sequence, encoding), where each
# sequence is drawn from the test set
data_from_test = []
accurate = 0
total = 0

# This loop populates the data_from_test list
# It also records performance on the test set 
# and prints any incorrect predictions
for example in test_set:
	pred, encoding = evaluate(encoder, decoder, example, input_to_output)
	data_from_test.append((example, encoding))
                 
	if tuple(input_to_output(example)) == tuple(pred):
		accurate += 1
	else:
		# For each incorrect prediction, predict that input and
		# the incorrect predicted output
		#print(example, tuple(pred))
		pass
	total += 1
        
print(args.model_prefix, "test_acc", accurate, total)

# To be populated with 2-tuples of the form (sequence, encoding), where each
# sequence is drawn from the test set
data_from_dev = []

accurate = 0
total = 0

# This loop populates the data_from_test list.
# It also records performance on the test set 
# and prints any incorrect predictions
for example in dev_set:
	pred, encoding = evaluate(encoder, decoder, example, input_to_output)
	data_from_dev.append((example, encoding))
                 
	if tuple(input_to_output(example)) == tuple(pred):
		accurate += 1
	else:
		# For each incorrect prediction, predict that input and
		# the incorrect predicted output
		#print(example, tuple(pred))
		pass
	total += 1
        
print(args.model_prefix, "dev_acc", accurate, total)

# To be populated with 2-tuples of the form (sequence, encoding), where each
# sequence is drawn from the test set
data_from_train = []

accurate = 0
total = 0

# This loop populates the data_from_test list.
# It also records performance on the test set 
# and prints any incorrect predictions
for example in train_set:
	pred, encoding = evaluate(encoder, decoder, example, input_to_output)
	data_from_train.append((example, encoding))
                 
	if tuple(input_to_output(example)) == tuple(pred):
		accurate += 1
	else:
		# For each incorrect prediction, predict that input and	
		# the incorrect predicted output
		#print(example, tuple(pred))
		pass
	total += 1
        
print(args.model_prefix, "train_acc", accurate, total)

# Save all encodings to a file
fo_train = open("data/" + args.model_prefix + ".data_from_train", "w")
fo_dev = open("data/" + args.model_prefix + ".data_from_dev", "w")
fo_test = open("data/" + args.model_prefix + ".data_from_test", "w")

for training_item in data_from_train:
	sequence = training_item[0]
	encoding = training_item[1].data.cpu().numpy()[0][0]

	sequence = [str(x) for x in sequence]
	encoding = [str(x) for x in encoding]

	fo_train.write(" ".join(sequence) + "\t" + " ".join(encoding) + "\n")

for dev_item in data_from_dev:
	sequence = dev_item[0]
	encoding = dev_item[1].data.cpu().numpy()[0][0]

	sequence = [str(x) for x in sequence]
	encoding = [str(x) for x in encoding]

	fo_dev.write(" ".join(sequence) + "\t" + " ".join(encoding) + "\n")


for test_item in data_from_test:
	sequence = test_item[0]
	encoding = test_item[1].data.cpu().numpy()[0][0]

	sequence = [str(x) for x in sequence]
	encoding = [str(x) for x in encoding]

	fo_test.write(" ".join(sequence) + "\t" + " ".join(encoding) + "\n")









