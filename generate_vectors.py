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
from print_read_functions import *


# Given a trained model, generate encodings for sequences from that model

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", help="prefix for your training/dev data", type=str, default="digits")
parser.add_argument("--encoder", help="encoder type", type=str, default="ltr")
parser.add_argument("--decoder", help="decoder type", type=str, default="ltr")
parser.add_argument("--task", help="training task", type=str, default="auto")
parser.add_argument("--vocab_size", help="vocab size for the training language", type=int, default=10)
parser.add_argument("--emb_size", help="embedding size", type=int, default=10)
parser.add_argument("--hidden_size", help="hidden size", type=int, default=60)
parser.add_argument("--enc_prefix", help="prefix for the trained encoder", type=str, default=None)
parser.add_argument("--dec_prefix", help="prefix for the trained decoder", type=str, default=None)
parser.add_argument("--n_hidden_enc", help="number of hidden layers for an MLP encoder", type=int, default=None)
parser.add_argument("--n_hidden_dec", help="number of hidden layers for an MLP decoder", type=int, default=None)
parser.add_argument("--enc_role_scheme", help="role scheme for a TPR encoder", type=str, default=None)
parser.add_argument("--dec_role_scheme", help="role scheme for a TPR decoder", type=str, default=None)
parser.add_argument("--max_length", help="maximum length of a sequence", type=int, default=6)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
    thisdev = 'cuda'
else:
    device = torch.device('cpu')
    thisdev = 'cpu'


fi_train = open('data/' + args.prefix + '.train', 'r')
fi_dev = open('data/' + args.prefix + '.dev', 'r')
fi_test = open('data/' + args.prefix + '.test', 'r')

# Load the data sets
train_set = file_to_lists(fi_train)
dev_set = file_to_lists(fi_dev)
test_set = file_to_lists(fi_test)

input_to_output = lambda sequence: transform(sequence, args.task)

layers_prefix = ""
# Load the models
if args.encoder == "ltr":
        encoder = EncoderRNN(args.vocab_size, args.emb_size, args.hidden_size)
elif args.encoder == "bi":
        encoder = EncoderBiRNN(args.vocab_size, args.emb_size, args.hidden_size)
elif args.encoder == "tree":
        encoder = EncoderTreeRNN(args.vocab_size, args.emb_size, args.hidden_size)
elif args.encoder == "mlp":
        encoder = MLPEncoder(args.hidden_size,args.hidden_size,args.n_hidden_enc,args.hidden_size)
        layers_prefix = str(args.n_hidden_enc) + "enclayers_" + layers_prefix
elif args.encoder == "tpr":
	encoder = TensorProductEncoder(n_fillers=args.vocab_size, filler_dim=args.emb_size, role_dim=args.emb_size, final_layer_width=args.hidden_size, role_scheme=args.enc_role_scheme, max_length=args.max_length)
else:
        print("Invalid encoder type")

parsing_fn = lambda x: None
role_fn = lambda x: None
if args.decoder == "ltr":
        decoder = DecoderRNN(args.vocab_size, args.emb_size, args.hidden_size)
elif args.decoder == "bi":
	decoder = DecoderBiRNN(args.vocab_size, args.emb_size, args.hidden_size)
elif args.decoder == "tree":
        decoder = DecoderTreeRNN(args.vocab_size, args.emb_size, args.hidden_size)
        parsing_fn = lambda x: [parse_digits(elt) for elt in x]	
elif args.decoder == "mlp":
	# This is a dummy decoder
        decoder = MLPDecoder(args.hidden_size,args.hidden_size,args.n_hidden_dec,args.hidden_size)
        layers_prefix = str(args.n_hidden_dec) + "declayers_" + layers_prefix
elif args.decoder == "tpr":
	n_r, role_fn_a = create_role_scheme(args.dec_role_scheme, args.max_length, args.vocab_size)
	role_fn = lambda x: [role_fn_a(elt) for elt in x]
	decoder = TensorProductDecoder(n_roles=n_r, n_fillers=args.vocab_size, filler_dim=args.emb_size, role_dim=args.emb_size, final_layer_width=args.hidden_size)
else:
        print("Invalid decoder type")

encoder = encoder.to(device=device)
decoder = decoder.to(device=device)

print(torch.cuda.is_available())
print(device)
print(thisdev)

encoder.load_state_dict(torch.load("models/encoder_" + args.enc_prefix + ".weights", map_location=thisdev))
decoder.load_state_dict(torch.load("models/decoder_" + args.dec_prefix + ".weights", map_location=thisdev))

# Save all encodings to a file
fo_train = open("data/" + args.enc_prefix + ".data_from_train", "w")
fo_dev = open("data/" + args.enc_prefix + ".data_from_dev", "w")
fo_test = open("data/" + args.enc_prefix + ".data_from_test", "w")


# To be populated with 2-tuples of the form (sequence, encoding), where each
# sequence is drawn from the test set
accurate = 0
total = 0

# This loop populates the data_from_tes list
# It also records performance on the test set 
# and prints any incorrect predictions
for example in test_set:
	pred, encoding = evaluate(encoder, decoder, example, input_to_output, parsing_fn, role_fn)
 
	sequence = [str(x) for x in example]
	enc = [str(x) for x in encoding.data.cpu().numpy()[0][0]]

	fo_test.write(" ".join(sequence) + "\t" + " ".join(enc) + "\n")

                
	if tuple(input_to_output(example)) == tuple(pred):
		accurate += 1
	else:
		# For each incorrect prediction, predict that input and
		# the incorrect predicted output
		#print(example, input_to_output(example), tuple(pred))
		pass
	total += 1

#corrcount, totalcount = score(encoder, decoder, test_set, input_to_output)
#print(corrcount, totalcount) 
print(args.enc_prefix, args.dec_prefix, "test_acc", accurate, total)

# To be populated with 2-tuples of the form (sequence, encoding), where each
# sequence is drawn from the test set

accurate = 0
total = 0

# This loop populates the ata_from_test list.
# It also records performance on the test set 
# and prints any incorrect predictions
for example in dev_set:
	pred, encoding = evaluate(encoder, decoder, example, input_to_output, parsing_fn, role_fn)
 

	sequence = [str(x) for x in example]
	enc = [str(x) for x in encoding.data.cpu().numpy()[0][0]]

	fo_dev.write(" ".join(sequence) + "\t" + " ".join(enc) + "\n")

                
	if tuple(input_to_output(example)) == tuple(pred):
		accurate += 1
	else:
		# For each incorrect prediction, predict that input and
		# the incorrect predicted output
		#print(example, tuple(pred))
		pass
	total += 1
        
print(args.enc_prefix, args.dec_prefix, "dev_acc", accurate, total)

# To be populated with 2-tuples of the form (sequence, encoding), where each
# sequence is drawn from the test set

accurate = 0
total = 0

# This loop populates the ata_from_test list.
# It also records performance on the test set 
# and prints any incorrect predictions
for example in train_set:
	pred, encoding = evaluate(encoder, decoder, example, input_to_output, parsing_fn, role_fn)

	sequence = [str(x) for x in example]
	enc = [str(x) for x in encoding.data.cpu().numpy()[0][0]]

	fo_train.write(" ".join(sequence) + "\t" + " ".join(enc) + "\n")


	if tuple(input_to_output(example)) == tuple(pred):
		accurate += 1
	else:
		# For each incorrect prediction, predict that input and	
		# the incorrect predicted output
		#print(example, tuple(pred))
		pass
	total += 1
        
print(args.enc_prefix, args.dec_prefix, "train_acc", accurate, total)





