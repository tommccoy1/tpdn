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
from print_read_functions import *

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
parser.add_argument("--train_enc", help="whether to train the encoder", type=str, default="True")
parser.add_argument("--train_dec", help="whether to train the decoder", type=str, default="True")
parser.add_argument("--enc_file_prefix", help="prefix of encoder file to load and evaluate on", type=str, default=None)
parser.add_argument("--dec_file_prefix", help="prefix of decoder file to load and evaluate on", type=str, default=None)
parser.add_argument("--n_hidden_enc", help="number of hidden layers for an MLP encoder", type=int, default=None)
parser.add_argument("--n_hidden_dec", help="number of hidden layers for an MLP decoder", type=int, default=None)
parser.add_argument("--enc_role_scheme", help="role scheme for a TPR encoder", type=str, default=None)
parser.add_argument("--dec_role_scheme", help="role scheme for a TPR decoder", type=str, default=None)
parser.add_argument("--prefix_prefix", help="start of the file prefix", type=str, default="")
parser.add_argument("--max_length", help="maximum length of a sequence", type=int, default=6)
parser.add_argument("--save_intermediate", help="whether to save intermediate weight files, rather than just one at the end", type=str, default="False")
parser.add_argument("--random_init", help="whether to save the randomly initialized, untrained models", type=str, default="False")
parser.add_argument("--filter", help="whether to use the subtraction method of training", type=str, default="False")
parser.add_argument("--stages", help="whether to train in stages", type=str, default="False")
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

prefix_prefix = args.prefix_prefix

fi_train = open('data/' + args.prefix + '.train', 'r')
fi_dev = open('data/' + args.prefix + '.dev', 'r')
fi_test = open('data/' + args.prefix + '.test', 'r')

# Load the data sets
train_set = file_to_lists(fi_train)
dev_set = file_to_lists(fi_dev)
test_set = file_to_lists(fi_test)

if args.generalization_prefix is not None:
	fi_gen = open('data/' + args.generalization_prefix + '.gen', 'r')
	generalization_set = file_to_lists(fi_gen)


# Define the training function
input_to_output = lambda sequence: transform(sequence, args.task)

layers_prefix = ""
# Define the architecture
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
elif args.decoder == "bidouble":
        decoder = DecoderBiDoubleRNN(args.vocab_size, args.emb_size, args.hidden_size)
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
	#print(n_r)
	decoder = TensorProductDecoder(n_roles=n_r, n_fillers=args.vocab_size, filler_dim=args.emb_size, role_dim=args.emb_size, final_layer_width=args.hidden_size)
elif args.decoder == "tprrnn":
        decoder = DecoderTPRRNN(args.vocab_size, args.emb_size, args.hidden_size, n_roles=1, n_fillers=10, filler_dim=10, role_dim=6)
elif args.decoder == "tprrnnb":
        decoder = DecoderTPRRNNB(args.vocab_size, args.emb_size, args.hidden_size, n_roles=2, n_fillers=10, filler_dim=10, role_dim=6)
else:
	print("Invalid decoder type")

encoder = encoder.to(device=device)
decoder = decoder.to(device=device)

if args.encoder == "tpr":
    args.encoder = args.encoder + "_" + args.enc_role_scheme
if args.decoder == "tpr":
    args.decoder = args.decoder + "_" + args.dec_role_scheme


if args.random_init == "True":
	prefix_prefix = prefix_prefix + "notrain_"
else:
	if args.train_enc != "True":
		prefix_prefix = prefix_prefix + "frozen_enc_" + args.enc_file_prefix + "_"
	if args.train_dec != "True":
		prefix_prefix = prefix_prefix + "frozen_dec_" + args.dec_file_prefix + "_"

if args.filter == "True":
    prefix_prefix = "filter_" + args.enc_role_scheme + "_" + args.dec_role_scheme + "_" + prefix_prefix

# Set the prefix for saving weights
file_prefix = prefix_prefix + layers_prefix + "_".join([args.encoder, args.decoder, args.task]) +  "_"
directories = os.listdir("./models")
found = False
suffix = 0
while not found:
	if "decoder_" + file_prefix + str(suffix) + ".weights" not in directories:
		found = 1
	else:
		suffix += 1

suffix = str(suffix)



if args.random_init == "True":
	file_prefix = file_prefix + suffix
	torch.save(encoder.state_dict(), "models/encoder_" + file_prefix + ".weights")
	torch.save(decoder.state_dict(), "models/decoder_" + file_prefix + ".weights")
elif args.filter == "True":
	tpr_enc = TensorProductEncoder(n_fillers=args.vocab_size, filler_dim=args.emb_size, role_dim=args.emb_size, final_layer_width=args.hidden_size, role_scheme=args.enc_role_scheme, max_length=args.max_length)
	tpr_dec = TensorProductEncoder(n_fillers=args.vocab_size, filler_dim=args.emb_size, role_dim=args.emb_size, final_layer_width=args.hidden_size, role_scheme=args.dec_role_scheme, max_length=args.max_length)

	tpr_enc = tpr_enc.to(device=device)
	tpr_dec = tpr_dec.to(device=device)


	if args.enc_file_prefix is not None:
		encoder.load_state_dict(torch.load("models/encoder_" + args.enc_file_prefix + ".weights", map_location=device))
		#tpr_enc.load_state_dict(torch.load("models/tpr_enc_" + args.enc_file_prefix + ".weights", map_location=device)) 
	if args.dec_file_prefix is not None:
		decoder.load_state_dict(torch.load("models/decoder_" + args.dec_file_prefix + ".weights", map_location=device))
		tpr_dec.load_state_dict(torch.load("models/tpr_dec_" + args.dec_file_prefix + ".weights", map_location=device))

	# Train the model
	if args.train == "True":
		file_prefix = file_prefix + suffix

		train_iters_filter(encoder, decoder, tpr_enc, tpr_dec, train_set, dev_set, file_prefix, input_to_output, max_epochs=100, patience=1, print_every=10000//args.batch_size, learning_rate=0.001, batch_size=args.batch_size, parsing_fn=parsing_fn, role_fn=role_fn, train_enc=args.train_enc=="True", train_dec=args.train_dec=="True", save_intermediate=args.save_intermediate=="True")


	# Evaluate the trained model
	if args.train_enc == "True":
		encoder.load_state_dict(torch.load("models/encoder_" + file_prefix + ".weights", map_location=device))
		tpr_enc.load_state_dict(torch.load("models/tpr_enc_" + file_prefix + ".weights", map_location=device)) 
	else:
		encoder.load_state_dict(torch.load("models/encoder_" + args.enc_file_prefix + ".weights", map_location=device))
		#tpr_enc.load_state_dict(torch.load("models/tpr_enc_" + file_prefix + ".weights", map_location=device))

	if args.train_dec == "True":
		decoder.load_state_dict(torch.load("models/decoder_" + file_prefix + ".weights", map_location=device))
		tpr_dec.load_state_dict(torch.load("models/tpr_dec_" + file_prefix + ".weights", map_location=device))
	else:
		decoder.load_state_dict(torch.load("models/decoder_" + args.dec_file_prefix + ".weights", map_location=device))
		tpr_dec.load_state_dict(torch.load("models/tpr_dec_" + args.dec_file_prefix + ".weights", map_location=device))
		tpr_enc.load_state_dict(torch.load("models/tpr_enc_" + args.dec_file_prefix + ".weights", map_location=device))	

else:
	if args.enc_file_prefix is not None:
		encoder.load_state_dict(torch.load("models/encoder_" + args.enc_file_prefix + ".weights", map_location=device))
	if args.dec_file_prefix is not None:
		decoder.load_state_dict(torch.load("models/decoder_" + args.dec_file_prefix + ".weights", map_location=device))

	# Train the model
	if args.train == "True":
		file_prefix = file_prefix + suffix
		
		if args.stages == "True":	
			train_iters_stages(encoder, decoder, train_set, dev_set, file_prefix, input_to_output, max_epochs=100, patience=1, print_every=10000//args.batch_size, learning_rate=0.001, batch_size=args.batch_size, parsing_fn=parsing_fn, role_fn=role_fn, train_enc=args.train_enc=="True", train_dec=args.train_dec=="True", save_intermediate=args.save_intermediate=="True")
		else:
			train_iters(encoder, decoder, train_set, dev_set, file_prefix, input_to_output, max_epochs=100, patience=1, print_every=10000//args.batch_size, learning_rate=0.001, batch_size=args.batch_size, parsing_fn=parsing_fn, role_fn=role_fn, train_enc=args.train_enc=="True", train_dec=args.train_dec=="True", save_intermediate=args.save_intermediate=="True")


	# Evaluate the trained model
	if args.train_enc == "True":
		encoder.load_state_dict(torch.load("models/encoder_" + file_prefix + ".weights", map_location=device))
	else:
		encoder.load_state_dict(torch.load("models/encoder_" + args.enc_file_prefix + ".weights", map_location=device))

	if args.train_dec == "True":
		decoder.load_state_dict(torch.load("models/decoder_" + file_prefix + ".weights", map_location=device))
	else:
		decoder.load_state_dict(torch.load("models/decoder_" + args.dec_file_prefix + ".weights", map_location=device))

report_file = open("models/results_" + file_prefix + ".txt", "w")

if args.filter == "True":
    correct, total = score_filter(encoder, decoder, tpr_enc, tpr_dec, batchify(test_set, 1), input_to_output, parsing_fn, role_fn)
else:
    correct, total = score(encoder, decoder, batchify(test_set, 1), input_to_output, parsing_fn, role_fn)

report_file.write("Test set results:\nCorrect:\t" + str(correct) + "\nTotal:\t" + str(total) + "\nAccuracy:\t" + str(correct * 1.0 / total) + "\n\n")



if args.generalization_prefix is not None:
	correct, total = score(encoder, decoder, batchify(generalization_set, 1), input_to_output, parsing_fn, role_fn)
	report_file.write("Generalization set results:\nCorrect:\t" + str(correct) + "\nTotal:\t" + str(total) + "\nAccuracy:\t" + str(correct * 1.0 / total) + "\n\n")






