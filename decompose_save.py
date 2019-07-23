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
parser.add_argument("--decomp_type", help="TPBN or TPUN", type=str, default="tpbn")
args = parser.parse_args()

# Create the logfile
if args.decomp_type == "tpbn":
	if args.final_linear != "True":
		results_page = open("logs/" + args.data_prefix.split("/")[-1] + str(args.role_prefix).split("/")[-1] + str(args.role_scheme) + ".filler" + str(args.filler_dim) + ".role" + str(args.role_dim) + ".tpr_encomp.nf", "w")
	else:
		results_page = open("logs/" + args.data_prefix.split("/")[-1] + str(args.role_prefix).split("/")[-1] + str(args.role_scheme) + ".filler" + str(args.filler_dim) + ".role" + str(args.role_dim) + "." + str(args.embed_squeeze) + ".tpr_encomp", "w")

else:
	if args.final_linear != "True":
		results_page = open("logs/" + args.data_prefix.split("/")[-1] + str(args.role_prefix).split("/")[-1] + str(args.role_scheme) + ".filler" + str(args.filler_dim) + ".role" + str(args.role_dim) + ".tpr_decomp.nf", "w")
	else:
		results_page = open("logs/" + args.data_prefix.split("/")[-1] + str(args.role_prefix).split("/")[-1] + str(args.role_scheme) + ".filler" + str(args.filler_dim) + ".role" + str(args.role_dim) + "." + str(args.embed_squeeze) + ".tpr_decomp", "w")


# Load the decoder for computing swapping accuracy
if args.test_decoder == "True":
	parsing_fn = lambda x: None
	role_fn = lambda x: None
	if args.decoder == "ltr":
		decoder = DecoderRNN(args.vocab_size, args.decoder_embedding_size, args.hidden_size)
	elif args.decoder == "bi":
		decoder = DecoderBiRNN(args.vocab_size, args.decoder_embedding_size, args.hidden_size)
	elif args.decoder == "tree":
		decoder = DecoderTreeRNN(args.vocab_size, args.decoder_embedding_size, args.hidden_size)
		parsing_fn = lambda x: [parse_digits(elt) for elt in x]
	elif args.decoder == "tpr":
		n_r, role_fn_a = create_role_scheme(args.dec_role_scheme, args.max_length, args.vocab_size)
		role_fn = lambda x: [role_fn_a(elt) for elt in x]
		decoder = TensorProductDecoder(n_roles=n_r, n_fillers=args.vocab_size, filler_dim=args.emb_size, role_dim=args.emb_size, final_layer_width=args.hidden_size)
	else:
		print("Invalid decoder type")

	input_to_output = lambda seq: transform(seq, args.decoder_task)

	decoder.load_state_dict(torch.load("models/decoder_" + args.decoder_prefix + ".weights"))

	if use_cuda:
		decoder = decoder.cuda()

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



weights_matrix = None

# Prepare the embeddings
# If a file of embeddings was provided, use those.
embedding_dict = None
if args.embedding_file is not None:
	weights_matrix = create_embedding_dictionary(args.embedding_file, args.filler_dim, filler_to_index, index_to_filler, emb_squeeze=args.embed_squeeze, unseen_words=args.unseen_words)


if args.decomp_type == "tpbn":
	# Initialize the TPDN
	if args.final_linear == "True":
		tpr_encoder = TensorProductEncoder(role_scheme=args.role_scheme, n_fillers=vocab_size, final_layer_width=args.hidden_size, filler_dim=args.filler_dim, role_dim=args.role_dim, pretrained_embeddings=weights_matrix, embedder_squeeze=args.embed_squeeze, max_length=max_length, role_filea=role_filea, filler_filea=filler_filea, role_fileb=role_fileb, filler_fileb=filler_fileb,role_filec=role_filec, filler_filec=filler_filec,role_filed=role_filed, filler_filed=filler_filed)
	else:
		tpr_encoder = TensorProductEncoder(role_scheme=args.role_scheme, n_fillers=vocab_size, final_layer_width=None, filler_dim=args.filler_dim, role_dim=args.role_dim, pretrained_embeddings=weights_matrix, embedder_squeeze=args.embed_squeeze, max_length=max_length, role_filea=role_filea, filler_filea=filler_filea, role_fileb=role_fileb, filler_fileb=filler_fileb,role_filec=role_filec, filler_filec=filler_filec,role_filed=role_filed, filler_filed=filler_filed)

	tpr_encoder = tpr_encoder.to(device=device)



	args.data_prefix = args.data_prefix.split("/")[-1] + ".filler" + str(args.filler_dim) + ".role" + str(args.role_dim)
	if args.final_linear != "True":
		args.data_prefix += ".no_final"

	# Train the TPDN
	args.role_prefix = str(args.role_prefix).split("/")[-1]
	if args.train == "True":
		end_loss = train_iters_tpr(train_set, dev_set, tpr_encoder, 100, print_every=1000//32, learning_rate = 0.001, weight_file="models/" + args.data_prefix + str(args.role_prefix) + str(args.role_scheme) + ".tpr_enc", batch_size=32)

	# Load the trained TPDn
	tpr_encoder.load_state_dict(torch.load("models/" + args.data_prefix + str(args.role_prefix) + str(args.role_scheme) + ".tpr_enc"))

	total_mse = 0

	test_data = batchify_tpr(test_set, 1)
	# Evaluate on test set
	for i in range(len(test_data)): 
		encoding = tpr_encoder(test_data[i][0])

		total_mse += torch.mean(torch.pow(encoding.data - test_data[i][1].data, 2))




	final_test_loss = total_mse / len(test_data) 

	results_page.write(args.data_prefix + str(args.role_prefix) + str(args.role_scheme) + ".tpr_enc" +  " MSE on test set: " + str( final_test_loss.item()) + "\n" )

	if args.test_decoder == "True":
		correct, total = score(tpr_encoder, decoder, batchify([x[0] for x in test_set], 1), input_to_output, parsing_fn, role_fn)
		results_page.write(args.data_prefix + str(args.role_prefix) + str(args.role_scheme) + ".tpr_enc" + " Swapping encoder performance: " + str(correct) + " " +  str(total) + "\n")

else:
	n_r, role_fn_a = create_role_scheme(args.role_scheme, max_length, vocab_size, role_filea=role_filea, filler_filea=filler_filea, role_fileb=role_fileb, filler_fileb=filler_fileb,role_filec=role_filec, filler_filec=filler_filec,role_filed=role_filed, filler_filed=filler_filed)
	role_fn = lambda x: [role_fn_a(elt) for elt in x]

	# Initialize the TPDN
	if args.final_linear == "True":
		tpr_decoder = TensorProductDecoder(n_roles=n_r, n_fillers=vocab_size, final_layer_width=args.hidden_size, filler_dim=args.filler_dim, role_dim=args.role_dim, pretrained_embeddings=weights_matrix, embedder_squeeze=args.embed_squeeze)
	else:
		tpr_decoder = TensorProductDecoder(n_roles=n_r, n_fillers=vocab_size, final_layer_width=None, filler_dim=args.filler_dim, role_dim=args.role_dim, pretrained_embeddings=weights_matrix, embedder_squeeze=args.embed_squeeze)

	tpr_decoder = tpr_decoder.to(device=device)

	args.data_prefix = args.data_prefix.split("/")[-1] + ".filler" + str(args.filler_dim) + ".role" + str(args.role_dim)
	if args.final_linear != "True":
		args.data_prefix += ".no_final"

	input_to_output = lambda sequence: transform(sequence, args.decoder_task)

	# Train the TPDN
	args.role_prefix = str(args.role_prefix).split("/")[-1]
	if args.train == "True":
		end_loss = train_iters_tpr_unbind(train_set, dev_set, tpr_decoder, 100, input_to_output, print_every=1000, learning_rate = 0.001, weight_file="models/" + args.data_prefix + str(args.role_prefix) + str(args.role_scheme) + ".tpr_dec", batch_size=32, role_fn=role_fn)

	# Load the trained TPDn
	tpr_decoder.load_state_dict(torch.load("models/" + args.data_prefix + str(args.role_prefix) + str(args.role_scheme) + ".tpr_dec"))

	test_data = batchify_tpr(test_set, 1)
	total_mse = dev_loss_unbind(tpr_decoder, nn.NLLLoss(), test_data, input_to_output, None, role_fn)

	final_test_loss = total_mse / len(test_data) 

	results_page.write(args.data_prefix + str(args.role_prefix) + str(args.role_scheme) + ".tpr_enc" +  " MSE on test set: " + str( final_test_loss.item()) + "\n" )

	if args.test_decoder == "True":
		correct, total = score_tpun(tpr_decoder, test_data, input_to_output, parsing_fn, role_fn)
		results_page.write(args.data_prefix + str(args.role_prefix) + str(args.role_scheme) + ".tpr_enc" + " Swapping encoder performance: " + str(correct) + " " +  str(total) + "\n")




