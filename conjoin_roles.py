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
parser.add_argument("--prefix", help="prefix for your training/dev data", type=str, default="digits")
parser.add_argument("--generalization_prefix", help="prefix for generalization test set", type=str, default=None)
parser.add_argument("--role_scheme1", help="first pre-coded role scheme to use", type=str, default=None)
parser.add_argument("--role_scheme2", help="second pre-coded role scheme to use", type=str, default=None)
args = parser.parse_args()


fi_train = open('data/' + args.prefix + '.train', 'r')
fi_dev = open('data/' + args.prefix + '.dev', 'r')
fi_test = open('data/' + args.prefix + '.test', 'r')

# Load the data sets
train_set = file_to_lists(fi_train)
dev_set = file_to_lists(fi_dev)
test_set = file_to_lists(fi_test)

max_length = 6
n_f = 10

if args.role_scheme1 == "bow":
	n_r1, seq_to_roles1 = create_bow_roles(max_length, n_f)
elif args.role_scheme1 == "ltr":
	n_r1, seq_to_roles1 = create_ltr_roles(max_length, n_f)
elif args.role_scheme1 == "rtl":
	n_r1, seq_to_roles1 = create_rtl_roles(max_length, n_f)
elif args.role_scheme1 == "bi":
	n_r1, seq_to_roles1 = create_bidirectional_roles(max_length, n_f)
elif args.role_scheme1 == "wickel":
	n_r1, seq_to_roles1 = create_wickel_roles(max_length, n_f)
elif args.role_scheme1 == "tree":
	n_r1, seq_to_roles1 = create_tree_roles(max_length, n_f)
elif args.role_scheme1 == "lth":
	n_r1, seq_to_roles1 = create_lth_roles(max_length, n_f)
elif args.role_scheme1 == "htl":
	n_r1, seq_to_roles1 = create_htl_roles(max_length, n_f)
elif args.role_scheme1 == "birank":
	n_r1, seq_to_roles1 = create_birank_roles(max_length, n_f)
elif args.role_scheme1 == "inter":
	n_r1, seq_to_roles1 = create_inter_roles(max_length, n_f)
elif args.role_scheme1 == "treerev":
	n_r1, seq_to_roles1 = create_treerev_roles(max_length, n_f)
elif args.role_scheme1 == "treelth":
	n_r1, seq_to_roles1 = create_treelth_roles(max_length, n_f)
elif args.role_scheme1 == "treeinter":
	n_r1, seq_to_roles1 = create_treeinter_roles(max_length, n_f)
else:
	print("Invalid role scheme")


if args.role_scheme2 == "bow":
	n_r2, seq_to_roles2 = create_bow_roles(max_length, n_f)
elif args.role_scheme2 == "ltr":
	n_r2, seq_to_roles2 = create_ltr_roles(max_length, n_f)
elif args.role_scheme2 == "rtl":
	n_r2, seq_to_roles2 = create_rtl_roles(max_length, n_f)
elif args.role_scheme2 == "bi":
	n_r2, seq_to_roles2 = create_bidirectional_roles(max_length, n_f)
elif args.role_scheme2 == "wickel":
	n_r2, seq_to_roles2 = create_wickel_roles(max_length, n_f)
elif args.role_scheme2 == "tree":
	n_r2, seq_to_roles2 = create_tree_roles(max_length, n_f)
elif args.role_scheme2 == "lth":
	n_r2, seq_to_roles2 = create_lth_roles(max_length, n_f)
elif args.role_scheme2 == "htl":
	n_r2, seq_to_roles2 = create_htl_roles(max_length, n_f)
elif args.role_scheme2 == "birank":
	n_r2, seq_to_roles2 = create_birank_roles(max_length, n_f)
elif args.role_scheme2 == "inter":
	n_r2, seq_to_roles2 = create_inter_roles(max_length, n_f)
elif args.role_scheme2 == "treerev":
	n_r2, seq_to_roles2 = create_treerev_roles(max_length, n_f)
elif args.role_scheme2 == "treelth":
	n_r2, seq_to_roles2 = create_treelth_roles(max_length, n_f)
elif args.role_scheme2 == "treeinter":
	n_r2, seq_to_roles2 = create_treeinter_roles(max_length, n_f)
else:
	print("Invalid role scheme")


fo_train = open('data/' + args.prefix + "_" + args.role_scheme1 + "_" + args.role_scheme2 + '.data_from_train.roles', 'w')
for elt in train_set:
	roles1 = seq_to_roles1(elt)
	roles2 = seq_to_roles2(elt)

	conjunction = [str(roles1[i]) + "_" + str(roles2[i]) for i in range(len(roles1))]

	fo_train.write(" ".join(conjunction) + "\n")

fo_dev = open('data/' + args.prefix + "_" + args.role_scheme1 + "_" + args.role_scheme2 + '.data_from_dev.roles', 'w')
for elt in dev_set:
	roles1 = seq_to_roles1(elt)
	roles2 = seq_to_roles2(elt)

	conjunction = [str(roles1[i]) + "_" + str(roles2[i]) for i in range(len(roles1))]

	fo_dev.write(" ".join(conjunction) + "\n")

fo_test = open('data/' + args.prefix + "_" + args.role_scheme1 + "_" + args.role_scheme2 + '.data_from_test.roles', 'w')
for elt in test_set:
	roles1 = seq_to_roles1(elt)
	roles2 = seq_to_roles2(elt)

	conjunction = [str(roles1[i]) + "_" + str(roles2[i]) for i in range(len(roles1))]

	fo_test.write(" ".join(conjunction) + "\n")












