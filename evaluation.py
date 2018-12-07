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

# Functions for evaluating seq2seq models and TPDNs

# Given an encoder, a decoder, and an input, return the guessed output
# and the encoding of the input
def evaluate(encoder1, decoder1, example, input_to_output):
    encoding = encoder1([example])
    predictions = decoder1(encoding, len(example), [parse_digits(example)])
    correct = input_to_output(example)
        
    guessed_seq = []
        
    for prediction in predictions:
        topv, topi = prediction.data.topk(1)
        ni = topi.item() 
            
        guessed_seq.append(ni)
        
    return guessed_seq, encoding
            
# Given an encoder, decoder, evaluation set, and function for generating
# the correct outputs, return the number of correct predictions
# and the total number of predictions
def score(encoder1, decoder1, evaluation_set, input_to_output):
    total_correct = 0
    total = 0
    

    for batch in evaluation_set:
            for example in batch:
                correct = input_to_output(example)
        
                guess = evaluate(encoder1, decoder1, example, input_to_output)
        
                if tuple(guess[0]) == tuple(correct):
                    total_correct += 1
                total += 1
        
    return total_correct, total

# This function takes a tensor product encoder and a standard decoder, as well as a sequence
# of digits as inputs. It then uses the tensor product encoder to encode the sequence and uses
# the standard decoder to decode it, and returns the result.
def evaluate2(encoder, decoder, example):
    
    encoder_hidden = encoder(Variable(torch.LongTensor(example[0])).cuda().unsqueeze(0), Variable(torch.LongTensor(example[1])).cuda().unsqueeze(0))
    predictions = decoder(encoder_hidden, len(example[0]), [parse_digits(example[0])])
        
    guessed_seq = []
    for prediction in predictions:
        topv, topi = prediction.data.topk(1)
        ni = topi.item()
            
        guessed_seq.append(ni)
    
        
    return guessed_seq

def score2(encoder, decoder, input_to_output, test_set, index_to_filler):
    # Evaluate this TPR encoder for how well it can encode sequences in a way
    # that our original mystery_decoder can decode
    accurate = 0
    total = 0

    for batch in test_set:
        for example in batch:
            example = example[0]
            pred = evaluate2(encoder, decoder, example)
            
                     
            if tuple(input_to_output([index_to_filler[x] for x in example[0]])) == tuple([str(x) for x in pred]):
                accurate += 1
            total += 1
    
    # Gives how many sequences were properly decoded, out of the total number of test sequences    
    return accurate, total


