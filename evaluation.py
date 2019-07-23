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

use_cuda = torch.cuda.is_available()


# Given an encoder, a decoder, and an input, return the guessed output
# and the encoding of the input
def evaluate(encoder1, decoder1, example, input_to_output, parsing_fn, role_fn):
    encoding = encoder1([example])
    #print(example)
    #print(role_fn([example]))
    predictions = decoder1(encoding, output_len=len(example), tree=parsing_fn([example]), role_list=role_fn([example]))
    correct = input_to_output(example)
        
    guessed_seq = []
        
    for prediction in predictions:
        topv, topi = prediction.data.topk(1)
        ni = topi.item() 
            
        guessed_seq.append(ni)
        
    return guessed_seq, encoding
 
# Given an encoder, a decoder, and an input, return the guessed output
# and the encoding of the input
def evaluate_tpun(decoder1, example, input_to_output, parsing_fn, role_fn):
    #print(example)
    #print(role_fn([example]))
    #print(example)
    #print(len(example[0]))
    #print(role_fn([example[0]]))
    predictions = decoder1(example[1], output_len=len(example[0][0]), tree=parsing_fn([example[0][0]]), role_list=role_fn([example[0][0]]))
    #correct = input_to_output(example[0])
    correct = example[0]
        
    guessed_seq = []
       
    #print(predictions) 
    for prediction in predictions:
        topv, topi = prediction.data.topk(1)
        ni = topi.item() 
            
        guessed_seq.append(ni)
        
    return guessed_seq
            
# Given an encoder, decoder, evaluation set, and function for generating
# the correct outputs, return the number of correct predictions
# and the total number of predictions
def score(encoder1, decoder1, evaluation_set, input_to_output, parsing_fn, role_fn):
    total_correct = 0
    total = 0
    

    for batch in evaluation_set:
            for example in batch:
                correct = input_to_output(example)
        
                guess = evaluate(encoder1, decoder1, example, input_to_output, parsing_fn, role_fn)
        
                if tuple(guess[0]) == tuple(correct):
                    total_correct += 1
                #else:
                #    pass#print(tuple(guess[0]), tuple(correct))
                total += 1
        
    return total_correct, total

# Given an encoder, decoder, evaluation set, and function for generating
# the correct outputs, return the number of correct predictions
# and the total number of predictions
def score_tpun(decoder1, evaluation_set, input_to_output, parsing_fn, role_fn):
    total_correct = 0
    total = 0
    

    for example in evaluation_set:
            #print(example)
            correct = example[0][0] #.data.cpu().numpy())
            #correct = input_to_output(example[0][0])
        
            guess = evaluate_tpun(decoder1, example, input_to_output, parsing_fn, role_fn)
            #print(guess)        
            if tuple(guess) == tuple(correct):
                total_correct += 1
                #else:
                #    pass#print(tuple(guess[0]), tuple(correct))
            total += 1
        
    return total_correct, total
            
# Given an encoder, decoder, evaluation set, and function for generating
# the correct outputs, return the number of correct predictions
# and the total number of predictions
def score_double(encoder1, encoder2, decoder1, evaluation_set, input_to_output, parsing_fn, role_fn):
    total_correct = 0
    total = 0
    

    for batch in evaluation_set:
            for example in batch:
                correct = input_to_output(example)
        
                guess = evaluate_double(encoder1, encoder2, decoder1, example, input_to_output, parsing_fn, role_fn)
        
                if tuple(guess[0]) == tuple(correct):
                    total_correct += 1
                #else:
                #    pass#print(tuple(guess[0]), tuple(correct))
                total += 1
        
    return total_correct, total

#
# Given an encoder, a decoder, and an input, return the guessed output
# and the encoding of the input
def evaluate_double(encoder1, encoder2, decoder1, example, input_to_output, parsing_fn, role_fn):
    encoding = encoder1([example])
    encoding2 = encoder2([example])
    #print(example)
    #print(role_fn([example]))
    predictions = decoder1(encoding + encoding2, output_len=len(example), tree=parsing_fn([example]), role_list=role_fn([example]))
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
def score_filter(encoder1, decoder1, tpr_enc, tpr_dec, evaluation_set, input_to_output, parsing_fn, role_fn):
    total_correct = 0
    total = 0
    

    for batch in evaluation_set:
            for example in batch:
                correct = input_to_output(example)
        
                guess = evaluate_filter(encoder1, decoder1, tpr_enc, tpr_dec, example, input_to_output, parsing_fn, role_fn)
        
                if tuple(guess[0]) == tuple(correct):
                    total_correct += 1
                #else:
                #    pass#print(tuple(guess[0]), tuple(correct))
                total += 1
        
    return total_correct, total


# Given an encoder, a decoder, and an input, return the guessed output
# and the encoding of the input
def evaluate_filter(encoder1, decoder1, tpr_enc, tpr_dec, example, input_to_output, parsing_fn, role_fn):
    encoding = encoder1([example])
    tpr_encoding = tpr_enc([example])
    tpr_dec_encoding = tpr_dec([input_to_output(x) for x in [example]])
    #print(example)
    #print(role_fn([example]))
    #predictions = decoder1(encoding - tpr_encoding, output_len=len(example), tree=parsing_fn([example]), role_list=role_fn([example]))
    predictions = decoder1(tpr_dec_encoding, output_len=len(example), tree=parsing_fn([example]), role_list=role_fn([example]))
    correct = input_to_output(example)
        
    guessed_seq = []
        
    for prediction in predictions:
        topv, topi = prediction.data.topk(1)
        ni = topi.item() 
            
        guessed_seq.append(ni)
        
    return guessed_seq, encoding - tpr_enc([example])
 




 
