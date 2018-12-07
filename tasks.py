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

# Provide the predefined digit sequence tasks

def interleaved(sequence):
    if len(sequence) <= 1:
        return list(sequence)
    else:
        return [sequence[0], sequence[-1]] + interleaved(sequence[1:-1])

def transform(sequence, task):
    if task == "auto":
        return sequence
    if task == "rev":
        return sequence[::-1]
    if task == "sort":
        return sorted(sequence)
    if task == "interleave":
        return interleaved(sequence)

