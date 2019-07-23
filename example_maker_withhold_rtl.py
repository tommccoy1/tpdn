import numpy as np
from random import shuffle
import sys
import argparse

from print_read_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", help="maximum sequence length", type=int, default=6)
parser.add_argument("--min_seq_length", help="minimum sequence length", type=int, default=1)
parser.add_argument("--vocab_size", help="vocabulary size", type=int, default=10)
parser.add_argument("--num_train", help="number of training examples to generate", type=int, default=40000)
parser.add_argument("--num_dev", help="number of dev examples to generate", type=int, default=5000) 
parser.add_argument("--num_test", help="number of test examples to generate", type=int, default=5000) 
parser.add_argument("--prefix", help="prefix for saving the generated values", type=str, default="digits")
args = parser.parse_args()


# Creates a training set, dev set, and test set
# of size num_train, num_dev, and num_test
# Each example consists of a sequence of digits of
# length seq_length, where each digit is randomly
# drawn from 0 to (vocab_size - 1)
def generate_examples(min_seq_length, max_seq_length, vocab_size, num_train, num_dev, num_test):
    train_set = []
    dev_set = []
    test_set = []
    
    list_examples = []
    dict_examples = {}
    
    num_examples = 0
    while num_examples < num_train + num_dev + num_test:
        seq_length = min_seq_length + np.random.randint(max_seq_length - min_seq_length + 1)
        seq = tuple(np.random.randint(vocab_size,size=seq_length))
        works = True
        if seq not in dict_examples:
            for index, elt in enumerate(list(seq)[::-1]):
                if index == elt:
                    works = False

            if works:
                list_examples.append(seq) #(seq, range(len(seq)), seq))
                dict_examples[seq] = 1
                num_examples += 1
            
    shuffle(list_examples)
    train_set = list_examples[:num_train]
    dev_set = list_examples[num_train:num_train + num_dev]
    test_set = list_examples[num_train + num_dev:]
    
    return train_set, dev_set, test_set

train_set, dev_set, test_set = generate_examples(args.min_seq_length, args.max_seq_length, args.vocab_size, args.num_train, args.num_dev, args.num_test)


fo_train = open('data/' + args.prefix + '.train', 'w')
fo_dev = open('data/' + args.prefix + '.dev', 'w')
fo_test = open('data/' + args.prefix + '.test', 'w')

lists_to_file(train_set, fo_train)
lists_to_file(dev_set, fo_dev)
lists_to_file(test_set, fo_test)



