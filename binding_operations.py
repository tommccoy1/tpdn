import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import numpy as np
import pickle

# Defines various functions for binding fillers and roles

# Defines the tensor product, used in tensor product representations
class SumFlattenedOuterProduct(nn.Module):
    def __init__(self):
        super(SumFlattenedOuterProduct, self).__init__()
           
    def forward(self, input1, input2):
        outer_product = torch.bmm(input1.transpose(1,2), input2)
        flattened_outer_product = outer_product.view(outer_product.size()[0],-1).unsqueeze(0)
        sum_flattened_outer_product = flattened_outer_product
        return sum_flattened_outer_product

# The next several functions define circular convolution, used in 
# holographic reduced representations
def permutation_matrix(dim, offset):
    matrix = []
    
    for i in range(dim):
        row = [0 for j in range(dim)]
        row[(offset + 1 + i)%dim] = 1
        matrix.append(row)
        
    return matrix
    
def permutation_tensor(dim):
    tensor = []
    
    for offset in range(dim)[::-1]:
        tensor.append(permutation_matrix(dim, offset))
        
    return tensor


class CircularConvolutionHelper(nn.Module):
    def __init__(self, dim):
        super(CircularConvolutionHelper, self).__init__()
        self.permuter = Variable(torch.FloatTensor(permutation_tensor(dim))).cuda()
        
           
    def forward(self, input1, input2):
        outer_product = torch.bmm(input1.unsqueeze(2), input2.unsqueeze(1))
        permuted = torch.bmm(self.permuter, outer_product.transpose(0,2))
        circular_conv = torch.sum(permuted, dim = 0)
        sum_rep = torch.sum(circular_conv, dim = 1)
        
        return sum_rep.unsqueeze(0).unsqueeze(0)


class CircularConvolution(nn.Module):
    def __init__(self, dim):
        super(CircularConvolution, self).__init__()
        self.helper = CircularConvolutionHelper(dim) 
           
    def forward(self, input1, input2):
        conv = None

        for i in range(len(input1)):
            this_conv = self.helper(input1[i], input2[i]) 
            if conv is None:
                conv = this_conv
            else:
                conv = torch.cat((conv, this_conv), 1)

        
        return conv

# Elementwise product
class EltWise(nn.Module):
    def __init__(self):
        super(EltWise, self).__init__()
        
           
    def forward(self, input1, input2):
        
        eltwise_product = input1 * input2
        
        sum_rep = torch.sum(eltwise_product, dim = 1)
        
        return sum_rep.unsqueeze(0)






