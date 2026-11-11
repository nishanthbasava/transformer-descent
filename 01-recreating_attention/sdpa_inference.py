# This file is just for inference time, I'll do another one with training but for now just trying to lock in inference.

import numpy as np
import torch
import torch.nn as nn # only using this for text and positional encodings
from transformers import AutoTokenizer # just for tokenization before using nn.Embedding
import math
import torch.nn.functional as F # stateless functions version of whats in torch.nn, softmax, relu, crossentropy, etc

rng = np.random.default_rng(seed=42)
device = "cuda" if torch.cuda.is_available() else "cpu"   # or "mps" on Mac

# settings the dimensions
dmodel = 512
h = 8
dk = 64
dv = 64
T = 0 #hmm
B = 6 # number of attention blocks
#dvocab set after tokenizer is loaded in below

# learned parameters that I'm just initializing to ones and zeros for now
# these are used in layer normalization
# gamma = torch.ones(dmodel)
# beta = torch.zeros(dmodel)
ln = nn.LayerNorm(dmodel).to(device)

 ''' 
 How to do LayerNormalization from scratch
    mean = x.mean(dim=-1, keepdims=True)
    var = x.var(dim=-1, keepdims=True, unbiased=True)
    x = gamma * (x - mean) / torch.sqrt(var + 1e-5) + beta
'''

# creating the necessary weight matrices
# again, initializing them randomly, not actually learning them right now.

'''
Learned Weight Matrices
Wq, [dmodel, dk]
Wk, [dmodel, dk]
Wv, [dmodel, dv]
Wo, [hdv, dmodel]
'''

Wq = rng.random((dmodel, dmodel))
Wk = rng.random((dmodel, dmodel))
Wv = rng.random((dmodel, dmodel))
Wo = rng.random((h*dk, dmodel))

'''
Text Tokenizing and Embedding
Just using nn.Embedding here so it's chill
'''
tokenizer = AutoTokenizer.from_pretrained("gpt-2")
dvocab = tokenizer.vocab_sizewhat 
embedding_layer = nn.Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=512).to(device)
Wf = rng.random((dmodel, dvocab))

'''
Positional Encodings
Using nn.Embedding here again
'''

pos_emb = nn.Embedding(max_len, d_model)

'''
Actual inference loop
Takes in a sentence of tokens and outputs JUST the next token
Multi-headed SDPA
'''

# send the stuff involved in inference to a GPU before (even though it's not that much compute required because not training so no crazy gradient calculations)
# just for experience with .device and .to (maybe Triton later down the road?)

def inference(text):

    token_ids = tokenizer.encode(text)
    ids = torch.tensor(token_ids, dtype=torch.long) #LongTensor is the datatype that nn.Embedding requires
    # dims (T,)
    T = ids.shape[0]

    x = embedding_layer + pos_emb(torch.arange(T, device=ids.device))
    print(x.shape) # x.shape = (T, 512)

    # Entering the attention block
    # need to create the individual attention heads for multi-headed attention.
    for j in range(B):
        Q = x @ Wq
        K = x @ Wk
        V = x @ Wv

        dk = dmodel // h
        out = []
        for i in range(h): # for each of the attention heads
            Q_i = Q[:, i*dk:(i+1)*dk]
            K_i = K[:, i*dk:(i+1)*dk]
            V_i = V[:, i*dv:(i+1)*dv]

            y = Q_i @ K_i.T
            y = y / math.sqrt(dk)
            # no masking here since I'm not implementing the inference loop. But when I do, remember the mask

            attn = F.softmax(y, dim=-1) #dim=-1 normalizes over the last axis, which in this case is the rows, so each row of attention weights sums to 1.

            '''
            Here is softmax implemented with just raw NumPy
            def softmax(x, axis=-1):
                e = np.exp(x - x.max(axis=axis, keepdims=True))   # subtract max for stability
                return e / e.sum(axis=axis, keepdims=True)
            '''

            y = attn @ V_i
            out.append(y)

        # concatenate all of the heads and run through linear layer (Wo)    
        out = torch.cat(outs, dim=-1) 
        out = out @ Wo # [T, h*dk] @ [h*dk, dmodel] => [T, dmodel]

        # add and apply LayerNorm here
        x += out
        x = ln(x)

        # Now time to pass through the MLP
        # ok so i need to actually understand what is going on in the MLP
        dff = 2048
        w1 = torch.randn(dmodel, dff)
        b1 = torch.zeros(dff)
        w2 = torch.randn(dff, dmodel)
        b2 = torch.zeros(dmodel)

        # first linear transformation for expanding the space hmm
        x = x @ w1 + b1

        # activation function (GELU)
        x = F.gelu(x)

        # second linear transformation
        # I think that ill get a better intuition for this MLP when I'm building the training process 
        x = x @ w2 + b2 

        #final LayerNorm
        x = ln(x)

    # Exiting the attention blocks
    x = x @ Wf
    x = F.softmax(x, dim=-1)
    print(f"Final Softmaxes: " + x)
    return x

    # sample from the softmaxes and output the next token.
    # now it's time to understand batch training

    # for now ill just implement greedy decoding 






    return x