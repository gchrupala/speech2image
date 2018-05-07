#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:36:08 2018

@author: danny

evaluation functions. contains only recall@n now. convenience functions for embedding
the data and calculating the recall@n to keep your NN training script clean
"""
import sklearn.metrics.pairwise as sk
import numpy as np
import torch
from torch.autograd import Variable

# small convenience functions for combining everything in this script.
# N.B. make sure the order in which you pass the embedding functions is the 
# order in which the iterator yields the appropriate features!

def image2speech(iterator, image_embed_function, speech_embed_function, n, dtype):
    im_embeddings, speech_embeddings = embed_data(iterator, image_embed_function, speech_embed_function, dtype)
    return recall_at_n(im_embeddings, speech_embeddings, n)

def speech2image(iterator, image_embed_function, speech_embed_function, n, dtype):
    im_embeddings, speech_embeddings = embed_data(iterator, image_embed_function, speech_embed_function, dtype)
    return recall_at_n(speech_embeddings, im_embeddings, n)  

# embeds the validation or test data using the trained neural network. Takes
# an iterator (minibatcher) and the embedding functions (i.e. deterministic 
# output functions for your network). 
def embed_data(iterator, embed_function_1, embed_function_2, dtype):
    # set to evaluation mode
    embed_function_1.eval()
    embed_function_2.eval()
    for batch in iterator:
        img, sp = batch
        # convert data to pytorch variables
        img, sp = Variable(dtype(img), requires_grad=False), Variable(dtype(sp),requires_grad=False)
        # embed the data
        img = embed_function_1(img)
        sp = embed_function_2(sp)
        # concat to existing tensor or create one if non-existent yet
        try:
            speech = torch.cat((speech, sp.data))
        except:
            speech = sp.data
        try:
            image = torch.cat((image, img.data))
        except:
            image = img.data
    return speech, image

###########################################################################################

<<<<<<< HEAD
def recall_at_n(emb_1, emb_2, n, cosine = True):
=======
def recall_at_n(embeddings_1, embeddings_2, n, cosine = True):
>>>>>>> 9ed6340bbfd703bf0a13889b7f6a5b9f30b05aca
# calculate the recall at n for a retrieval task where given an embedding of some
# data we want to retrieve the embedding of a related piece of data (e.g. images and captions)
# recall works in the direction of embeddings_1 to embeddings_2, so if you pass images, and speech
# respectively you calculate image to speech retrieval scores.
<<<<<<< HEAD
    # total number of the embeddings
    n_emb = emb_1.size()[0]
    recall = np.zeros(np.shape(n))
    median_rank = 0
    # split the embeddings up in 5 parts, that is the 5 captions per image and take the average over 
    # the 5 captions
    for x in range (0,5):
        embeddings_1 = emb_1[int(x*(n_emb/5)) : int((x+1)*(n_emb/5))]
        embeddings_2 = emb_2[int(x*(n_emb/5)) : int((x+1)*(n_emb/5))]
    # get the cosine similarity matrix for the embeddings by default. pass cosine = False to use the
    # ordered distance measure proposed by vendrov et al. 
        if cosine:
            sim = torch.matmul(embeddings_1, embeddings_2.t())
        else:
            sim = torch.clamp(embeddings_1 - embeddings_2[0], min = 0).norm(1, dim = 1, keepdim = True)**2
            for x in range(1, embeddings_2.size(0)):
                s = torch.clamp(embeddings_1 - embeddings_2[x], min = 0).norm(1, dim = 1, keepdim = True)**2
                sim = torch.cat((sim, s), 1)
            sim = - sim   
        # apply sort two times to get a matrix where the values for each position indicate its rank in the column
        sorted, indices = sim.sort(dim = 1, descending = True)
        sorted, indices = indices.sort(dim = 1)
        # the diagonal of the resulting matrix diagonal now holds the rank of the correct embedding pair (add 1 cause 
        # sort counts from 0, we want the top rank to be indexed as 1)
        diag = indices.diag() +1
        if type(n) == int:
            r = diag.le(n).double().mean()
        elif type(n) == list:
            r = []
            for x in n:
                r.append(diag.le(x).double().mean())    
        # the average rank of the correct output
        median_rank += diag.median()
        recall += r        
    return(recall/5, median_rank/5)
=======
    
    # get the cosine similarity matrix for the embeddings by default. pass cosine = False to use the
    # ordered distance measure proposed by vendrov et al. 
    if cosine:
        sim = torch.matmul(embeddings_1, embeddings_2.t())
    else:
        sim = torch.clamp(embeddings_1 - embeddings_2[0], min = 0).norm(1, dim = 1, keepdim = True)**2
        for x in range(1, embeddings_2.size(0)):
            s = torch.clamp(embeddings_1 - embeddings_2[x], min = 0).norm(1, dim = 1, keepdim = True)**2
            sim = torch.cat((sim, s), 1)
        sim = - sim   
    # apply sort two times to get a matrix where the values for each position indicate its rank in the column
    sorted, indices = sim.sort(dim = 1, descending = True)
    sorted, indices = indices.sort(dim = 1)
    # the diagonal of the resulting matrix diagonal now holds the rank of the correct embedding pair (add 1 cause 
    # sort counts from 0, we want the top rank to be indexed as 1)
    diag = indices.diag() +1
    if type(n) == int:
        recall = diag.le(n).double().mean()
    elif type(n) == list:
        recall = []
        for x in n:
            recall.append(diag.le(x).double().mean())    
    # the average rank of the correct output
    median_rank = diag.median()
    return(recall, median_rank)
>>>>>>> 9ed6340bbfd703bf0a13889b7f6a5b9f30b05aca