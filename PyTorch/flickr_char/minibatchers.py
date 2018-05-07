#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:06:06 2018

@author: danny
minibatchers for the neural network training. the flickr dataset has its own batcher
because there are 5 captions to an image, the batcher loads all captions while making sure
the same image does not pop up twice in the same minibatch. 
"""
import numpy as np
from prep_text import char_2_1hot, char_2_index
# minibatcher which takes a list of nodes and returns the visual and audio features, possibly resized.
# visual and audio should contain a string of the names of the visual and audio features nodes in the h5 file.
#frames is the desired length of the time sequence, the batcher pads or truncates.
def iterate_minibatches(f_nodes, batchsize, visual, audio, frames = 1024, shuffle=True):  
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):  
        # take a batch of nodes of the given size               
        excerpt = f_nodes[start_idx:start_idx + batchsize]        
        speech=[]
        images=[]
        for ex in excerpt:
            # extract and append the vgg16 features
            images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
            # retrieve the audio features
            sp = eval('ex.' + audio + '._f_list_nodes()[0].read().transpose()')
            # padd to the given output size
            n_frames = sp.shape[1]
            if n_frames < frames:
                sp = np.pad(sp, [(0, 0), (0, frames - n_frames )], 'constant')
            # truncate to the given input size
            if n_frames > frames:
                sp = sp[:,:frames]
            speech.append(sp)
        # reshape the features into appropriate shape and recast as float32
        speech = np.float64(speech)
        images_shape = np.shape(images)
        # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
        images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
        yield images, speech    

# the flickr minibatcher returns all 5 captions per image during training, for test or validation
# epochs, set the value for test to false to load just one caption per image. 
def iterate_audio_flickr(f_nodes, batchsize, visual, audio, frames = 1024, shuffle = True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for i in range(0, 5):
        for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
            # take a batch of nodes of the given size               
            excerpt = f_nodes[start_idx:start_idx + batchsize]
            speech=[]
            images=[]
            for ex in excerpt:
                # extract and append the vgg16 features
                images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
                # extract the audio features
                sp = eval('ex.' + audio + '._f_list_nodes()[i].read().transpose()')
                # padd to the given output size
                n_frames = sp.shape[1]
                if n_frames < frames:
                    sp = np.pad(sp, [(0, 0), (0, frames - n_frames )], 'constant')
                # truncate to the given input size
                if n_frames > frames:
                    sp = sp[:,:frames]
                speech.append(sp)
            # reshape the features and recast as float64
            speech = np.float64(speech)
            images_shape = np.shape(images)
            # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
            images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
            yield images, speech

# iterate over text input. the value for chars indicates the max sentence lenght in characters
def iter_text_flickr(f_nodes, batchsize, visual, text, chars = 200, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for i in range(0,5):
        for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
            # take a batch of nodes of the given size               
            excerpt = f_nodes[start_idx:start_idx + batchsize]
            caption = []
            images = []
            for ex in excerpt:
                # extract and append the vgg16 features
                images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
                # extract the audio features
                cap = eval('ex.' + text + '._f_list_nodes()[i].read()')
                cap = cap.decode('utf-8')
                # append an otherwise unused character as a start of sentence character and 
                # convert the sentence to lower case.
                caption.append(cap)
            # converts the sentence to character ids. 
            caption = char_2_index(caption, batchsize, chars)
            images_shape = np.shape(images)
            # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
            images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
            yield images, caption