#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:13:00 2018

@author: danny
"""
#!/usr/bin/env python
from __future__ import print_function

import time
import tables
import argparse
import torch
from torch.autograd import Variable

from minibatchers import iterate_minibatches, iterate_minibatches_flickr
from costum_loss import batch_hinge_loss
from evaluate import speech2image
from encoders import img_encoder, Harwath_audio_encoder, RHN_audio_encoder, GRU_audio_encoder, RCNN_audio_encoder
from data_split import split_data

## implementation of an CNN for af recognition. made for use with mel filterbank features
parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')

parser.add_argument('-data_loc', type = str, default = '/prep_data/flickr_features.h5',
                    help = 'location of the feature file, default: /data/processed/fbanks.h5')
<<<<<<< HEAD
parser.add_argument('-batch_size', type = int, default = 32, help = 'batch size, default: 128')

parser.add_argument('-lr', type = float, default = 0.0002, help = 'learning rate, default:0.00005')
parser.add_argument('-n_epochs', type = int, default = 25, help = 'number of training epochs, default: 50')
=======
parser.add_argument('-batch_size', type = int, default = 32, help = 'batch size, default: 32')

parser.add_argument('-lr', type = float, default = 0.0002, help = 'learning rate, default:0.0002')
parser.add_argument('-n_epochs', type = int, default = 25, help = 'number of training epochs, default: 25')
>>>>>>> 4a9103b37fbb0b858302f961dda36a3bae957bb2
parser.add_argument('-loss', type = list, default = [False, True], help = 'determines which embeddings are normalised by the loss function')
parser.add_argument('-cuda', type = bool, default = True, help = 'use cuda, default: True')
parser.add_argument('-data_base', type = str, default = 'flickr', help = 'database to train on, options: places, flickr')
parser.add_argument('-visual', type = str, default = 'vgg', help = 'name of the node containing the visual features')
<<<<<<< HEAD
parser.add_argument('-audio', type = str, default = 'hw_norm_fbanks', help = 'name of the node containing the audio features')
=======
parser.add_argument('-audio', type = str, default = 'fbanks', help = 'name of the node containing the audio features')
>>>>>>> 4a9103b37fbb0b858302f961dda36a3bae957bb2

args = parser.parse_args()

# open the data file
data_file = tables.open_file(args.data_loc, mode='r+') 

# check if cuda is availlable and user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
    # if cuda cast all variables as cuda tensor
    dtype = torch.cuda.FloatTensor
else:
    print('using cpu')
    dtype = torch.FloatTensor

#get a list of all the nodes in the file. The places database was so big 
# it needs to be split into subgroups at the root node so the iterator needs
# to look one node deeper in the tree.
def iterate_places(h5_file):
    for x in h5_file.root:
        for y in x:
            yield y
def iterate_flickr(h5_file):
    for x in h5_file.root:
        yield x

if args.data_base == 'places':
    f_nodes = [node for node in iterate_places(data_file)]
    # define the batcher type to use.
    batcher = iterate_minibatches
elif args.data_base == 'flickr':
    f_nodes = [node for node in iterate_flickr(data_file)]
    # define the batcher type to use.
    batcher = iterate_minibatches_flickr
else:
    print('incorrect database option')
    exit()  

# total number of nodes (i.e. files) 
n_nodes= len(f_nodes)

# shuffle before dividing into train test and validation sets

#np.random.shuffle(f_nodes)
#args.data_split = [int(np.floor(x * n_nodes)) for x in args.data_split]
#train = f_nodes[0 : args.data_split[0]]
#val = f_nodes[args.data_split[0] : args.data_split[0] + args.data_split[1]]
#test = f_nodes[args.data_split[0] + args.data_split[1] : args.data_split[0] + args.data_split[1] + args.data_split[2]]

train, test, val = split_data(f_nodes)
#####################################################

# network modules
img_net = img_encoder()
audio_net = GRU_audio_encoder()
# move graph to gpu if cuda is availlable
if cuda:
    img_net.cuda()
    audio_net.cuda()

# optimiser
optimizer = torch.optim.Adam(list(img_net.parameters())+list(audio_net.parameters()), args.lr)

# this sets the learning rate for the model to a learning rate adapted for the number of
# epochs.
def lr_decay(optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // 5))
    for groups in optimizer.param_groups:
        groups['lr'] = lr

# training routine 
def train_epoch(epoch, img_net, audio_net, optimizer, f_nodes, batch_size):
    img_net.train()
    audio_net.train()
    # perform learning rate decay
    lr_decay(optimizer, epoch)
    # for keeping track of the average loss over all batches
    train_loss = 0
    num_batches =0
    for batch in batcher(f_nodes, batch_size, args.visual, args.audio, shuffle = True):
        img, audio = batch
        num_batches +=1
        # convert data to pytorch variables
        img, audio = Variable(dtype(img)), Variable(dtype(audio))
        # reset the gradients of the optimiser
        optimizer.zero_grad()
        # embed the images and audio using the networks
        img_embedding = img_net(img)
        audio_embedding = audio_net(audio)
        # calculate the loss
        loss = batch_hinge_loss(img_embedding, audio_embedding, args.loss, cuda)
        # calculate the gradients and perform the backprop step
        loss.backward()
        optimizer.step()
        # add loss to average
        train_loss += loss.data
        print(train_loss.cpu()[0]/num_batches)
    return train_loss/num_batches

def test_epoch(img_net, audio_net, f_nodes, batch_size):
    # set to evaluation mode
    img_net.eval()
    audio_net.eval()
    # for keeping track of the average loss
    test_batches = 0
    test_loss = 0
    for batch in batcher(f_nodes, batch_size, args.visual, args.audio, shuffle = False):
        img, audio = batch 
        test_batches += 1
        # convert data to pytorch variables
        img, audio = Variable(dtype(img)), Variable(dtype(audio))
        # embed the images and audio using the networks
        img_embedding = img_net(img)
        audio_embedding = audio_net(audio)
        loss = batch_hinge_loss(img_embedding, audio_embedding, args.loss, cuda)
        # add loss to average
        test_loss += loss.data 
    return test_loss/test_batches 
epoch = 1
# run the training loop for the indicated amount of epochs 
while epoch <= args.n_epochs:
    # keep track of runtime
    start_time = time.time()

    print('training epoch: ' + str(epoch))
    # Train on the train set
    train_loss = train_epoch(epoch, img_net, audio_net, optimizer, train, args.batch_size)
    # evaluate on the validation set
    val_loss = test_epoch(img_net, audio_net, val, args.batch_size)
    
    # calculate the recall@n
    # create a minibatcher over the validation set
    iterator = iterate_minibatches(val, args.batch_size, args.visual, args.audio, shuffle = False)
    # calc recal, pass it the iterator, the embedding functions and n
    # returns the measures columnise (speech2image retrieval) and rowwise(image2speech retrieval)
    recall, avg_rank = speech2image(iterator, img_net, audio_net, [1, 5, 10], dtype)

    # print some info about this epoch
    print("Epoch {} of {} took {:.3f}s".format(
            epoch, args.n_epochs, time.time() - start_time))
    print("training loss:\t\t{:.6f}".format(train_loss.cpu()[0]))
    print("validation loss:\t\t{:.6f}".format(val_loss.cpu()[0]))
    epoch += 1
    print('recall@1 = ' + str(recall[0]*100) + '%')
    print('recall@5 = ' + str(recall[1]*100) + '%')
    print('recall@10 = ' + str(recall[2]*100) + '%')
    print('average rank= ' + str(avg_rank))

test_loss = test_epoch(img_net, audio_net, test, args.batch_size)
# calculate the recall@n
# create a minibatcher over the test set
iterator = iterate_minibatches(test, args.batch_size, args.visual, args.audio, shuffle = False)
# calc recal, pass it the iterator, the embedding functions and n
# returns the measures columnise (speech2image retrieval) and rowwise(image2speech retrieval)
recall, avg_rank = speech2image(iterator, img_net, audio_net, [1, 5, 10], dtype)

print("test loss:\t\t{:.6f}".format(test_loss.cpu()[0]))
print('test recall@1 = ' + str(recall[0]*100) + '%')
print('test recall@5 = ' + str(recall[1]*100) + '%')
print('test recall@10 = ' + str(recall[2]*100) + '%')
print('test average rank= ' + str(avg_rank))

