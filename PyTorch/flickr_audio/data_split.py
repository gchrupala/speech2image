#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:57:38 2018

@author: danny
"""

# script that loads the flickr database json file in order to split
# the data into Karpathy's  train test and validation set.
import json

def split_data(f_nodes, loc):
    file = json.load(open(loc))
    split_dict = {}
    for x in file['images']:
        split_dict[x['filename'].replace('.jpg', '')] = x['split']
    
    train = []
    val = []
    test = []

    for x in f_nodes:
        name = x._v_name.replace('flickr_', '')
        if split_dict[name] == 'train':
            train.append(x)
        if split_dict[name] == 'val':
            val.append(x)    
        if split_dict[name] == 'test':
            test.append(x) 
    return train, val, test
