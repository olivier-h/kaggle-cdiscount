#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 22:20:02 2017

@author: hondagneu
"""

import numpy as np
import pandas as pd
import io
import bson
import matplotlib.pyplot as plt
from skimage.data import imread

categories = pd.read_csv('category_names.csv', index_col='category_id')

rows, cols = 14, 8

with open('train_example.bson', 'rb') as f:
    data = bson.decode_file_iter(f)
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    ax = ax.ravel()
    
    i = 0
    try:
        for c, d in enumerate(data):
            product_id = d['_id']
            category_id = d['category_id']
            for e, pic in enumerate(d['imgs']):
                picture = imread(io.BytesIO(pic['picture']))
                ax[i].imshow(picture)
                ax[i].set_title(categories.loc[category_id, 'category_level3'][:12] + ' ('+ str(e) + ')')
                i = i + 1
        plt.tight_layout()
        plt.show()
                
    except Exception as IndexError:
        print(IndexError)