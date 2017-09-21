#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 22:48:05 2017

@author: hondagneu
"""

import pandas as pd
import numpy as np
import bson
import io
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import tables ##enables hdf tables
import cv2 #opencv helpful for storing image as array

from matplotlib.colors import ListedColormap

path = './'

categories = pd.read_csv('{}{}'.format(path,'category_names.csv'), index_col=0)

with open('{}{}'.format(path,'train_example.bson'),'rb') as b:
    df = pd.DataFrame(bson.decode_all(b.read()))

df['imgs'] = df['imgs'].apply(lambda rec: rec[0]['picture'])

df.set_index('category_id',inplace=True)

df[categories.columns.tolist()] = categories.loc[df.index]

df['imgs'] = df['imgs'].apply(lambda img: Image.open(io.BytesIO(img)))

fig,axs  = plt.subplots(7,5 ,figsize=(10,10))
title = df['category_level1'].str.split('-').str[0].str.strip()
# title += ','
# title += df['category_level2'].str.split('-').str[0].str.strip()
title = title.tolist()
axs = axs.flatten()
for i,ax in enumerate(axs):
    ax.imshow(df.iloc[i,1],
              interpolation='nearest', 
              aspect='auto')
    ax.set_title(title[i])
    #remove frame and ticks
    ax.axis('off')
    
plt.tight_layout()

