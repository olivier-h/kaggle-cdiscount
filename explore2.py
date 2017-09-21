#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 23:06:29 2017

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

CHUNK_SIZE = 10

with open('{}{}'.format(path,'train_example.bson'),'rb',buffering=True) as b:
    i=0
    lst = []
    for line in bson.decode_file_iter(b):
        if i%CHUNK_SIZE == 0 and i !=0:
            df = pd.DataFrame(lst)
            # df['imgs'] = df['imgs'].apply(
            #lambda reclst: ','.join(rec['picture'].hex() for rec in reclst))
            try:
                df.iloc[:,:-1].to_hdf('file.h5',
                                      key='train',
                                      format='table',
                                      append=True)
            except Exception as e:
                #catch disk full
                print('error: ',e)
                break
            lst=[]
        lst.append(line)
        i+=1

train = pd.read_hdf('file.h5',key='train')
#combine with categries
train = pd.merge(categories,train,right_on='category_id',left_index=True)

train.info()

cats = train['category_level1'].value_counts()
cats.head()

abbriv = cats.index.str.split('\W').str[0].str.strip()

sns.set_style('white')
fig,ax = plt.subplots(1,figsize=(12,6))
pal = ListedColormap(sns.color_palette('Paired').as_hex())
colors = pal(np.interp(cats,[cats.min(),cats.max()],[0,1]))
bars = ax.bar(range(1,len(cats)+1),cats,color=colors);
ax.set_xticks([]);
ax.set_xlim(0,len(cats))
ax1 = plt.twiny(ax)
ax1.set_xlim(0,len(cats))
ax1.set_xticks(range(1,len(abbriv)+1,1));
ax1.set_xticklabels(abbriv.values,rotation=90);
# sns.despine();
