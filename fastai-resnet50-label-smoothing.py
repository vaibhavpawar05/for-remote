from fastai.vision.all import *
from fastai.vision.learner import _update_first_layer

import os
import pandas as pd
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
import gc
import shutil

import argparse

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-bs", type=int)
args = parser.parse_args()

bs = args.bs

# Set the path to the photos.
path = Path('/media/manu/only_one_cropped_data_aug_resized_images/jordan1_foot_bed_2/')

torch.cuda.empty_cache()
 
# # Getting image files.
train_files = get_image_files(path/'train')
val_files = get_image_files(path/'val')

jorda1_footbed_stitching = DataBlock(blocks = (ImageBlock, CategoryBlock),
                                     get_items=get_image_files, 
                                     splitter=GrandparentSplitter(train_name='train', valid_name='val'), #RandomSplitter(valid_pct=0.15, seed=42)
                                     get_y=parent_label,
                                     batch_tfms=None)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device")

# Creating train dataloader
train_dls = jorda1_footbed_stitching.dataloaders(path, num_workers=4, batch_size=bs, device=device)

# Adding for gpu support
train_dls = train_dls.cuda()

model_path = Path('/dbfs/tmp/vaibhav/fastai-resnet50-label-smoothing')

learn = cnn_learner(train_dls, resnet50, loss_func=LabelSmoothingCrossEntropy(), 
                    path=model_path, metrics=[error_rate, accuracy, Precision(), Recall()])

learn.model = learn.model.cuda()

print(learn.summary())

learn.fit_one_cycle(10, 1e-3, cbs=[SaveModelCallback(fname='fine_tuning_last_few_layers', every_epoch=True, monitor='accuracy')])
