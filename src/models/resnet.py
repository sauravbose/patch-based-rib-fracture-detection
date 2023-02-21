"""
@author: Saurav Bose
"""

'''
To run locally, change the following:
- directory paths
- code to load img_idx
- dataloader code
'''

import os,time
import sys, getopt, copy
import random
import dill, pickle

from PIL import Image
import SimpleITK as sitk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin

#Utility modules
# utilsDirectory = './'
utilsDirectory = '/mnt/isilon/prarie/boses1/Projects/RibFracture/src/models/'
sys.path.append(utilsDirectory)
from utils import GreyToDuplicatedThreeChannel, ImageData

#_____________________________________________________________________________________________#

#CONSTANTS
# metadatapath = '../../data/metadata/'
metadatapath = '/mnt/isilon/prarie/boses1/Projects/RibFracture/data/metadata/'
# imagepath = '../../data/processed_sample/'
imagepath = '/mnt/isilon/prarie/RibFractureStudy/data/processed/'
test_img_path = '/mnt/isilon/prarie/boses1/Projects/RibFracture/data/test_set/data/'
# resultpath = '../results/'
resultpath = '/mnt/isilon/prarie/boses1/Projects/RibFracture/src/results/'

#_____________________________________________________________________________________________#

#Extract command line arguements
'''
mode: OTS, OTSFT
learning_rate: for the optimizer
weight_decay: for the configure_optimizer
num_gpus: Number of GPUs to train on
batch_size: Size of the batch in each epoch
num_epoch: Max number of epochs to train for

'''
try:
    opts,args = getopt.getopt(sys.argv[1:],"m:l:w:g:b:e:t:v:",["mode=", "learning_rate=", "weight_decay=", "num_gpus=", "batch_size=", "num_epoch=", "train_meta=", "valid_meta="])

except getopt.GetoptError:
    print("Invalid input arguments")
    sys.exit(2)

for opt,arg in opts:
    if opt in ["-m","--mode"]:
        mode = arg

    elif opt in ["-l","--learning_rate"]:
        LEARNING_RATE = float(arg)

    elif opt in ["-w","--weight_decay"]:
        WEIGHT_DECAY = float(arg)

    elif opt in ["-g","--num_gpus"]:
        NUM_GPUS = int(arg)

    elif opt in ["-b","--batch_size"]:
        BATCH_SIZE = int(arg)

    elif opt in ["-e","--num_epoch"]:
        NUM_EPOCHS = int(arg)

    elif opt in ["-t","--train_meta"]:
        TRAIN_META_FILE = arg

    elif opt in ["-v","--valid_meta"]:
        VALID_META_FILE = arg

class LitResnet50(pl.LightningModule):

    def __init__(self):
        super().__init__()
        # self.num_classes = 2

        #Change this to torchvision.models.resnet18 to train Resnet18
        self.model = torchvision.models.resnet50(pretrained = True)
        if mode == 'OTS':
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)


    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        images = batch['image']
        labels = batch['label'].float()

        outputs = self.model(images)
        output_pred_proba = torch.sigmoid(outputs)

        # return self.model(x)
        return output_pred_proba, labels


    def loss_func(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
        return optimizer


    def train_dataloader(self):
        train_dataset = ImageData(metadata_path = metadatapath + TRAIN_META_FILE + '.csv', img_path = imagepath + 'train_val/data/', train_mode = "train")
        train_dl = DataLoader(train_dataset, batch_size = BATCH_SIZE, pin_memory = True, shuffle = True, num_workers = 2)
        # train_dl = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)

        return train_dl

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # It is independent of forward
        images = batch['image']
        labels = batch['label']

        outputs = self.model(images)
        criterion = self.loss_func()
        loss = criterion(outputs, labels)

        return {'loss':loss}


    def training_epoch_end(self, train_step_outputs):
        epoch_train_loss = torch.tensor([batch_result['loss'] for batch_result in train_step_outputs]).mean()

        self.log('epoch_train_loss', epoch_train_loss)


    def val_dataloader(self):
        val_dataset = ImageData(metadata_path = metadatapath + VALID_META_FILE + '.csv', img_path = imagepath + 'train_val/data/', train_mode = "val")
        val_dl = DataLoader(val_dataset, batch_size = BATCH_SIZE, pin_memory = True, num_workers = 2)
        # val_dl = DataLoader(val_dataset, batch_size = BATCH_SIZE)

        return val_dl


    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label'].float()

        outputs = self.model(images)
        criterion = self.loss_func()
        loss = criterion(outputs, labels)

        output_pred_proba = torch.sigmoid(outputs)

        return {'loss':loss, 'pred_proba':output_pred_proba, 'labels':labels}

    def validation_epoch_end(self, val_step_outputs):
        #output of validation_step is passed into validation_epoch_end. val_step_outputs is a list of dicts returns by validation_step, one entry per batch
        epoch_val_loss = torch.tensor([batch_result['loss'] for batch_result in val_step_outputs],device=self.device).mean()
        self.log('epoch_val_loss', epoch_val_loss,sync_dist=True)

        prediction_probabilities = torch.tensor([],device=self.device)
        prediction_labels = torch.tensor([],device=self.device)

        for batch_result in val_step_outputs:
            prediction_probabilities = torch.cat([prediction_probabilities, batch_result['pred_proba']])
            prediction_labels = torch.cat([prediction_labels, batch_result['labels']])

        self.prediction_probabilities = prediction_probabilities
        self.prediction_labels = prediction_labels
        self.val_outputs = val_step_outputs

    def test_dataloader(self):
        test_dataset = ImageData(metadata_path = metadatapath + 'test_set/test.csv', img_path = test_img_path, train_mode = "val")
        test_dl = DataLoader(test_dataset, batch_size = BATCH_SIZE, pin_memory = True, num_workers = 2)
        # test_dl = DataLoader(test_dataset, batch_size = BATCH_SIZE)

        return test_dl

    def test_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']

        outputs = self.model(images)
        output_pred_proba = torch.sigmoid(outputs)

        return {'pred_proba':output_pred_proba, 'labels':labels}

    def test_epoch_end(self, test_step_outputs):
        #output of validation_step is passed into validation_epoch_end. val_step_outputs is a list of dicts returns by validation_step, one entry per batch
        prediction_probabilities = torch.tensor([],device=self.device)
        prediction_labels = torch.tensor([],device=self.device)

        for batch_result in test_step_outputs:
            prediction_probabilities = torch.cat([prediction_probabilities, batch_result['pred_proba']])
            prediction_labels = torch.cat([prediction_labels, batch_result['labels']])

        self.test_prediction_probabilities = prediction_probabilities
        self.test_prediction_labels = prediction_labels

if __name__ == '__main__':
    seed_everything(987634982, workers = True)

    start_time = time.time()

    log_folder_name = f'Resnet50_{mode}_numGPU_{NUM_GPUS}_lr_{LEARNING_RATE}_wd_{WEIGHT_DECAY}_bs_{BATCH_SIZE}_numEpoch_{NUM_EPOCHS}_train_{TRAIN_META_FILE}_val_{VALID_META_FILE}_logs'
    logger = CSVLogger(resultpath, name=log_folder_name)

    checkpoint_callback = ModelCheckpoint(monitor='epoch_val_loss')
    early_stop_callback = EarlyStopping(monitor='epoch_val_loss', min_delta=0.0001,patience=10,verbose=False, mode='min')

    if NUM_GPUS > 1:
        trainer = Trainer(gpus = NUM_GPUS, accelerator='ddp', plugins=DDPPlugin(find_unused_parameters=False), callbacks=[checkpoint_callback, early_stop_callback], logger=logger, max_epochs = NUM_EPOCHS, progress_bar_refresh_rate=0)

    else:
        trainer = Trainer(gpus = NUM_GPUS, callbacks=[checkpoint_callback, early_stop_callback], logger=logger, max_epochs = NUM_EPOCHS, progress_bar_refresh_rate=0)

    model = LitResnet50()
    trainer.fit(model)

    model.eval()
    trainer.test()

    print('Took', time.time()-start_time)
    print(f'Resnet50_{mode}_numGPU_{NUM_GPUS}_lr_{LEARNING_RATE}_wd_{WEIGHT_DECAY}_bs_{BATCH_SIZE}_numEpoch_{NUM_EPOCHS}_train_{TRAIN_META_FILE}_val_{VALID_META_FILE}')
    #To extract the best model checkpoint path
    # print(trainer.checkpoint_callback.best_model_path)

    results = {'res':model.val_outputs,'prediction_probabilities':model.prediction_probabilities, 'prediction_labels':model.prediction_labels,'test_prediction_probabilities':model.test_prediction_probabilities, 'test_prediction_labels':model.test_prediction_labels}

    torch.save(results, resultpath + f'Resnet50_{mode}_numGPU_{NUM_GPUS}_lr_{LEARNING_RATE}_wd_{WEIGHT_DECAY}_bs_{BATCH_SIZE}_numEpoch_{NUM_EPOCHS}_train_{TRAIN_META_FILE}_val_{VALID_META_FILE}_predictions.pth')
