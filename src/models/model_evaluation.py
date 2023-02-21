# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 18:34:42 2021

@author: danie
"""
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch

import os
import pandas as pd
import torchvision.models as models
from torch_snippets import * 
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import random
import torchvision
import numpy as np
import statistics 
torch.cuda.empty_cache()

from transforms import imagenet_tranforms
from dataloader import RibfractureDataset

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix
import itertools
#%% RELOAD THE MODEL 
save_path = "K:\CHOP Studies\RibFracture\models"
image_data_path = "K:\\CHOP Studies\\RibFracture\\Data\AP_ChestLabeled_All\\nii_cropped_resized"
data_dir = "K:\CHOP Studies\RibFracture\Data\processed"

#%%Reloading the model
def get_model():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Sequential(nn.Flatten(),
                             nn.Linear(512, 128),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.ReLU(inplace=True),
                             nn.Linear(128, len(id2int)))
    
    loss_fn = nn.CrossEntropyLoss(weight=normedWeights) #nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
    return model.to(device), loss_fn, optimizer

model, loss_fn, optimizer = get_model()


model.load_state_dict(torch.load(os.path.join(save_path, 'resnet_18_test.pth')))
for param in model.parameters():
    param.requires_grad = True
model.eval();


#%% VISUALIZE THE RESULTS
validation = pd.read_csv(os.path.join(data_dir, 'validation.csv'))
validation = validation[validation.label != 1]
print(len(training), len(validation))

val_dataset = RibfractureDataset(df = validation, 
                                 transform=transforms.Compose([imagenet_tranforms()]))
val_dataloader = DataLoader(val_dataset, batch_size=1,shuffle=True, num_workers=0)


int2id = {0:'NoFracture', 1:'Fracture'}
#info = 'Actual: ' + int2id[y[0]] + ' , Predicted: ' + int2id[pred_fracture[0]]
for ix, batch in enumerate(iter(val_dataloader)):
    if ix == 0:
        x, y = batch['image'].to(device), batch['label'].float().to(device)
        y = y.cpu().detach().numpy()
        fracture = model(x.to(device))
        print(fracture)
        pred_fracture = torch.max(fracture, 1).indices.cpu().detach().numpy()
        info = 'Actual: ' + int2id[y[0]] + ' , Predicted: ' + int2id[pred_fracture[0]]
        show(x[0,0], title=info, sz=10)   
        break
    
    
#%%
model.to('cuda')
y_true, y_score, y_pred = [], [], []
for ix, batch in enumerate(iter(val_dataloader)):
    
    x, y = batch['image'].to('cuda'), batch['label'].float().to('cuda')
    y = y.cpu().detach().numpy()
    pred_fracture = torch.max(model(x), 1).indices.cpu().detach().numpy()
    probabilities = F.softmax(model(x), dim = 1).cpu().detach().numpy()[0]
    prob = probabilities[1] #int(y)
    '''
    print('Actual: ', int(y),
          'Pred: ', pred_fracture[0],
          'Probablities: ', probabilities,
         'Correct Prob Predict: ', probabilities[1])
    '''
    y_true.append(int(y))
    y_score.append(prob)
    y_pred.append(pred_fracture[0])
    
    #if ix == 125:
    #    break
#%%



# Compute fpr, tpr, thresholds and roc auc
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_true, y_score)

# calculate AUC
auc = roc_auc_score(y_true, y_score)
print('AUC: %.3f' % auc)

# calculate F1 score
f1 = f1_score(y_true, y_pred)
print('F1-Score: %.3f' % f1)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
#%%
cnf_matrix = confusion_matrix(y_true, y_pred)

#%%

#Evaluation of Model - Confusion Matrix Plot
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['No Fracture','Fracture'],
                      title='Confusion matrix')

#%% GRAD-CAM
class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        self.gradients = [grad_output[0]] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []        
        return self.model(x)

class BaseCAM:
    def __init__(self, model, target_layer, use_cuda=False):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        return output[:, target_category]

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_tensor.shape[2:][::-1])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(GradCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor, 
                              target_category, 
                              activations, grads):
        return np.mean(grads, axis=(1, 2))
    
#%%
target_layer = model.layer4[-1]

int2id = {0:'NoFracture', 1:'Fracture'}
for ix, batch in enumerate(iter(val_dataloader)):
    if ix == 0:
        x, y = batch['image'].to(device), batch['label'].float().to(device)
        y = y.cpu().detach().numpy()
        
        # Running Grad-CAM
        cam = GradCAM(model=model, target_layer=target_layer)
        grayscale_cam = cam(input_tensor=x, target_category=1)
        
        # Model Prdiction
        fracture = model(x.to(device))
        pred_fracture = torch.max(fracture, 1).indices.cpu().detach().numpy()
        info = 'Actual: ' + int2id[y[0]] + ' , Predicted: ' + int2id[pred_fracture[0]]

        # subplots
        #f, axarr = plt.subplots(1,2, figsize=(15,8))
        #f.suptitle(info)
        #axarr[0].imshow(x[0,0].cpu(), cmap = 'gray')
        #axarr[1].imshow(grayscale_cam) 
        break
fig = plt.figure(figsize=(15,8))
fig.suptitle(info, fontsize=20)
plt.imshow(x[0,0].cpu(), cmap='gray') # I would add interpolation='none'
plt.imshow(grayscale_cam, cmap='jet', alpha=0.3) # interpolation='none'

