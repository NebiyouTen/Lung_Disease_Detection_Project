# -*- coding: utf-8 -*-
"""binaryOpenI

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CIE0929g9sFgV-KV6o1unhnd6RSas3zD
"""

# http://pytorch.org/
from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'
print (accelerator)

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.3.0.post4-{platform}-linux_x86_64.whl torchvision
import torch
!pip install kaggle

!pip install --no-cache-dir -I pillow
!pip install Pillow==4.1.1 

!pip install PIL
!pip install  image

!wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
!wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz

!tar xvf NLMCXR_png.tgz
!mkdir png && mv `ls *.png` png

!tar xvf NLMCXR_reports.tgz

!ls

import numpy as np 
import torch.nn as nn
from PIL import Image
import os
from matplotlib.pyplot import imshow
import torch.nn.functional as F
# %matplotlib inline
import torchvision.models.densenet as dnet
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import parse
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch 
import torch.optim as optim
import math
import unicodedata
import re
import torch.nn.utils.rnn as rnn

xml_dir = "ecgen-radiology/"
dict_image_MeSH = {}
import os
for id ,filename in enumerate(os.listdir(xml_dir)):
    doc = parse(xml_dir+"/"+filename).getroot()
    major = ''
    for element in doc.findall('MeSH/major'):
        major +=element.text + '/'
    for element in doc.findall('parentImage'):
        dict_image_MeSH[element.attrib['id']] = major

labels = np.zeros(len(dict_image_MeSH.values()))
for c,i in enumerate(dict_image_MeSH.values()): 
    if(i.lower()=='normal/'):
        continue;
    labels[c]=1

def to_tensor(numpy_array):
    return torch.from_numpy(numpy_array).float()

def to_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor,requires_grad=True)

def to_variable2(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)
  
from PIL import Image
def get_images(images):
    image_dir = 'png/'
    image_data = torch.zeros(len(images),224,224,3)
    for k,image in enumerate(images):
        image  = Image.open(image_dir+image+'.png').resize((224,224),Image.ANTIALIAS).convert('RGB')
        i = np.array(image)
        i=train_train(image)
        image_data[k]=i
    return image_data
def get_images2(images):
    image_dir = 'png/'
    image_data = torch.zeros(len(images),224,224,3)
    for k,image in enumerate(images):
        image  = Image.open(image_dir+image+'.png').resize((224,224),Image.ANTIALIAS).convert('RGB')
        i = np.array(image)
        i=test_trans(image)
        image_data[k]=i
    return image_data
  
  
  
_use_shared_memory=False
mini_batch_size=16
def my_collate(batch):
    numpy_labels = labels
    
    keys = np.array(list(dict_image_MeSH.keys()))
    images = keys[batch]
    
    image_array = get_images(images)
    tr= image_array
    tr = tr.transpose(3,1)
    return tr,to_tensor(numpy_labels[batch])
def my_collate2(batch):
    numpy_labels = labels
    
    keys = np.array(list(dict_image_MeSH.keys()))
    images = keys[batch]
    
    image_array = get_images2(images)
    tr= image_array
    tr = tr.transpose(3,1)
    return tr,to_tensor(numpy_labels[batch])

train_loader = DataLoader(
        np.arange(labels.shape[0]-1000), shuffle=True,
        batch_size=mini_batch_size, collate_fn=my_collate)


test_loader = DataLoader(
        np.arange(labels.shape[0]-1000,labels.shape[0]), shuffle=False,
        batch_size=mini_batch_size, collate_fn=my_collate2)


from torchvision import datasets, models, transforms
train_train=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
test_trans=transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

import pandas as pd
val_table = pd.read_pickle('val_table.pkl')

def my_collate2(batch):
    label = val_table.iloc[batch,2]
    img_name = val_table.iloc[batch,0]
    image_array = get_images2(img_name)
    tr= image_array
    tr = tr.transpose(3,1)
    return tr,to_tensor(np.array(label))

test_loader = DataLoader(
        np.arange(len(val_table)), shuffle=False,
        batch_size=1, collate_fn=my_collate2)

np.arange(3,6)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
my_net = model_ft.cuda()
loss_fn = torch.nn.CrossEntropyLoss() 
loss_fn.cuda()

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

def validate():
    from sklearn.metrics import confusion_matrix
    print("Testing started");
    my_net.eval()
    losses = []
    train_loss = 0
    correct = 0
    total = 0
    import time  
    startTime = time.time()
    pt = []
    lt = []
    probs = []
    for batch_index, (input_val, label) in enumerate(test_loader):
        (input_val, label) = to_variable(input_val), to_variable2(label)
        prediction = my_net(input_val)  # Feed forward
        loss = loss_fn(prediction, label.long().view(-1))  # Compute losses
        losses.append(loss.data.cpu().numpy())
        _, predicted = torch.max(prediction.data, 1)
#         print (predicted.cpu().numpy())
        total += label.size(0)
        correct += predicted.eq(label.data.view(-1).long()).cpu().sum()
        
        if (batch_index == 0):
            probs = prediction.cpu().data.numpy()
            pt = predicted.cpu().numpy().reshape(-1)
            lt = label.cpu().data.long().view(-1).numpy().reshape(-1)
        else:
            probs = np.vstack((probs,prediction.cpu().data.numpy()))
            lt = np.hstack((lt,label.cpu().data.long().view(-1).numpy().reshape(-1)))
            pt = np.hstack((pt,predicted.cpu().numpy().reshape(-1)))
#     print(type(np.array(lt)),np.array(lt).shape)
    print (confusion_matrix(lt.flatten(),pt.flatten()))
    return round(100 *correct/total,4),probs,lt

my_net.load_state_dict(torch.load("model.pt"))

accuracy,probs,lab = validate()
print (accuracy)

print (np.load('val_probs.npy'))
# np.save('val_probs',probs)

print (probs.shape)
print (lab.shape)
!pip install scikit-plot

import scikitplot as skplt
import matplotlib.pyplot as plt

y_true = lab# ground truth labels
y_probas = probs# predicted probabilities generated by sklearn classifier
skplt.metrics.plot_roc_curve(y_true, y_probas)

plt.figure(facecolor="white")

plt.show()

combined_est = np.load('combined_estimates.npy')
combined_prob = np.load('combined_probs.npy')
# print (combined_est.shape)
print (combined_prob[0,0])
print (confusion_matrix(lab,combined_est[:,1]))

predictions_prob = probs
false_positive_rate, recall, thresholds = roc_curve(lab, predictions_prob[:,1])
false_positive_rate2, recall2, thresholds = roc_curve(lab, combined_prob[:,0,1])
false_positive_rate3, recall3, thresholds = roc_curve(lab, combined_prob[:,1,1])
roc_auc = auc(false_positive_rate, recall)
fig=plt.plot(false_positive_rate, recall, 'g', label = 'AUC %s = %0.2f' % ('Image Classi- Model', roc_auc))
plt.plot(false_positive_rate2, recall2, 'r', label = 'AUC %s = %0.2f' % ('Text Classi- Model', auc(false_positive_rate2, recall2)))
plt.plot(false_positive_rate3, recall3, 'b', label = 'AUC %s = %0.2f' % ('Combined Model', auc(false_positive_rate3, recall3)))
plt.plot([0,1], [0,1], 'r--')
plt.legend(loc = 'lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve')
ax = plt.axes()
ax.grid(color='k',linestyle='-', linewidth=0.2)
ax.set_facecolor('white')
print(plt.axes())

tprs = np.array(recall)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)

tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std


plt.fill_between(false_positive_rate, recall -0.05 ,np.minimum(1,recall + 0.05), color='grey', alpha=0.3)

plt.fill_between(false_positive_rate2, recall2 -0.04 ,np.minimum(1,recall2 + 0.04), color='green', alpha=0.3)
# plt.fill_between(false_positive_rate3, recall3 -0.04 ,np.minimum(1,recall3 + 0.04), color='blue', alpha=0.3)
# plt.set

print(thresholds[0])

from sklearn.metrics import confusion_matrix
print("Training started");
num_epochs = 250
learning_rate=5e-5
best_val_accuray = validate()
print ('Val acc ',best_val_accuray)
for epoch in range(num_epochs):
    optim = torch.optim.Adam( my_net.parameters(),lr=learning_rate)
    my_net.train()
    losses = []
    train_loss = 0
    correct = 0
    total = 0
    import time  
    startTime = time.time()
    for batch_index, (input_val, label) in enumerate(train_loader):
        optim.zero_grad()  # Reset the gradients
        (input_val, label) = to_variable(input_val), to_variable2(label)
        prediction = my_net(input_val)  # Feed forward
        loss = loss_fn(prediction, label.long().view(-1))  # Compute losses
        loss.backward()  # Backpropagate the gradients
        losses.append(loss.data.cpu().numpy())
        optim.step()  # Update the network 
        _, predicted = torch.max(prediction.data, 1)
        total += label.size(0)
        correct += predicted.eq(label.data.view(-1).long()).cpu().sum()
        if (batch_index % 10)==0: 
            print("Batch:",batch_index,"/",len(train_loader),"| loss ",np.mean(losses)
                , ' | correct ',correct, ' of ',total," | Accuracy ", round(100 *correct/total,4) )
            print (confusion_matrix(label.long().view(-1).cpu().data.numpy(),predicted.cpu().numpy()))
    val_acc = validate()
    if (best_val_accuray<val_acc):
        learning_rate =learning_rate/2
        best_val_accuray = val_acc
        print("Model saved Val-accuray improved ",val_acc, 'model name ',"models_DenseNet_"+str(epoch)+str(batch_index)+".pt")
        torch.save(my_net.state_dict(), "models_DenseNet_"+str(epoch)+str(batch_index)+".pt")
    else:
        print("Val accuray is ", val_acc," Not an improvement")
    print("--------------------------")
    print("Epoch {} Loss: {:.4f}".format(epoch, np.asscalar(np.mean(losses))))
    print ('Time per epoc ',(time.time()-startTime)/60)

from google.colab import files
# files.download("val_probs.npy")
files.upload()

!ls -tr

!ls -rt
