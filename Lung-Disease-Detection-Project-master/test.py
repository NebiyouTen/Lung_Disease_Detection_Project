import os
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision import transforms
from create_dict import *
from models import *
from PIL import Image
import torch.nn.functional as F
import numpy as np
from utils import *

name            = 'train'
train_table     = pd.read_pickle(os.path.join('data','{}_table.pkl'.format(name)))
train_reports   = train_table.report
train_reports   = [filter_sen(report) for report in train_reports]

name          = 'val'
val_table     = pd.read_pickle(os.path.join('data','{}_table.pkl'.format(name)))
val_reports   = val_table.report
val_reports   = [filter_sen(report) for report in val_reports]
lang_word, lang_char = buildDict([val_reports, train_reports])

vocab_size = len(lang_word.word2index)
#print('vocab: ', vocab_size)
rnn_model = ReportAnalysis(vocab_size=vocab_size,
                       embedding_dim=256, 
                       hidden_dim=64, 
                       batch_size=1, 
                       num_layers=1,
                       label_size=1,
                       bidirectional=False)

rnn_model.load_state_dict(torch.load('models/report.pkl'))
cnn_model = models.resnet18(pretrained=True)
num_ftrs  = cnn_model.fc.in_features
cnn_model.fc = nn.Linear(num_ftrs, 2)

cnn_model.cuda()
cnn_model.load_state_dict(torch.load('model.pt'))

#print(table)
transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
estms = []
probs = []
val_probs = np.load('val_probs_update.npy')
# print ('first ',val_probs.shape,val_probs[0].shape)
for idx in range(len(val_table)):
    report   = val_table.iloc[idx, 1]
    img_name = val_table.iloc[idx, 0] 
    label    = val_table.iloc[idx, 2]
    image    = Image.open(os.path.join('data/images', img_name+'.png')).resize((224,224),Image.ANTIALIAS).convert('RGB')
    image    = transform(image)
    estm, prob     = infer_from_img_report(cnn_model, rnn_model, image, report, lang_word,torch.FloatTensor(val_probs[idx,:].tolist()))
    estms.append(estm)
    probs.append(prob)
    print(idx,'/',len(val_table), estm, label)
estms  = np.array(estms)
probs = np.array(probs)
labels = np.array(val_table.iloc[:, 2])
# print(estms.shape)
rnn_acc = sum(estms[:, 0] == labels)/len(labels)
cnn_acc = sum(estms[:, 1] == labels)/len(labels)
combined_acc = sum(estms[:, 2] == labels)/len(labels)
np.save('combined_estimates',estms)
np.save('combined_probs',probs)
print('rnn acc: ', rnn_acc, 'cnn acc: ', cnn_acc, 'combined: ', combined_acc)
    
                       
'''
labels   = [label for label in table.label]
print('abnormal: ', sum(labels)/len(labels))
print('normal: ', 1-(sum(labels)/len(labels)))
prop_normal = 0.635
data_dist = torch.FloatTensor([1-prop_normal, prop_normal])
print(data_dist)
print(len(table))
plt.hist(np.array(labels))
plt.show()
'''
