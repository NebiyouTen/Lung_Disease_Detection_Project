import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import time
import pickle
from models import * 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from data_loader import *
from utils import *
from create_dict import *

def validate(model, data_loader, criterion, bsz=4):
    
    val_loss = 0
    correct  = 0
    model.eval()

    for batch_index, batch_dict in enumerate(data_loader):

        reports   = to_var(batch_dict['reports'])
        print('reports: ', reports.shape)
        seq_lens  = batch_dict['seq_lens']
        labels    = to_var(batch_dict['labels'])
        if reports.shape[1]!=bsz:break
        out       = model(reports, seq_lens)
        pred      = out.data>0.5

        predicted = pred.eq(labels.data.view_as(pred).byte())
        correct  += predicted.sum()

        loss      = criterion(out.squeeze(1), labels.float())

        val_loss += loss.data.sum()
        break
    print('val acc : ', correct/770)
    return val_loss

def main(args):

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    data_dir  = 'data/images'
    name      = 'train'
    train_table     = pd.read_pickle(os.path.join('data','{}_table.pkl'.format(name)))	
    train_reports   = train_table.report
    train_reports   = [filter_sen(report) for report in train_reports]
    name          = 'val'
    val_table     = pd.read_pickle(os.path.join('data','{}_table.pkl'.format(name)))	
    val_reports   = val_table.report
    val_reports   = [filter_sen(report) for report in val_reports]


    lang_word, lang_char = buildDict([val_reports, train_reports])

    train_ds    = OpenIReportDataset(lang_word, data_dir, name='train')
    train_data_loader = data.DataLoader(train_ds, 
                         batch_size = args.batch_size,
                         shuffle = True,
                         collate_fn = report_collate)

    val_ds      = OpenIReportDataset(lang_word, data_dir, name='val')
    val_data_loader = data.DataLoader(val_ds, 
                         batch_size = args.batch_size,
                         shuffle = True,
                         collate_fn = report_collate)
    vocab_size = len(lang_word.word2index)
    print('vocab: ', vocab_size)
    model = ReportAnalysis(vocab_size=vocab_size,
                           embedding_dim=args.embed_size, 
                           hidden_dim=args.hidden_size, 
                           batch_size=args.batch_size, 
                           num_layers=args.num_layers,
                           label_size=1,
                           bidirectional=args.bi)

    criterion = nn.BCELoss()
    print(model)

    model_path = os.path.join(args.model_dir, args.model_name)	
    try:

        model.load_state_dict(torch.load(model_path))
        print('using pre-trained model: ', model_path)
    except:
        print("using new model")

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)


    print('validating.....')
    n_train_batchs   = len(train_ds)//args.batch_size
    n_val_batchs     = len(val_ds)//args.batch_size
    print('val ds: ',  len(val_ds))
    best_val = validate(model, val_data_loader, 
                        criterion, bsz=args.batch_size)/n_val_batchs
    return
    print("starting val loss {:f}".format(best_val))

    for epoch in range(args.num_epochs):

        model.train()
        epoch_loss = 0
        correct    = 0
        epoch_time = time.time()
        for batch_index, batch_dict in enumerate(train_data_loader):
            optimizer.zero_grad()

            reports   = to_var(batch_dict['reports'])
            if reports.shape[1]!=args.batch_size:break
            seq_lens  = batch_dict['seq_lens']
            labels    = to_var(batch_dict['labels'])

            model.hidden = model.init_hidden()

            out       = model(reports, seq_lens)
            pred      = out.data>0.5
            predicted = pred.eq(labels.data.view_as(pred).byte())
            correct  += predicted.sum()

            loss      = criterion(out.squeeze(1), labels.float())
            epoch_loss += loss.data.sum()

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm(model.parameters(), 200)
            optimizer.step()

            # Print log info
            if batch_index % args.log_step == 0:
                print('|batch {:4d}|train loss {:5.2f}|'.format(
                batch_index+1,
                epoch_loss / (batch_index+1)))

        val_loss = validate(model, val_data_loader, criterion, args.batch_size)/n_val_batchs
        print('train acc: ', correct/len(train_ds))
        print('=' * 83)
        print(
            '|epoch {:3d}|valid loss {:5.4f}|'
            'train loss {:8.4f}'.format(
                epoch + 1,
                val_loss,
                epoch_loss/n_train_batchs))
            # Save the models
        if (epoch+1) % args.save_step == 0:
                if val_loss < best_val:
                    best_val = val_loss
                    print("Found new best val")
                    torch.save(model.state_dict(),model_path)
                    if val_loss<0.6:
                        break
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='models/' ,
                        help='path for saving trained models')
    parser.add_argument('--data_dir', type=str, default='data/' ,
                        help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1,
                        help='step size for saving trained models')
    parser.add_argument('--model_name', type=str, default='report.pkl',
                        help='path for trained encoder')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=64 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                       help='number of layers in lstm')  
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--split', type=float, default=0.9)
    parser.add_argument('--bi', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    
    args = parser.parse_args()
    main(args)
