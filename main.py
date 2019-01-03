import torch
import torch.nn as nn
import numpy as np
from models import BagOfWords_SST, BagOfWords_SNLI, LSTM_RNN_SST, LSTM_RNN_SNLI, BCN_SST, BCN_SNLI
from data import get_dataset

use_cuda = torch.cuda.is_available()

def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()

def accuracy(ys, ts):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = torch.eq(torch.max(ys, 1)[1], ts)
    # averaging the one-hot encoded vector
    return torch.mean(correct_prediction.float())

def train_model(model, dataset):

    model = model.lower()
    DATASET = dataset.lower()
    BATCH_SIZE = 128 if DATASET == "snli" else 32

    train_iter, test_iter, TEXT, LABEL = get_dataset(DATASET, use_cuda, BATCH_SIZE)

    net = None
    if model == "bow":
        if DATASET == "sst-5" or DATASET == "sst-2":
            net = BagOfWords_SST(TEXT, LABEL)
        elif DATASET == "snli":
            net = BagOfWords_SNLI(TEXT, LABEL)
    elif model == "lstm":
        if DATASET == "sst-5" or DATASET == "sst-2":
            net = LSTM_RNN_SST(TEXT, LABEL)
        elif DATASET == "snli":
            net = LSTM_RNN_SNLI(TEXT, LABEL)
    elif model == "bcn":
        if DATASET == "sst-5" or DATASET == "sst-2":
            net = BCN_SST(TEXT, LABEL)
        elif DATASET == "snli":
            net = BCN_SNLI(TEXT, LABEL)
    
    if net == None:
        return

    if use_cuda:
        net.cuda()

    criterion = net.criterion
    optimizer = net.optimizer

    max_iter = 30000
    eval_every = 1000
    log_every = 200


    train_loss, train_accs = [], []

    best_val_accuracy = 0
    net.train()
    for i, batch in enumerate(train_iter):
        if i % eval_every == 0 and i != 0:
            net.eval()
            val_losses, val_accs, val_lengths = 0, 0, 0
            for val_batch in test_iter:
                if DATASET == "sst-5" or DATASET == "sst-2":
                    output = net(val_batch.text)
                elif DATASET == "snli":
                    output = net(val_batch.premise, val_batch.hypothesis)

                val_losses += criterion(output['out'], val_batch.label) * val_batch.batch_size
                val_accs += accuracy(output['out'], val_batch.label) * val_batch.batch_size
                val_lengths += val_batch.batch_size

            val_losses /= val_lengths
            val_accs /= val_lengths

            best_val_accuracy = best_val_accuracy if best_val_accuracy > get_numpy(val_accs) else get_numpy(val_accs)
            
            print("valid, it: {} loss: {:.4f} accs: {:.4f}\n".format(i, get_numpy(val_losses), get_numpy(val_accs)))
            
            net.train()
        
        if DATASET == "sst-5" or DATASET == "sst-2":
            output = net(batch.text)
        elif DATASET == "snli":
            output = net(batch.premise, batch.hypothesis)

        batch_loss = criterion(output['out'], batch.label)
        
        train_loss.append(get_numpy(batch_loss))
        train_accs.append(get_numpy(accuracy(output['out'], batch.label)))
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        if i % log_every == 0 and i != 0 and i < max_iter:        
            print("train, it: {} loss: {:.4f} accs: {:.4f}".format(i, 
                                                                np.mean(train_loss), 
                                                                np.mean(train_accs)))
            train_loss, train_accs = [], []
            
        if max_iter < i:
            break

    print(str(type(net)), 'with dataset', DATASET, 'and batch size', BATCH_SIZE, 'got a max accuracy of', str(float(best_val_accuracy)))