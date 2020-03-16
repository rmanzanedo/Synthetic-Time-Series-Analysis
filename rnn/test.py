import os
import torch

import parser
import models
import data_test as data

import csv

import numpy as np
import pandas as pd



if __name__ == '__main__':

    args = parser.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    X_test = pd.read_csv('../MyriadChallenge/TestMyriad.csv').astype('float64')
    val_loader = torch.utils.data.DataLoader(data.DATA(X_test),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=True)
    ''' load model '''
    print('===> prepare model ...')
    model = models.rnn().double()

    checkpoint = torch.load(args.load_model)
    model.load_state_dict(checkpoint)

    model.cuda()
    model.eval()

    preds = []

    with torch.no_grad():
        for idx, (imgs) in enumerate(val_loader):
            train_info = 'Epoch: [{0}/{1}]'.format(idx + 1, len(val_loader))

            ''' move data to gpu '''

            imgs=imgs.double().reshape(1, imgs.shape[0],253)

            pred, _ = model(imgs.double().cuda())

            pred = pred.cpu().numpy()[:, 1]
            preds.append(pred)

            print(train_info)

            group = []
            cls_group = []

    preds = np.concatenate(preds)
    with open(args.save_csv, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(['class'])
        for i in preds:
            wr.writerow([str(i)])
