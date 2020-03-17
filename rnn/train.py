import os
import torch
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
import parser
import models
import data
# import test

import numpy as np
import torch.nn as nn

import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score



def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)



def evaluate(model, data_loader):
    ''' set model to evaluate mode '''
    model.eval()

    preds = []
    gts = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.double().reshape(1, imgs.shape[0], 253)

            pred,_ = model(imgs.cuda())
            # _, pred = torch.max(pred, dim=1)
            #
            # pred = np.array(pred.cpu().numpy())
            pred = pred.cpu().numpy()[:, 1]
            gt = gt.numpy()

            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    # return accuracy_score(gts, preds)
    return roc_auc_score(gts, preds)

if __name__ == '__main__':

    args = parser.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    df1 = pd.read_csv(args.data_dir).astype('float64')
    x_tr = df1.drop(columns=['Class'])
    y_tr = df1['Class']
    X_train, X_test, y_train, y_test = train_test_split(x_tr, y_tr, random_state=1)

    train_loader = torch.utils.data.DataLoader(data.DATA(X_train,y_train),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(data.DATA(X_test,y_test),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=True)

    ''' load model '''
    print('===> prepare model ...')

    model = models.rnn().double()



    model.cuda()  # load model to gpu

    ''' define loss '''
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)




    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0
    

    for epoch in range(1, args.epoch + 1):
        model.train()

        for idx, (imgs, cls) in enumerate(train_loader):

            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, (int(idx) + 1), round(len(train_loader)))
            iters += 1
            ''' move data to gpu '''

            imgs=imgs.double().reshape(1, imgs.shape[0],253)


            output,_ = model(imgs.double().cuda())

            ''' compute loss, backpropagation, update parameters '''
            loss = criterion(output, cls.long().cuda()) # compute loss

            optimizer.zero_grad()  # set grad of all parameters to zero
            loss.backward()  # compute gradient for each parameters
            optimizer.step()  # update parameters



            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print('\r', train_info, end='' )

        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate(model, val_loader)

            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
                best_acc = acc

        ''' save model '''
        #save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))