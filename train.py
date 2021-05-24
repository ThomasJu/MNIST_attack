import time
import copy

import torch
import torch.nn as nn
import torch.optim as optimizer
from sklearn.metrics import recall_score, confusion_matrix, f1_score, accuracy_score

def train_model(net, trn_loader, val_loader, optim, num_epoch=50,
        collect_cycle=30, device='cpu', verbose=True):
    # Initialize:
    # -------------------------------------
    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_acc, best_conf_mat = None, 0, None
    net.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    if verbose:
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    for epoch in range(num_epoch):
        ############## Training ##############
        net.train()
        for x, label in trn_loader:
            num_itr += 1
            x, label = x.to(device), label.to(device)
            
            ############ TODO: calculate loss, update weights ############
            optim.zero_grad()
            pred = net(x)
            loss = loss_fn(pred, label)
            pred = torch.argmax(pred, dim=1)
            loss.backward()
            optim.step()
            ###################### End of your code ######################
            
            if num_itr % collect_cycle == 0:  # Data collection cycle
                train_loss.append(loss.item())
                train_loss_ind.append(num_itr)
        if verbose:
            print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}-- train accuracy = {3:.4f}'.format(
                epoch + 1,
                num_itr,
                loss.item(),
                accuracy_score(label.numpy(), pred.numpy())
                ))

        ############## Validation ##############
        acc, wf1, loss, confusion_mat = get_validation_performance(net, loss_fn, val_loader, device)
        val_loss.append(loss)
        val_loss_ind.append(num_itr)
        if verbose:
            print("Validation weighted macro F-1: {0:.4f} accuracy: {1:.4f}".format(wf1, acc))
        # update stats
        if acc > best_acc:
#             best_model = copy.deepcopy(net)
            best_acc, best_conf_mat = acc, confusion_mat
    
    t_end = time.time()
    
    print('Training lasted {0:.2f} minutes'.format((t_end - t_start)/60))
    print(f'best_acc: { best_acc }')
    print('------------------------ Training Done ------------------------')
    stats = {
             'Accuracy': best_acc,
             'confusion_mat': best_conf_mat
    }

    
    
    return best_model, stats


def get_validation_performance(net, loss_fn, data_loader, device):
    """
    Evaluate model performance.
    Input:
        - net: model
        - loss_fn: loss function
        - data_loader: data to evaluate, i.e. val or test
        - device: device to use
    Return:
        - uar: unweighted average recall on the data
        - w_f1: weighted macro F-1 score
        - loss: loss of the last batch
        - confusion_mat: confusion matrix of predictions on the data
    """
    net.eval()
    net.to(device)
    y_true = [] # true labels
    y_pred = [] # predicted labels

    with torch.no_grad():
        for x, label in data_loader:
            x, label = x.to(device), label.to(device)
            loss = None # loss for this batch
            pred = None # predictions for this battch

            ######## TODO: calculate loss, get predictions #########
            pred = net(x)
            loss = loss_fn(pred, label)
            pred = torch.argmax(pred, dim=1)
            ####### You don't need to average loss across iterations #####
            
            ###################### End of your code ######################
            y_true += list(label.numpy())
            y_pred += list(pred.numpy())
            
    acc = accuracy_score(y_true, y_pred)
    w_f1 = f1_score(y_true, y_pred, labels=list(range(7)), average='weighted')
    confusion_mat = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    
    return acc, w_f1, loss.item(), confusion_mat
