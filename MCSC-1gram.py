import numpy as np
import pandas as pd
import Data_loader as D

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

from Model import Net
from Custom import CustomDataset
from collections import Counter
from tqdm import tqdm

if __name__ == '__main__':
    read_path = 'D:virus/image/1gram_768/'
    
    temp = [[],[]]
    
    Loader = D.File_loader()
    data_a, label_a = Loader.read_files(read_path, interp = True)
    
    idx = np.argsort(label_a)
    
    sorted_data = data_a[idx]
    sorted_label = sorted(label_a)
        
    BATCH_SIZE = 64
    TOTAL = 30
    EPOCH = 500
    NUM_CLASS = 9
    LR = 0.005
    SEED = [s for s in range(TOTAL)]
    
    CUDA_N = 'cuda:0'
    
    # creating data indices for spliting
    full_dataset = CustomDataset(sorted_data, sorted_label)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # spliting
    torch.manual_seed(10)
    train_dataset, test_dataset = data.random_split(full_dataset, [train_size, test_size])
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = False)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    loss_total = []
    acc_total = []
    pred_total = []
    true_total = []
    
    
    for i in tqdm(range(TOTAL)):
        image_shape = full_dataset.x_data.shape[1:]
        
        device = torch.device(CUDA_N if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(SEED[i])
        net = Net(image_shape, NUM_CLASS)
        net.to(device)
        print(net)
        
        softmax = nn.Softmax()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum = 0.1)
        
        loss_list = []
        train_acc_list = []
        test_acc_list = []
        
        pred_temp = []
        true_temp = []
        
        for epoch in range(EPOCH):
            net.train()
            running_loss = 0
            total = train_size
            correct = 0 
            
            for step, images_labels in enumerate(train_loader):
                inputs, labels = images_labels
                inputs, labels = inputs.type(torch.FloatTensor).to(device), labels.type(torch.LongTensor).to(device)
                
                outputs = net(inputs)
                
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                _, pred = torch.max(outputs, dim=1)
                correct += (pred == labels).sum().item()
                
            train_acc = correct/total
            loss_list.append(running_loss)
            train_acc_list.append(train_acc)
            print('{}th- epoch: {}, train_loss = {}, train_acc = {}'.format(i+1, epoch, running_loss, train_acc))
            
            with torch.no_grad():
                net.eval()
                correct = 0
                total = test_size
                pt, tt = [], []
                
                for step_t, images_labels_t in enumerate(test_loader):
                    inputs_t, labels_t = images_labels_t
                    inputs_t, labels_t = inputs_t.type(torch.FloatTensor).to(device), labels_t.type(torch.LongTensor).to(device)
                    
                    outputs_t = net(inputs_t)
                    outputs_t = softmax(outputs_t)
                    
                    # test accuracy
                    _, pred_t = torch.max(outputs_t, dim = 1)
                    
                    pt.append(pred_t)
                    tt.append(labels_t)
                    
                    correct += (pred_t == labels_t).sum().item()
                    
                pred_temp.append(torch.cat(pt))
                true_temp.append(torch.cat(tt))
                
                test_acc = correct/total
                test_acc_list.append(test_acc)
                
                print('test Acc {}:'.format(test_acc))
                
        best_result_index = np.argmax(np.array(test_acc_list))
        loss_total.append(loss_list[best_result_index])
        acc_total.append(test_acc_list[best_result_index])
        pred_total.append(pred_temp[best_result_index].tolist())
        true_total.append(true_temp[best_result_index].tolist())
        
    file_name = 'res/1gram MCSC'
    torch.save(net.state_dict(), file_name +'.pth')
    
    loss_DF = pd.DataFrame(loss_total)
    loss_DF.to_csv(file_name+" loss.csv")
    
    acc_DF = pd.DataFrame(acc_total)
    acc_DF.to_csv(file_name +" acc.csv")
    
    pred_DF = pd.DataFrame(pred_total)
    pred_DF.to_csv(file_name +" pred.csv")
    
    true_DF = pd.DataFrame(true_total)
    true_DF.to_csv(file_name +" true.csv")