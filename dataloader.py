import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

def data_split(data,train_size, val_size, test_size, random_state=3):
    # data shuffle
    if random_state is not None:
        np.random.seed(random_state)
    
    np.random.shuffle(data)
    
    n_total = len(data)
    
    # split data
    n_train = int(train_size * n_total)
    n_val = int(val_size * n_total)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    return train_data, val_data, test_data
    
    

class windowDataset(Dataset):
    def __init__(self, data, input_window, output_window, stride):
        
        #총 데이터의 개수
        n_sub = data[0].shape
        
        # 한 subject당 가지는 timeseries의 길이
        ts_length = data[2].shape
        roi = data[1].shape
        
        #stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (ts_length - input_window - output_window) // stride + 1

        #input과 output
        
        
        total_X = []
        total_y = []
        
        for num in range(n_sub):
            X = np.zeros([roi,input_window, num_samples])
            Y = np.zeros([roi,output_window, num_samples])

            for i in range(num_samples):
                start_x = stride*i
                end_x = start_x + input_window
                X[:,:,i] = data[num][:,start_x:end_x]

                start_y = stride*i + input_window
                end_y = start_y + output_window
                Y[:,:,i] = data[num][:,start_y:end_y]
                
            total_X.append(X)
            total_y.append(Y)
            
        # 리스트의 배열들을 첫 번째 축(axis=0)으로 연결
        total_X_concat = np.concatenate(total_X, axis=2)  # 세 번째 축(num_samples 축)을 기준으로 결합
        total_y_concat = np.concatenate(total_y, axis=2)

        print("Concatenated total_X shape:", total_X_concat.shape)
        print("Concatenated total_y shape:", total_y_concat.shape)

        self.x = total_X_concat
        self.y = total_X_concat
        
        self.len = len(X)
    def __getitem__(self, i):
        return self.x[i], self.y[i, :-1], self.y[i,1:]
    def __len__(self):
        return self.len