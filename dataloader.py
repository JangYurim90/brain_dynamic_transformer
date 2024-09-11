import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


def hcp_data_load(atlas, data_dir):
    dataset = np.load(data_dir + '/HCP_'+atlas+'_data_sub100.npy', allow_pickle=True)
    
    data_x = []
    
    for data in dataset[:50]:
        data_x.append(data['roiTimeseries'])
    
    data_x = np.array(data_x)
    
    print("Data shape:", data_x.shape)
    return data_x
    
    
    
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
    
    print("Train data shape:", train_data.shape)
    print("Validation data shape:", val_data.shape)
    print("Test data shape:", test_data.shape)
    
    return train_data, val_data, test_data
    
    

class windowDataset(Dataset):
    def __init__(self, data, input_window, output_window, stride,batch_size):
        
        #총 데이터의 개수
        n_sub = data.shape[0]
        
        # 한 subject당 가지는 timeseries의 길이
        ts_length = data.shape[1]
        roi = data.shape[2]
        
        #stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (ts_length - input_window - output_window) // stride + 1
        print("Number of samples:", num_samples)

        #input과 output
        
        
        total_X = []
        total_y = []
        
        for num in range(n_sub):
            X = np.zeros([num_samples,input_window,roi])
            Y = np.zeros([num_samples,output_window,roi])

            for i in range(num_samples):
                start_x = stride*i
                end_x = start_x + input_window
                X[i,:,:] = data[num][start_x:end_x,:]

                start_y = stride*i + input_window
                end_y = start_y + output_window
                Y[i,:,:] = data[num][start_y:end_y,:]
                
            total_X.append(X)
            total_y.append(Y)
        
        print("Total_X shape:", np.array(total_X).shape)
            
        # 리스트의 배열들을 첫 번째 축(axis=0)으로 연결
        total_X_concat = np.concatenate(total_X, axis=0)  # 세 번째 축(num_samples 축)을 기준으로 결합
        total_y_concat = np.concatenate(total_y, axis=0)
        
        #total_X_concat = total_X_concat.transpose(0,2,1)
        #total_y_concat = total_y_concat.transpose(0,2,1)

        print("Concatenated total_X shape:", total_X_concat.shape)
        print("Concatenated total_y shape:", total_y_concat.shape)
        
        adjusted_total_X = total_X_concat[:len(total_X_concat) // 32 * 32]
        adjusted_total_Y = total_y_concat[:len(total_y_concat) // 32 * 32]

        self.x = adjusted_total_X
        self.y = adjusted_total_Y
        
        self.len = len(self.x)
        
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    

    def __len__(self):
        return self.len