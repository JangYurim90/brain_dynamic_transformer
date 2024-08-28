import numpy as np
import pandas as pd
from operator import itemgetter
import pandas as pd
from matplotlib import pyplot, patches,axes
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from glob import glob
import seaborn as sns
import scipy
from torch.utils.data import DataLoader, Dataset
from argument import args_parser
from dataloader import data_split, windowDataset
import torch
from torch import nn
from model import TFModel,PositionalEncoding
from tqdm import tqdm

def evaluate(length):
    input = torch.tensor(data_train[-24*7*2:]).reshape(1,-1,1).to(device).float().to(device)
    output = torch.tensor(data_train[-1].reshape(1,-1,1)).float().to(device)
    model.eval()
    for i in range(length):
        src_mask = model.generate_square_subsequent_mask(input.shape[1]).to(device)
        tgt_mask = model.generate_square_subsequent_mask(output.shape[1]).to(device)

        predictions = model(input, output, src_mask, tgt_mask).transpose(0,1)
        predictions = predictions[:, -1:, :]
        output = torch.cat([output, predictions.to(device)], axis=1)
    return torch.squeeze(output, axis=0).detach().cpu().numpy()[1:]

args = args_parser()
device = torch.device("cuda:0")

load_data = np.load(args.HCPdata_dir)
# data만 따로 추출 하는 코드

# data split
train_data, val_data, test_data = data_split(load_data,args.train_size,args.val_size,args.test_size,random_state = 3)

# data_loader 
train_dataset = windowDataset(train_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride)
val_dataset = windowDataset(val_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride)
test_dataset = windowDataset(test_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride)

train_loader = DataLoader(train_dataset,batch_size=args.batch_size)
val_loader = DataLoader(val_dataset,batch_size=args.batch_size)
test_loader = DataLoader(test_dataset,batch_size=args.batch_size)

# Load model (transformer)
model = TFModel(args.roi,args.d_model, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# train
model.train()
progress = tqdm(range(epoch))
for i in progress:
    batchloss = 0.0
    
    for (inputs, dec_inputs, outputs) in train_loader:
        optimizer.zero_grad()
        src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
        tgt_mask = model.generate_square_subsequent_mask(dec_inputs.shape[1]).to(device)

        result = model(inputs.float().to(device), dec_inputs.float().to(device), src_mask, tgt_mask)
        loss = criterion(result.permute(1,0,2), outputs.float().to(device))
        
        loss.backward()
        optimizer.step()
        batchloss += loss
    progress.set_description("{:0.5f}".format(batchloss.cpu().item() / len(train_loader)))
    
    
# Validation 


result = evaluate(24*7)
result = min_max_scaler.inverse_transform(result)
real = rawdata["평균속도"].to_numpy()
real = min_max_scaler.inverse_transform(real.reshape(-1,1))