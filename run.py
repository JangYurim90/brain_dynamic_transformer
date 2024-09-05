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
import torch.utils
from torch.utils.data import DataLoader, Dataset
from argument import args_parser
from dataloader import data_split, windowDataset, hcp_data_load
import torch
from torch import nn
from transformer import TFModel,PositionalEncoding
from models.model import Transformer
from tqdm import tqdm


args = args_parser()
device = torch.device("cuda:0")

#load_data = np.load(args.HCPdata_dir)

# data만 따로 추출 하는 코드
load_data = hcp_data_load(args.atlas,args.HCPdata_dir)

# data split
train_data, val_data, test_data = data_split(load_data,args.train_size,args.val_size,args.test_size,random_state = 3)

# data_loader 
train_dataset = windowDataset(train_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride, batch_size = args.batch_size)
val_dataset = windowDataset(val_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride,batch_size = args.batch_size)
test_dataset = windowDataset(test_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride , batch_size = args.batch_size)


train_loader = DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True)
val_loader = DataLoader(val_dataset,batch_size=args.batch_size,drop_last=True)
test_loader = DataLoader(test_dataset,batch_size=args.batch_size,drop_last=True)

# Load model (transformer)
model = TFModel(d_model=args.d_model, nhead=args.nhead, nhid=args.nhid, nlayers=args.nlayers,input_length=args.i_win, output_length=args.o_win, dropout=args.dropout).to(device)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# train
model.train()
progress = tqdm(range(args.epoch))
for i in progress:
    batchloss = 0.0
    print("Epoch: ", i)
    for (inputs, outputs) in train_loader:
        optimizer.zero_grad()
        
        # Inputs to encoder (src), and targets for decoder (tgt)
        inputs = inputs.transpose(0, 1)

        src = inputs.float().to(device)  # Input timepoints (e.g., 100 timepoints)
        
        outputs = outputs.transpose(0, 1)

        #tgt = outputs[:, :-1, :].float().to(device)  # Previous timepoints for decoder (teacher forcing)
        #target_output = outputs[:, 1:, :].float().to(device)  # Next 10 timepoints
        tgt = outputs.float().to(device)
        

        src_mask = model.generate_square_subsequent_mask_3(src.shape[0]).to(device)
        tgt_mask = model.generate_square_subsequent_mask_3(tgt.shape[0]).to(device)
        
        tgt_len = tgt.shape[0]
        src_len = src.shape[0]
        
        
        
        #tgt_mask = model.generate_square_subsequent_mask(tgt_len, src_len).to(device)
        #src_mask = model.generate_square_subsequent_mask(tgt_len, src_len).to(device)    

        # Get the output and Q, K, V from both encoder and decoder
        result, enc_qkv, dec_qkv = model(src, tgt, src_mask, tgt_mask)

        loss = criterion(result.permute(1,0,2), outputs.float().to(device))
        
        loss.backward()
        optimizer.step()
        batchloss += loss
    
    if len(train_loader) > 0:
        progress.set_description("{:0.5f}".format(batchloss / len(train_loader)))
    else:
        progress.set_description("No data in train_loader")
        

# Validation loop
model.eval()
total_val_loss = 0.0

with torch.no_grad():
    for (inputs, outputs) in val_loader:
        inputs = inputs.transpose(0, 1)
        src = inputs.float().to(device)
        
        outputs = outputs.transpose(0, 1)
        tgt = outputs[:, :-1, :].float().to(device)
        target_output = outputs[:, 1:, :].float().to(device)

        src_mask = model.generate_square_subsequent_mask(src.shape[0]).to(device)
        tgt_mask = model.generate_square_subsequent_mask(tgt.shape[0]).to(device)

        # Get the output and Q, K, V from both encoder and decoder
        result, enc_qkv, dec_qkv = model(src, tgt, src_mask, tgt_mask)
        
        val_loss = criterion(result, target_output)
        total_val_loss += val_loss.item()

average_val_loss = total_val_loss / len(val_loader)
print(f"Validation Loss: {average_val_loss}")