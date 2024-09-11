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
from models_encoder.layers import TSTransformerEncoder
from tqdm import tqdm
from datetime import datetime


# 경로가 없으면 생성하는 함수
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# 현재 시간을 "년-월-일_시-분-초" 형식으로 반환하는 함수
def get_time_string():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def generate_padding_mask(batch_size, seq_length):
    # (batch_size, seq_length) 크기의 패딩 마스크를 생성
    # 패딩 위치는 0, 유효한 데이터 위치는 1
    mask = torch.ones(batch_size, seq_length)  # 기본적으로 모든 위치를 1로 설정
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_padding_mask_bool(batch_size, seq_length):
    # 기본적으로 모든 위치를 1로 설정 (유효한 위치)
    mask = torch.ones(batch_size,seq_length, dtype=torch.bool)  # Boolean 타입으로 설정
    return mask


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def train(model, train_loader,i):
    batchloss = 0.0
    print("Epoch: ", i)
    for (inputs, outputs) in train_loader:
        optimizer.zero_grad()
        
        # Inputs to encoder (src), and targets for decoder (tgt)
        inputs = inputs.transpose(0, 1)

        src = inputs.float().to(device)  # Input timepoints (e.g., 100 timepoints)
        
        #outputs = outputs.transpose(0, 1)

        #tgt = outputs[:, :-1, :].float().to(device)  # Previous timepoints for decoder (teacher forcing)
        #target_output = outputs[:, 1:, :].float().to(device)  # Next 10 timepoints
        tgt = outputs.float().to(device)
        

        src_mask = generate_padding_mask_bool(batch_size=args.batch_size,seq_length=src.shape[0]).to(device)
        tgt_mask = generate_padding_mask(batch_size=args.batch_size,seq_length=tgt.shape[0]).to(device)
     

        # Get the output and Q, K, V from both encoder and decoder
        result, all_q, all_k,all_v = model(src, src_mask)
        
        # 수정 후: 크기가 맞는지 확인하고 loss 계산
        result = result.to(device)  # result를 device로 이동
        outputs = outputs.to(device)  # outputs을 device로 이동
        

        # 배치 크기와 시퀀스 길이가 일치하는지 확인
        if result.shape == outputs.shape:
            loss = criterion(result, outputs.float())
        else:
            print("Shape mismatch:", result.shape, outputs.shape)

        loss = criterion(result, outputs.float().to(device))

        #batch_loss = torch.sum(loss)
        mean_loss = torch.mean(loss)  # mean loss (over active elements) used for optimization
        
        
        total_loss = mean_loss
        
        total_loss.backward()
        optimizer.step()
        batchloss += loss
    
    if len(train_loader) > 0:
        progress.set_description("{:0.5f}".format(batchloss / len(train_loader)))
        train_losses.append(batchloss / len(train_loader))
    else:
        progress.set_description("No data in train_loader")
        
    return train_losses

def validation(model, val_loader):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for (inputs, outputs) in val_loader:
            inputs = inputs.transpose(0, 1)
            src = inputs.float().to(device)

            #outputs = outputs.transpose(0, 1)
            tgt = outputs.float().to(device)

            src_mask = generate_padding_mask_bool(batch_size=args.batch_size,seq_length=src.shape[0]).to(device)
            tgt_mask = generate_padding_mask(batch_size=args.batch_size,seq_length=tgt.shape[0]).to(device)

            # Get the output and Q, K, V from both encoder and decoder
            result, all_q, all_k,all_v = model(src, src_mask)

            outputs = outputs.to(device)

            val_loss = criterion(result, outputs.float().to(device))
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(val_loader)
    # 초록색으로 텍스트 출력
    print(f"Validation Loss: {average_val_loss}")
    
    return average_val_loss
    
def test(model, test_loader,model_path):
    print("Testing...\n\n")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    total_test_loss = 0.0
    loss_per_subject = []
    plot = {'result': [], 'output': []}
    
    with torch.no_grad():
        num_samples = 0
        num_samples_loss = 0.0
        for (inputs, outputs) in test_loader:
            inputs = inputs.transpose(0, 1)
            src = inputs.float().to(device)

            #outputs = outputs.transpose(0, 1)
            tgt = outputs.float().to(device)

            src_mask = generate_padding_mask_bool(batch_size=args.batch_size,seq_length=src.shape[0]).to(device)
            tgt_mask = generate_padding_mask(batch_size=args.batch_size,seq_length=tgt.shape[0]).to(device)

            # Get the output and Q, K, V from both encoder and decoder
            result, all_q, all_k,all_v = model(src, src_mask)

            outputs = outputs.to(device)

            test_loss = criterion(result, outputs.float().to(device))
            total_test_loss += test_loss.item()
            
            plot['result'].append(result.cpu().numpy())
            plot['output'].append(outputs.cpu().numpy())    
            
            num_samples += inputs.shape[1]
            num_samples_loss += test_loss.item()
            if num_samples % 470 == 0:
                print(f"Test Loss: {num_samples_loss / 470}")
                loss_per_subject.append(num_samples_loss / 470)
                num_samples_loss = 0.0
                print("{}th_subject  done".format(num_samples/470))
                
        print(f"Test Loss: {total_test_loss / len(test_loader)}")
        average_test_loss = total_test_loss / len(test_loader)
        
        return average_test_loss, loss_per_subject, plot
    


   
args = args_parser()
device = torch.device("cuda:0")

# Save the model with the lowest validation loss
save_dir = "{}/A{}_i{}_o{}_lr{}_sub{}".format(args.loss_dir, args.atlas, args.i_win, args.o_win, args.lr,100)
create_directory_if_not_exists(save_dir)


time_str = get_time_string()  

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
#model = TFModel(d_model=args.d_model, nhead=args.nhead, nhid=args.nhid, nlayers=args.nlayers,input_length=args.i_win, output_length=args.o_win, dropout=args.dropout).to(device)
model = TSTransformerEncoder(feat_dim=args.feature_dim, max_len =1024,d_model=args.d_model,n_heads=args.nhead,num_layers=args.nlayers,dim_feedforward=args.feature_dim ,output_length=args.o_win,dropout=args.dropout).to(device)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Train 및 Validation loss 저장할 리스트 초기화
train_losses = []
val_losses = []

# train
if args.mode == 'train':
    model.train()
    progress = tqdm(range(args.epoch))
    for i in progress:
        train_losses = train(model, train_loader, i)
            
        # Validation loop
        val_loss = validation(model, val_loader)
        val_losses.append(val_loss)
        
    # 텐서를 numpy 배열로 변환할 때, CUDA 텐서를 CPU로 이동 후 변환
    train_losses_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    val_losses_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
    torch.save(model.state_dict(), f'{save_dir}/model_{time_str}.pth')
    
    # save results
    np.save(os.path.join(save_dir, f'train_losses_{time_str}.npy'), np.array(train_losses_cpu))
    np.save(os.path.join(save_dir, f'val_losses_{time_str}.npy'), np.array(val_losses_cpu))
        

if args.mode == 'test':
    # Test loop
    model_path = f'{save_dir}/model_2024-09-10_15-45-42.pth'
    test_loss, loss_per_subject,plot = test(model, test_loader, model_path)
    test_loss_cpu = test_loss.cpu().item() if isinstance(test_loss, torch.Tensor) else test_loss
    loss_per_subject_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in loss_per_subject]

    # Save test results
    np.save(os.path.join(save_dir, f'test_loss_{time_str}.npy'), np.array(test_loss_cpu))
    np.save(os.path.join(save_dir, f'loss_per_subject_{time_str}.npy'), np.array(loss_per_subject_cpu))
    np.save(os.path.join(save_dir, f'plot_{time_str}.npy'), plot)








        
'''
# Validation loop
model.eval()
total_val_loss = 0.0

with torch.no_grad():
    for (inputs, outputs) in val_loader:
        inputs = inputs.transpose(0, 1)
        src = inputs.float().to(device)
        
        outputs = outputs.transpose(0, 1)
        #tgt = outputs[:, :-1, :].float().to(device)
        tgt = outputs.float().to(device)
        #target_output = outputs[:, 1:, :].float().to(device)

        src_mask = model.generate_square_subsequent_mask_3(src.shape[0]).to(device)
        tgt_mask = model.generate_square_subsequent_mask_3(tgt.shape[0]).to(device)

        # Get the output and Q, K, V from both encoder and decoder
        result, enc_qkv, dec_qkv = model(src, tgt, src_mask, tgt_mask)
        
        outputs = outputs.to(device)
        
        val_loss = criterion(result, outputs.float().to(device))
        total_val_loss += val_loss.item()

average_val_loss = total_val_loss / len(val_loader)
print(f"Validation Loss: {average_val_loss}")
'''

