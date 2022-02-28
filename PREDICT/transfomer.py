#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl  # 用于设置曲线参数



# In[2]:




# In[163]:


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import os

# In[320]:


# 读取数据切割数据集并保存
TRAIN_WEIGHT = 0.9
SEQ_LEN = 99
LEARNING_RATE = 0.00001
BATCH_SIZE = 4
EPOCH = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# train_size=int(TRAIN_WEIGHT*(data.shape[0]))
train_path = "tr.csv"
test_path = "te.csv"
# Train_data=data[:train_size+SEQ_LEN]
# Test_data=data[train_size-SEQ_LEN:]
# Train_data.to_csv(train_path,sep=',',index=False,header=False)
# Test_data.to_csv(test_path,sep=',',index=False,header=False)


# In[321]:


mean_list = []
std_list = []


# In[358]:


# 完成数据集类
class Stock_Data(Dataset):
    def __init__(self, train=True, transform=None):
        if train == True:
            train_path = "tr.csv"
            with open(train_path) as f:
                self.data = np.loadtxt(f, delimiter=",")
                # 可以注释
                # addi=np.zeros((self.data.shape[0],1))
                # self.data=np.concatenate((self.data,addi),axis=1)
                self.data = self.data[:, 0:8]
            for i in range(len(self.data[0])):
                mean_list.append(np.mean(self.data[:, i]))
                std_list.append(np.std(self.data[:, i]))
                self.data[:, i] = (self.data[:, i] - np.mean(self.data[:, i])) / (np.std(self.data[:, i]) + 1e-8)
            self.value = torch.rand(self.data.shape[0] - SEQ_LEN, SEQ_LEN, self.data.shape[1])
            self.label = torch.rand(self.data.shape[0] - SEQ_LEN, 1)
            for i in range(self.data.shape[0] - SEQ_LEN):
                self.value[i, :, :] = torch.from_numpy(self.data[i:i + SEQ_LEN, :].reshape(SEQ_LEN, self.data.shape[1]))
                self.label[i, :] = self.data[i + SEQ_LEN, 0]
            self.data = self.value
            print(self.value.shape,self.label.shape)
        else:
            test_path = "te.csv"
            with open(test_path) as f:
                self.data = np.loadtxt(f, delimiter=",")
                # addi=np.zeros((self.data.shape[0],1))
                # self.data=np.concatenate((self.data,addi),axis=1)
                self.data = self.data[:, 0:8]
            for i in range(len(self.data[0])):
                self.data[:, i] = (self.data[:, i] - mean_list[i]) / (std_list[i] + 1e-8)
            self.value = torch.rand(self.data.shape[0] - SEQ_LEN, SEQ_LEN, self.data.shape[1])
            self.label = torch.rand(self.data.shape[0] - SEQ_LEN, 1)
            for i in range(self.data.shape[0] - SEQ_LEN):
                self.value[i, :, :] = torch.from_numpy(self.data[i:i + SEQ_LEN, :].reshape(SEQ_LEN, self.data.shape[1]))
                self.label[i, :] = self.data[i + SEQ_LEN, 0]
            self.data = self.value

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data[:, 0])


# In[388]:


stock_train = Stock_Data(train=True)
stock_test = Stock_Data(train=False)


# LSTM模型
class LSTM(nn.Module):
    def __init__(self, dimension):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=dimension, hidden_size=128, num_layers=3, batch_first=True)
        self.linear1 = nn.Linear(in_features=128, out_features=16)
        self.linear2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        x = self.linear1(x)
        x = self.linear2(x)
        return x


# In[391]:


# 传入tensor进行位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=SEQ_LEN):
        super(PositionalEncoding, self).__init__()
        # 序列长度，dimension d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# In[392]:


class TransAm(nn.Module):
    def __init__(self, feature_size=8, num_layers=6, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # 全连接层代替decoder
        self.decoder = nn.Linear(feature_size, 1)
        self.linear1 = nn.Linear(SEQ_LEN, 1)
        self.init_weights()
        self.src_key_padding_mask = None

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, seq_len=SEQ_LEN):
        src = self.pos_encoder(src)
        # print(src)
        # print(self.src_mask)
        # print(self.src_key_padding_mask)
        # output=self.transformer_encoder(src,self.src_mask,self.src_key_padding_mask)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = np.squeeze(output)
        output = self.linear1(output)
        return output


# In[394]:




def train(epoch):
    model.train()
    global loss_list
    global iteration
    dataloader = DataLoader(dataset=stock_train, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    for i, (data, label) in enumerate(dataloader):
        iteration = iteration + 1
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            loss_list.append(loss.item())
            print("epoch=", epoch, "iteration=", iteration, "loss=", loss.item())



# In[395]:


def test():
    model.eval()
    global accuracy_list
    global predict_list
    dataloader = DataLoader(dataset=stock_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    for i, (data, label) in enumerate(dataloader):
        with torch.no_grad():
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            predict = model.forward(data)
            predict_list.append(predict)
            loss = criterion(predict, label)
            accuracy_fn = nn.MSELoss()
            accuracy = accuracy_fn(predict, label)
            accuracy_list.append(accuracy.item())
    print("test_data MSELoss:(pred-real)/real=", np.mean(accuracy_list))


# In[396]:


def loss_curve(loss_list):
    x = np.linspace(1, len(loss_list), len(loss_list))
    x = 20 * x
    plt.plot(x, np.array(loss_list), label="train_loss")
    plt.ylabel("MSELoss")
    plt.xlabel("iteration")
    plt.savefig("train_loss.png", dpi=3000)
    plt.show()


# In[397]:


def contrast_lines(predict_list):
    real_list = []
    prediction_list = []
    dataloader = DataLoader(dataset=stock_test, batch_size=4, shuffle=False, drop_last=True)
    for i, (data, label) in enumerate(dataloader):
        for idx in range(BATCH_SIZE):
            real_list.append(np.array(label[idx] * std_list[0] + mean_list[0]))
    for item in predict_list:
        item = item.to("cpu")
        for idx in range(BATCH_SIZE):
            prediction_list.append(np.array((item[idx] * std_list[0] + mean_list[0])))
    x = np.linspace(1, len(real_list), len(real_list))
    plt.plot(x, np.array(real_list), label="real")
    plt.plot(x, np.array(prediction_list), label="prediction")
    plt.legend()
    plt.savefig("000001SZ_Pre.png", dpi=3000)
    plt.show()


# 选择模型为LSTM或Transformer，注释掉一个


model=TransAm(feature_size=8)
# save_path=transformer_path

model = model.to(device)
criterion = nn.MSELoss()


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


if __name__ == "__main__":
    '''    
    symbol = '000001.SZ'
    period = 100
    data = import_csv(symbol)
    df_draw = data[-period:]
    draw_Kline(df_draw, period, symbol)
    data.drop(['ts_code', 'Date'], axis=1, inplace=True)'''
    iteration = 0
    loss_list = []
    # 开始训练神经网络
    for epoch in range(1, EPOCH + 1):
        predict_list = []
        accuracy_list = []
        train(epoch)
        test()
    # 绘制损失函数下降曲线
    loss_curve(loss_list)
    # 绘制测试集pred-real对比曲线
    contrast_lines(predict_list)









