# coding: UTF-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class AttLSTM(nn.Module):
    def __init__(self, args):
        super(AttLSTM, self).__init__()
        
        self.lstm_1 = nn.LSTM(args.in_dim, args.h_dim, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.lstm_2 = nn.LSTM(args.in_dim, args.h_dim, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.w1 = nn.Parameter(torch.Tensor(args.h_dim * 2)).cuda()
        self.w2 = nn.Parameter(torch.Tensor(args.h_dim * 2)).cuda()
        
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(args.h_dim * 4, args.num_class)

    def forward(self, x1, x2):
         
        H_x1, _ = self.lstm_1(x1)  # [batch_size, seq_len, hidden_size * num_direction]=[32, *, 600]
        H_x2, _ = self.lstm_2(x2) 
        M_x1 = self.tanh1(H_x1)  # [32, *, 600]
        M_x2 = self.tanh2(H_x2)
        
        alpha_x1 = F.softmax(torch.matmul(M_x1, self.w1), dim=1).unsqueeze(-1)  # [32, *, 1]
        out_x1 = H_x1 * alpha_x1  # [128, 32, 256]
        out_x1 = torch.sum(out_x1, 1)  # [128, 256]

        alpha_x2 = F.softmax(torch.matmul(M_x2, self.w2), dim=1).unsqueeze(-1)  # [32, *, 1]
        out_x2 = H_x2 * alpha_x2  # [128, 32, 256]
        out_x2 = torch.sum(out_x2, 1)  # [128, 256]

        out = torch.cat((out_x1, out_x2), 1)
        out = self.tanh(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        # out_x1 = self.tanh2(out_x1)
        # out_x1 = self.fc(out_x1)
        # out_x1 = F.log_softmax(out_x1, dim=1)
                
        # out_x2 = self.tanh2(out_x2)
        # out_x2 = self.fc(out_x2)
        # out_x2 = F.log_softmax(out_x2, dim=1)
        
        return out