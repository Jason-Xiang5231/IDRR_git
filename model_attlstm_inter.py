# coding: UTF-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class AttLSTM(nn.Module):
    def __init__(self, args):
        super(AttLSTM, self).__init__()
        # level - 1
        self.lstm_Inter_1 = nn.LSTM(args.in_dim, args.h_dim, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.lstm_Inter_2 = nn.LSTM(args.in_dim, args.h_dim, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        # self.tanh_Inter_1 = nn.Tanh()
        # self.tanh_Inter_2 = nn.Tanh()
        # self.w_Inter_1 = nn.Parameter(torch.Tensor(args.h_dim * 2)).cuda()
        # self.w_Inter_2 = nn.Parameter(torch.Tensor(args.h_dim * 2)).cuda()
        # level - 2
        self.lstm_1 = nn.LSTM(args.in_dim * 4, args.h_dim * 4, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.lstm_2 = nn.LSTM(args.in_dim * 4, args.h_dim * 4, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.tanh1 = nn.Tanh()
        self.w1 = nn.Parameter(torch.Tensor(args.h_dim * 8)).cuda()
        self.tanh2 = nn.Tanh()
        self.w2 = nn.Parameter(torch.Tensor(args.h_dim * 8)).cuda()
        # output - classification
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(args.h_dim * 8, args.num_class)

    def forward(self, x1, x2):
        
        H_Inter_x1, _ = self.lstm_Inter_1(x1)  # [batch_size, arg_1_length, in_dim * num_direction] = [32, *, 600]
        H_Inter_x2, _ = self.lstm_Inter_2(x2)  # [32, *, 600]
        out_Inter_x1 = H_Inter_x1[:, -1, :]
        out_Inter_x2 = H_Inter_x2[:, -1, :]
        # M_Inter_x1 = self.tanh_Inter_1(H_Inter_x1)  # [32, *, 600]
        # M_Inter_x2 = self.tanh_Inter_2(H_Inter_x2) # [32, *, 600]
        
        # alpha_Inter_x1 = F.softmax(torch.matmul(M_Inter_x1, self.w_Inter_1), dim=1).unsqueeze(-1)  # [32, *, 1]
        # out_Inter_x1 = H_Inter_x1 * alpha_Inter_x1  # [32, *, 600]
        # out_Inter_x1 = torch.sum(out_Inter_x1, 1) # [32, 600]

        # alpha_Inter_x2 = F.softmax(torch.matmul(M_Inter_x2, self.w_Inter_2), dim=1).unsqueeze(-1)  # [32, *, 1]
        # out_Inter_x2 = H_Inter_x2 * alpha_Inter_x2  # [32, *, 600]
        # out_Inter_x2 = torch.sum(out_Inter_x2, 1)  # [32, 600]
        
        # x1的每个词向量和x2的attlstm输出向量拼接 [32, *, 1200]
        x1_inter = torch.cat((H_Inter_x1, out_Inter_x2.expand(H_Inter_x1.shape[1], out_Inter_x2.shape[0], out_Inter_x2.shape[1]).permute(1, 0, 2)), 2)
        # x1的每个词向量和x2的attlstm输出向量拼接 [32, *, 1200]
        x2_inter = torch.cat((H_Inter_x2, out_Inter_x1.expand(H_Inter_x2.shape[1], out_Inter_x1.shape[0], out_Inter_x1.shape[1]).permute(1, 0, 2)), 2)
        
        H_x1, _ = self.lstm_1(x1_inter)  # [32, *, 2400]
        M_x1 = self.tanh1(H_x1)  # [32, *, 2400]
        H_x2, _ = self.lstm_2(x2_inter)  # [32, *, 2400]
        M_x2 = self.tanh2(H_x2) #[32, *, 2400]
        
        alpha_x1 = F.softmax(torch.matmul(M_x1, self.w1), dim=1).unsqueeze(-1)   # [32, *, 1]
        out_x1 = H_x1 * alpha_x1  # [32, *, 2400]
        out_x1 = torch.sum(out_x1, 1)   # [32, *, 2400]

        alpha_x2 = F.softmax(torch.matmul(M_x2, self.w2), dim=1).unsqueeze(-1)  # [32, *, 1]
        out_x2 = H_x2 * alpha_x2  # [32, *, 2400]
        out_x2 = torch.sum(out_x2, 1)  # [32, *, 2400]
        
        # out_x1, _ = out_x1.chunk(2, dim=1)
        # out_x2, _ = out_x2.chunk(2, dim=1)

        out = torch.cat((out_x1, out_x2), 1) #[32, 4800]
        out = self.tanh(out) #[32, 4800]
        out = self.fc(out) #[32, 4]
        out = F.log_softmax(out, dim=1) #[32, 4]
        
        return out