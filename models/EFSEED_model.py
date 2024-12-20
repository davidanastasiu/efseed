#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import random

random.seed("a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderLSTM(nn.Module):
    def __init__(self, opt):
        super(EncoderLSTM, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.layer_dim = opt.layer
        self.cnn_dim = opt.cnn_dim
        self.predict_days = opt.output_len
        cnn_d2 = [4, 3, 2, 2]
        self.dim_in = 2 + opt.r_shift
        fc_dim = 400

        if(opt.watershed == 0):
            self.dim_in = 1
            
        self.fc_embed = nn.Linear(self.dim_in, fc_dim)
    
        self.cnn0 = nn.Sequential(  
            nn.Conv1d(fc_dim, self.cnn_dim, cnn_d2[0], stride=cnn_d2[0], padding=0),            
        )        
        self.cnn1 = nn.Sequential(  
            nn.Conv1d(fc_dim, self.cnn_dim, cnn_d2[1], stride=cnn_d2[1], padding=0),
        ) 
            
        self.cnn3 = nn.Sequential(     
            nn.Conv1d(fc_dim, self.cnn_dim, cnn_d2[2], stride=cnn_d2[2], padding=0),
        ) 
   
        self.cnn4 = nn.Sequential(       
            nn.Conv1d(fc_dim, self.cnn_dim, cnn_d2[3], stride=cnn_d2[3], padding=0),
        ) 

        self.lstm0 = nn.LSTM(
            self.cnn_dim,
            self.hidden_dim,
            self.layer_dim,
            batch_first=True,
        )
        self.lstm1 = nn.LSTM(
            self.cnn_dim,
            self.hidden_dim,
            self.layer_dim,
            batch_first=True,
        )
        self.lstm3 = nn.LSTM(
            self.cnn_dim,
            self.hidden_dim,
            self.layer_dim * 2,
            batch_first=True,
        )
        self.lstm4 = nn.LSTM(
            self.cnn_dim,
            self.hidden_dim,
            self.layer_dim * 2,
            batch_first=True,
        )
        self.lstm5 = nn.LSTM(
            2,
            self.hidden_dim,
            self.layer_dim * 2,
            batch_first=True,
        )

    def forward(self, x, h, c):
        # Initialize hidden and cell state with zeros
        h0 = h
        c0 = c

        T = nn.Tanh()

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)    
        h1 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).to(device)
        c1 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).to(device)    

        # FC embedding   
        x1 = T(self.fc_embed(x[:,:,:self.dim_in]))
        x2 = x[:,-480:,0:2] 
    
        cnn_out0 = self.cnn0(x1.permute(0,2,1))
        cnn_out0 = cnn_out0.permute(0,2,1)  
        
        cnn_out1 = self.cnn1(x1.permute(0,2,1))
        cnn_out1 = cnn_out1.permute(0,2,1)  
        
        cnn_out3 = self.cnn3(x1.permute(0,2,1))
        cnn_out3 = cnn_out3.permute(0,2,1) 
        
        cnn_out4 = self.cnn4(x1.permute(0,2,1))
        cnn_out4 = cnn_out4.permute(0,2,1) 
       
        hn = []
        cn = []
        out, (hn0, cn0) = self.lstm0(cnn_out0, (h0,c0))
        out, (hn1, cn1) = self.lstm1(cnn_out1, (h0,c0))
        out, (hn3, cn3) = self.lstm3(cnn_out3, (h1,c1))
        out, (hn4, cn4) = self.lstm4(cnn_out4, (h1,c1))
        out, (hn5, cn5) = self.lstm5(x2, (h1,c1))

        hn.append(hn0)
        hn.append(hn1)
        hn.append(hn1)
        hn.append(hn3)
        hn.append(hn4)
        hn.append(hn5)

        cn.append(cn0)
        cn.append(cn1)
        cn.append(cn1)
        cn.append(cn3)
        cn.append(cn4)
        cn.append(cn5)
        
        return hn, cn


class DecoderLSTM(nn.Module):
    def __init__(self, opt):
        super(DecoderLSTM, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.layer_dim = opt.layer
        self.cnn_dim = opt.cnn_dim
        self.predict_days = opt.output_len
        self.opt = opt

        self.lstm00 = nn.LSTM(2, self.hidden_dim, self.layer_dim, batch_first=True) #8
        self.lstm01 = nn.LSTM(self.hidden_dim, self.hidden_dim, self.layer_dim,  batch_first=True) #32
        self.lstm03 = nn.LSTM(self.hidden_dim, self.hidden_dim, self.layer_dim*2, batch_first=True) #96
        self.lstm04 = nn.LSTM(self.hidden_dim, self.hidden_dim, self.layer_dim*2, batch_first=True) #288
        self.lstm05 = nn.LSTM(2, self.hidden_dim, self.layer_dim*2,  batch_first=True) #288

        self.cnn00 = nn.Sequential(            
        nn.Conv1d(self.hidden_dim, self.cnn_dim, 3, stride=1, padding=1),
        nn.Tanh(),
        nn.Conv1d(self.cnn_dim, 1, 5, stride=1, padding=2),
        ) 
        self.cnn01 = nn.Sequential(            
        nn.Conv1d(self.hidden_dim, self.cnn_dim, 3, stride=1, padding=1),
        nn.Tanh(),
        nn.Conv1d(self.cnn_dim, 1, 5, stride=1, padding=2),
        )
        self.cnn03 = nn.Sequential(            
        nn.Conv1d(self.hidden_dim, self.cnn_dim, 3, stride=1, padding=1),
        nn.Tanh(),
        nn.Conv1d(self.cnn_dim, 1, 5, stride=1, padding=2),
        )
        self.cnn04 = nn.Sequential(            
        nn.Conv1d(self.hidden_dim, self.cnn_dim, 3, stride=1, padding=1),
        nn.Tanh(),
        nn.Conv1d(self.cnn_dim, 1, 5, stride=1, padding=2),
        ) 
        self.cnn02 = nn.Sequential(            
        nn.Conv1d(self.hidden_dim, self.cnn_dim, 3, stride=1, padding=1),
        nn.Tanh(),
        nn.Conv1d(self.cnn_dim, 1, 5, stride=1, padding=2),         
        )        


    def forward(self, x1, x3, encoder_h, encoder_c):  # x1: time sin & cos; x3: input sequence
        # Initialize hidden and cell state with zeros
        h0 = encoder_h
        c0 = encoder_c

        uu = int(self.predict_days / 8)  # 32,
        bb = torch.cat([x1[:, 0:1, :], x1[:, uu: uu + 1, :]], dim=1)
        bb = torch.cat([bb, x1[:, 2 * uu: 2 * uu + 1, :]], dim=1)
        bb = torch.cat([bb, x1[:, 3 * uu: 3 * uu + 1, :]], dim=1)
        bb = torch.cat([bb, x1[:, 4 * uu: 4 * uu + 1, :]], dim=1)
        bb = torch.cat([bb, x1[:, 5 * uu: 5 * uu + 1, :]], dim=1)
        bb = torch.cat([bb, x1[:, 6 * uu: 6 * uu + 1, :]], dim=1)
        bb = torch.cat([bb, x1[:, 7 * uu: 7 * uu + 1, :]], dim=1)
        bb = torch.cat([bb, x1[:, 8 * uu: 8 * uu + 1, :]], dim=1)

        s_l = int(self.predict_days / 36)  #8
        # segment predict with 8 width
        out, (hn, cn) = self.lstm00(bb, (h0[0], c0[0]))

        out00 = self.cnn00(out.permute(0, 2, 1))
        out00 = out00.permute(0, 2, 1)
        out0 = torch.squeeze(out00)  # aggr level 0

        # expand seg0 8 to 32 width
        for i in range(4*s_l):
            if i == 0:
                temp0 = out[:, 0:1, :]
            else:
                temp0 = torch.cat(
                    [temp0, out[:, int(i / 4): int(i / 4) + 1, :]], dim=1
                )

        out, (hn, cn) = self.lstm01(temp0, (h0[1], c0[1]))
        out = temp0 + out

        out01 = self.cnn01(out.permute(0, 2, 1))
        out01 = out01.permute(0, 2, 1)
        out1 = torch.squeeze(out01)  # aggr level 1

        # expand seg1 32 to 96 width
        for i in range(3*4*s_l):
            if i == 0:
                temp0 = out[:, 0:1, :]
            else:
                temp0 = torch.cat(
                    [temp0, out[:, int(i / 6): int(i / 6) + 1, :]], dim=1
                )

        out, (hn, cn) = self.lstm03(temp0, (h0[3], c0[3]))
        out = temp0 + out

        out03 = self.cnn03(out.permute(0, 2, 1))
        out03 = out03.permute(0, 2, 1)
        out3 = torch.squeeze(out03,2)  # aggr level 3  

        # expand seg_label_p 96 to self.predict_days width
        for i in range(self.predict_days):
            if i == 0:
                temp0 = out[:, 0:1]
            else:
                temp0 = torch.cat(
                    [temp0, out[:, int(i / 3): int(i / 3) + 1, :]], dim=1
                )

        out, (hn, cn) = self.lstm04(temp0, (h0[4], c0[4]))
        out = temp0 + out

        out04 = self.cnn04(out.permute(0, 2, 1))
        out04 = out04.permute(0, 2, 1)
        
        # add residue layer
        in5 = x1
        out, (hn, cn) = self.lstm05(in5, (h0[5],c0[5]))  
        
        out = out.permute(0,2,1)          
        out05 = self.cnn02(out)
        out05 = out05.permute(0,2,1)  
        out2 = torch.squeeze(out05,2)
        
        out4 = out04 + out05
        out4 = torch.squeeze(out4,2)
        
        return out0, out1, out2, out3, out4
