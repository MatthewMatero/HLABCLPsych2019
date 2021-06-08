import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import csv
import pickle
import sys
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import argparse
import numpy as np
import pandas as pd
import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



class BiLSTM_Suicide_Classifier(nn.Module):


    def __init__(self, opt, sc_input_dim, user_adapt_dict, user_feat_vec_dict, user_adapt_size, user_feat_vec_dim, num_layer=1,num_direction=1):
        """Initialize the classifier: defines architecture and basic hyper-parameters. """
        super(BiLSTM_Suicide_Classifier, self).__init__()

        ##LSTM parameters:
        self.sc_input_dim = sc_input_dim ##size of sc input embedding 

        self.hidden_dim = opt.hidden_dim   ##size of hidden state vector

        self.num_layer = num_layer ##layers for Suicide LSTM
        self.num_direction = int(opt.num_direction) ##1 or 2 depending if bidirectiona;

        self.dropout_rate=float(opt.dropout) ##rate of dropout, applied between the two LSTMs
        self.is_cuda=opt.cuda  #whether cuda (GPUs) are available
        self.attn_act=opt.attn_act

        self.user_adapt_dict=user_adapt_dict
        self.user_feat_vec_dict=user_feat_vec_dict
        self.cell_type=opt.cell_type
        self.adapt=opt.adapt
        self.user_feat=opt.user_feat


        ##Architecture: suicide LSTM
        if opt.cell_type=='LSTM':
            self.sc_RNN = nn.LSTM(sc_input_dim, self.hidden_dim, bidirectional=num_direction==2)
        elif opt.cell_type=='GRU':
            self.sc_RNN = nn.GRU(sc_input_dim, self.hidden_dim, bidirectional=num_direction==2)
        else:
            print("Invalid Cell Type:",cell_type)
            sys.exit()

        self.sc_attn=nn.Linear(num_direction*self.hidden_dim, num_direction*self.hidden_dim)
        self.sc_attn_combine=nn.Linear(num_direction*self.hidden_dim, num_direction*self.hidden_dim, bias=False) # dimension-wise attention
        #self.sc_attn_combine=nn.Linear(num_direction*self.hidden_dim, 1, bias=False) # 1 attention weight per hidden vector
        self.sc_dropout=nn.Dropout(p=self.dropout_rate)



        ##Architecture: concant of (sc, non) hidden vector to prediction 
       
            

        self.hidden_cat_to_sc_level = nn.Linear(num_direction*self.hidden_dim, opt.class_num)
        
        



        ## Check the parameters for the current model
        print("[Model Initialization]:")
        print("Cell Type: "+str(self.cell_type))
        print("SW Input Dimenstion:",self.sc_input_dim)
        print("# of Adaptaion Factors:",user_adapt_size)
        print("# of User Features:",user_feat_vec_dim)
        print("Hidden Dimension: " +str(self.hidden_dim))
        print("Hidden Layers: "+str(self.num_layer))
        print("# of Directions for LSTM: "+str(self.num_direction))
        print("Dropout Rate: "+str(self.dropout_rate))
        print("CUDA Usage: "+str(self.is_cuda))


        if self.is_cuda:
            self.sc_RNN = self.sc_RNN.cuda()
            self.hidden_cat_to_sc_level = self.hidden_cat_to_sc_level.cuda()

    def attn_mul(self,rnn_outputs, attn_weights):
        attn_vectors = None
        for i in range(rnn_outputs.size(0)):
            h_i = rnn_outputs[i]
            a_i = attn_weights[i]
            h_i = a_i * h_i
            h_i = h_i.unsqueeze(0)
            if(attn_vectors is None):
                attn_vectors=h_i
            else:
                attn_vectors = torch.cat((attn_vectors,h_i),0)
        return torch.sum(attn_vectors, 0).unsqueeze(0)

    



    def forward(self, sc_embedding_seq, user_id):
        """Defines the forward pass through the full deep learning model"""
        
        # use batch input and get all the hidden vectors and concatenate them
        
        sc_embedding_seq = torch.cat(sc_embedding_seq).view(len(sc_embedding_seq), 1, -1)
        if self.is_cuda:
            sc_embedding_seq=sc_embedding_seq.cuda()
        if self.cell_type=='LSTM':
            sc_output, (sc_hidden, sc_cell_state) = self.sc_RNN(sc_embedding_seq,self.init_sc_hidden())
        elif self.cell_type=='GRU':
            sc_output, sc_hidden = self.sc_RNN(sc_embedding_seq,self.init_sc_hidden())

        if self.attn_act=='Tanh':
            sc_annotation = torch.tanh(self.sc_attn(sc_output))
        else:
            sc_annotation = self.sc_attn(sc_output)
        sc_attn = F.softmax(self.sc_attn_combine(sc_annotation),dim=0)
        sc_attn_vec = self.attn_mul(sc_output,sc_attn)
        dropped_sc_attn_vec= self.sc_dropout(sc_attn_vec)
        sc_class_vec = self.hidden_cat_to_sc_level(dropped_sc_attn_vec.view(1,-1))
        score = F.log_softmax(sc_class_vec,dim=1) # Calculate softmax for the sentiment space            

        return score

    
    ## Remove history of the hidden vector from the last instance
    def init_sc_hidden(self):
        if self.is_cuda:
            if self.cell_type=='LSTM':
                return (
                    autograd.Variable(
                        torch.zeros(self.num_layer * self.num_direction, 1, self.hidden_dim)).cuda(),
                    autograd.Variable(
                        torch.zeros(self.num_layer * self.num_direction, 1, self.hidden_dim)).cuda())
            elif self.cell_type=='GRU':
                return autograd.Variable(
                        torch.zeros(self.num_layer * self.num_direction, 1, self.hidden_dim)).cuda()
        else:
            if self.cell_type=='LSTM':
                return (
                    autograd.Variable(
                        torch.zeros(self.num_layer * self.num_direction, 1, self.hidden_dim)).cuda(),
                    autograd.Variable(
                        torch.zeros(self.num_layer * self.num_direction, 1, self.hidden_dim)).cuda())
            elif self.cell_type=='GRU':
                return autograd.Variable(
                        torch.zeros(self.num_layer * self.num_direction, 1, self.hidden_dim))


