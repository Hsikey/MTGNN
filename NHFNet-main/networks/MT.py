from graph_builder import *
from modules.GNN import GNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import sys

# sys.path.append('E:\GitHub\MSAF-master')
# from MSAF import MSAF
from modules.transformer import TransformerEncoder



class MT(nn.Module):
    def __init__(self, model_param):
        super(MT, self).__init__()
        self.g_dim = 32
        self.h1_dim = 128
        self.h2_dim = 32
        
        self.max_feature_layers = (
            1  # number of layers in unimodal models before classifier
        )

        # self.audio_visual_model = BottleAttentionNet()

        self.embed_dim = 32
        self.embed_dim1 = 64
        self.classifcation = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=8,
            layers=4,
            attn_dropout=0.6,
            attn_mask=False,
        )
        self.classifcation1 = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=8,
            layers=4,
            attn_dropout=0.6,
            attn_mask=False,
        )
        self.classifcation2 = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=8,
            layers=4,
            attn_dropout=0.6,
            attn_mask=False,
        )


        self.activation = nn.LeakyReLU(negative_slope=0.1)
        

        self.relu = nn.ReLU()


        if "visual" in model_param:
            visual_model = model_param["visual"]["model"]
            # visual model layers
            self.visual_model = nn.ModuleList([visual_model.lstm1, visual_model.lstm2])
            self.visual_id = model_param["visual"]["id"]
            # print("########## Visual ##########")
            # print(visual_model)

        if "audio" in model_param:
            audio_model = model_param["audio"]["model"]
            # audio model layers
            self.audio_model = nn.ModuleList([audio_model.lstm1, audio_model.lstm2])
            self.audio_id = model_param["audio"]["id"]
            # print("########## Audio ##########")
            # print(audio_model)

        if "bert" in model_param:
            text_model = model_param["bert"]["model"]
            # text model layers
            self.text_model = nn.ModuleList([text_model.lstm1, text_model.lstm2])
            self.text_id = model_param["bert"]["id"]


        self.layer_norm = nn.LayerNorm(self.embed_dim)


        self.multimodal_classifier = nn.Sequential(nn.Linear(self.embed_dim, 1))

        # self.label_dim = 1  # DataSet = mosi

    def forward(self, x):
        for i in range(self.max_feature_layers):
            if hasattr(self, "text_id"):
                x[self.text_id],_  = self.text_model[i](x[self.text_id])
                x[self.text_id],_  = self.text_model[i+1](x[self.text_id])

            if hasattr(self, "audio_id"):
                x[self.audio_id] = self.audio_model[i](x[self.audio_id])
                x[self.audio_id] = self.relu(x[self.audio_id])
                x[self.audio_id] = self.audio_model[i+1](x[self.audio_id])
                x[self.audio_id] = self.relu(x[self.audio_id])

            if hasattr(self, "visual_id"):
                x[self.visual_id] = self.visual_model[i](x[self.visual_id])
                x[self.visual_id] = self.relu(x[self.visual_id])
                x[self.visual_id] = self.visual_model[i+1](x[self.visual_id])
                x[self.visual_id] = self.relu(x[self.visual_id])

            # 生成text的结点
            t = x[self.text_id].permute(1, 0, 2)
            # text_feature = self.classifcation(t)  # [50, 12, 32]

            # 生成vision的结点
            v = x[self.visual_id].permute(1, 0, 2)
            # vision_feature = self.classifcation(v)

            # 生成audio的结点
            # x[self.audio_id] = self.layer_norm1(x[self.audio_id]).permute(1, 0, 2)
            a = x[self.audio_id].permute(1, 0, 2)

            result_ta = self.classifcation(t,a,a)
            result_tv = self.classifcation(t,v,v)
            result_at = self.classifcation(a,t,t)
            result_av = self.classifcation(a,v,v)
            result_vt = self.classifcation(v,t,t)
            result_va = self.classifcation(v,a,a)
            result = result_ta[-1] + result_tv[-1] + result_at[-1] + result_av[-1] + result_vt[-1] + result_va[-1]
            
            result_out = self.multimodal_classifier(result)
            # result = self.activation(result)

        return result, result_out
