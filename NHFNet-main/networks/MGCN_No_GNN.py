from graph_builder import *
from networks import *
from modules.GNN import GNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import sys

from modules.transformer import TransformerEncoder



class MGCN_No_GNN(nn.Module):
    def __init__(self, model_param):
        super(MGCN_No_GNN, self).__init__()
        self.g_dim = 32
        self.h1_dim = 256
        self.h2_dim = 64
        self.fsn_len = 2
        self.max_feature_layers = (
            1  # number of layers in unimodal models before classifier
        )

        # self.audio_visual_model = BottleAttentionNet()

        self.embed_dim = 32
        self.embed_dim1 = 64
        # self.cross_transformer = TransformerEncoder(
        #     embed_dim=self.embed_dim,
        #     num_heads=8,
        #     layers=4,
        #     attn_dropout=0.4,
        #     attn_mask=False,
        # )
        self.classifcation = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=8,
            layers=4,
            attn_dropout=0.5,
            attn_mask=False,
        )


        # self.dropout_a = nn.Dropout3d(p=0.95)
        # self.dropout_v = nn.Dropout3d(p=0.65)
        # self.dropout_t = nn.Dropout3d(p=0.4)
        # self.dropout_a = nn.Dropout3d(p=0.6)
        # self.dropout_v = nn.Dropout3d(p=0.6)
        self.dropout = nn.Dropout(p=0.5)

        # self.classifcation1 = TransformerEncoder(
        #     embed_dim=self.embed_dim1, num_heads=8, layers=4, attn_mask=False
        # )

        self.activation = nn.LeakyReLU(negative_slope=0.1)
        


        self.gcn = GNN(self.g_dim, self.h1_dim, self.h2_dim)

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

        
        if "adapter" in model_param:
            adapter = model_param["adapter"]["model"]
            # text model layers
            self.adapter = adapter.classifcation


        # self.encoder.append(tea.)

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
                # x[self.audio_id] = self.relu(x[self.audio_id])
                x[self.audio_id] = self.audio_model[i+1](x[self.audio_id])
                # x[self.audio_id] = self.relu(x[self.audio_id])

            if hasattr(self, "visual_id"):
                x[self.visual_id] = self.visual_model[i](x[self.visual_id])
                # x[self.visual_id] = self.relu(x[self.visual_id])
                x[self.visual_id] = self.visual_model[i+1](x[self.visual_id])
                # x[self.visual_id] = self.relu(x[self.visual_id])

            # 生成text的结点
            t = x[self.text_id]
            t = self.layer_norm(t).permute(1, 0, 2)
            # t = self.adapter(t)  # [50, 12, 32]

            # 生成vision的结点
            v = self.layer_norm(x[self.visual_id]).permute(1, 0, 2)
            # v = self.layer_norm(v)
            # v = self.adapter(v)

            # 生成audio的结点
            # x[self.audio_id] = self.layer_norm1(x[self.audio_id]).permute(1, 0, 2)
            # a = self.linear(x[self.audio_id])
            a = self.layer_norm(x[self.audio_id]).permute(1, 0, 2)
            # a = self.layer_norm(a)
            # a = self.adapter(a)



            t_a = self.classifcation(
                t, a, a
            )

            t_v = self.classifcation(
                t, v, v
            )

            a_t = self.classifcation(
                a, t, t
            )

            a_v = self.classifcation(
                a, v, v
            )

            v_a = self.classifcation(
                v, a, a
            )

            v_t = self.classifcation(
                v, t, t
            )
            

            t_a = self.classifcation(
                t_a
            )

            t_v = self.classifcation(
                t_v
            )

            a_t = self.classifcation(
                a_t
            )

            a_v = self.classifcation(
                a_v
            )

            v_a = self.classifcation(
                v_a
            )

            v_t = self.classifcation(
                v_t
            )

            result =  t_a[-1] + t_v[-1] + a_t[-1] + a_v[-1] + v_a[-1] + v_t[-1]


            result = self.multimodal_classifier(result)
            # result = self.activation(result)

        return result
