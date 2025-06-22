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




class Adapter(nn.Module):
    def __init__(self, model_param):
        super(Adapter, self).__init__()
        self.embed_dim = 32
        self.classifcation = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=8,
            layers=4,
            attn_dropout=0.5,
            attn_mask=False,
        )


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

        self.max_feature_layers = (
            1  # number of layers in unimodal models before classifier
        )

        self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.multimodal_classifier = nn.Sequential(nn.Linear(self.embed_dim, 1))


    def forward(self, x):
        for i in range(self.max_feature_layers):
            if hasattr(self, "text_id"):
                x[self.text_id],_  = self.text_model[i](x[self.text_id])
                x[self.text_id],_  = self.text_model[i+1](x[self.text_id])

            if hasattr(self, "audio_id"):
                x[self.audio_id] = self.audio_model[i](x[self.audio_id])
                x[self.audio_id] = self.audio_model[i+1](x[self.audio_id])

            if hasattr(self, "visual_id"):
                x[self.visual_id] = self.visual_model[i](x[self.visual_id])
                x[self.visual_id] = self.visual_model[i+1](x[self.visual_id])

            # 生成text的结点
            t = x[self.text_id]
            t = self.layer_norm(t).permute(1, 0, 2)
            # 生成vision的结点
            v = x[self.visual_id]
            v = self.layer_norm(v).permute(1, 0, 2)
            a = x[self.audio_id]
            a = self.layer_norm(a).permute(1, 0, 2)
            vision_feature = self.classifcation(v)
            text_feature = self.classifcation(t)
            audio_feature = self.classifcation(a)

            result = text_feature[-1] + vision_feature[-1]  + audio_feature[-1]

            result = self.multimodal_classifier(result)

        return result
