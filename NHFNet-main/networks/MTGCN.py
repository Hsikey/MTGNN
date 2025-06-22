from graph_builder import *
from networks import *
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



class MTGCN(nn.Module):
    def __init__(self, model_param):
        super(MTGCN, self).__init__()
        self.g_dim = 32
        self.h1_dim = 128
        self.h2_dim = 32
        
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

        
        # if "adapter" in model_param:
        #     adapter = model_param["adapter"]["model"]
        #     # text model layers
        #     self.adapter = adapter.classifcation


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
            # text_feature = self.classifcation(t)  # [50, 12, 32]

            # 生成vision的结点
            v = self.layer_norm(x[self.visual_id]).permute(1, 0, 2)
            # v = self.layer_norm(v)
            # vision_feature = self.classifcation(v)

            # 生成audio的结点
            # x[self.audio_id] = self.layer_norm1(x[self.audio_id]).permute(1, 0, 2)
            # a = self.linear(x[self.audio_id])
            a = self.layer_norm(x[self.audio_id]).permute(1, 0, 2)
            # a = self.layer_norm(a)
            # audio_feature = self.classifcation(a)

            # v = torch.matmul(v, self.weight)
            # a = torch.matmul(a, self.weight)
            # t = torch.matmul(t, self.weight)
            # t = t + text_feature
            nodes_features, edge_index, edge_type_list = create_two_edge_weight_graph_new(
                v, a
            ) #[256, 50 ,32]

            # nodes_features, edge_index, edge_type_list = create_two_edge_weight_graph_no_jhk(
            #     v, a
            # ) #[256, 50 ,32]
            
            # print(f'{nodes_features.shape}-----{fea[0].shape}-------{fea[1].shape}-------{fea[2].shape}')

            nodes_features_new = nodes_features.view(
                -1,
                nodes_features.shape[2],
            )  # [12800,32]
            # nodes_features = self.layer_norm(nodes_features)

            # 这里需要点特征，边的邻接图，边类型
            # graph_out = self.gcn(
            #     nodes_features, edge_index, edge_type_list
            # )  # [1200, 8]
            graph_out = self.gcn(nodes_features_new, edge_index, edge_type_list)
            # t_size = text_feature.shape[0]
            # graph_out = graph_out.view(-1,text_feature.shape[1],text_feature.shape[2])
            # print(graph_out.shape)

            # graph_out = F.relu(graph_out)

            # graph_out = graph_out[0:t_size,:,:]
            # result = graph_out + text_feature
            # result = self.multimodal_classifier(result[-1])

            # l_v = self.classifcation(
            #     text_feature, v, v
            # )

            # l_a = self.classifcation(
            #     text_feature, a, a
            # )
            # graph_out = torch.chunk(graph_out, 3, dim=0)[0]
            
            graph_out_new = graph_out.view(
                -1,
                t.shape[1],
                t.shape[2],
            )  #[104,128,32]
            graph_out_fsn = graph_out_new[0:4,:,:]
            graph_out_fsn = self.activation(graph_out_fsn)


            # t = t.permute(1, 0, 2) #[50,128,32]
            # result = torch.cat((graph_out, text_feature),dim=0)
            # result = torch.cat((graph_out_new, t),dim=0)
            # result = self.lin2(self.drop(F.relu(self.lin2(graph_out_new))))
            # result = self.classifcation(text_feature, graph_out, graph_out)
            # result_1 = self.classifcation(graph_out_new, t, t)
            # result_2 = self.classifcation(t, graph_out_new, graph_out_new)
            result_1 = self.classifcation(t,graph_out_fsn,graph_out_fsn)
            result_3 = self.classifcation(graph_out_fsn,t,t)
            text_feature = self.classifcation(t)
            result_2 = self.classifcation(graph_out_fsn)
            result =  result_1[-1]+ result_3[-1] + text_feature[-1] + result_2[-1] 
            
            result = self.multimodal_classifier(result)
            # result = self.activation(result)

        return result
