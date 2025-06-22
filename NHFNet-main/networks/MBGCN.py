from graph_builder import *
from modules.GNN import GNN
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import sys

# sys.path.append('E:\GitHub\MSAF-master')
# from MSAF import MSAF
from modules.transformer import TransformerEncoder


class BottleAttentionNet(nn.Module):
    def __init__(self):
        super(BottleAttentionNet, self).__init__()
        self.embed_dim = 32
        self.seq = 4
        self.layer_unimodal = 1
        self.layer_multimodal = 2  # cmu_mosei
        self.audio_linear = nn.Linear(74, self.embed_dim)
        self.visual_linear = nn.Linear(35, self.embed_dim)
        self.transformer = TransformerEncoder(
            embed_dim=self.embed_dim, num_heads=8, layers=4, attn_mask=False
        )

    def forward(self, audio, visual):
        audio = self.audio_linear(audio).permute(1, 0, 2)
        for i in range(self.layer_unimodal):
            audio = self.transformer(audio)

        visual = self.visual_linear(visual).permute(1, 0, 2)
        for i in range(self.layer_unimodal):
            visual = self.transformer(visual)

        fsn = torch.zeros(self.seq, audio.size(1), self.embed_dim).to(device)
        x = torch.cat([audio, fsn], dim=0)

        for i in range(self.layer_multimodal):
            if i == 0:
                x = self.transformer(x)
                x = torch.cat([x[audio.size(0) :, :, :], visual], dim=0)
                x = self.transformer(x)
            else:
                x = self.transformer(x)

        return x


class MBGCN(nn.Module):
    def __init__(self, model_param):
        super(MBGCN, self).__init__()
        self.g_dim = 32
        self.h1_dim = 32
        self.h2_dim = 32
        self.max_feature_layers = (
            1  # number of layers in unimodal models before classifier
        )

        self.audio_visual_model = BottleAttentionNet()

        self.embed_dim = 32
        self.embed_dim1 = 64
        self.cross_transformer = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=8,
            layers=4,
            attn_dropout=0.4,
            attn_mask=False,
        )
        self.classifcation = TransformerEncoder(
            embed_dim=self.embed_dim, num_heads=8, layers=4, attn_mask=False
        )

        self.classifcation1 = TransformerEncoder(
            embed_dim=self.embed_dim1, num_heads=8, layers=4, attn_mask=False
        )

        self.dropout_a = nn.Dropout3d(p=0.95)
        self.dropout_v = nn.Dropout3d(p=0.65)
        self.dropout_t = nn.Dropout3d(p=0.4)


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

        self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim1)

        self.linear = nn.Linear(self.embed_dim1, self.embed_dim)

        self.multimodal_classifier = nn.Sequential(nn.Linear(self.embed_dim, 1))

        self.label_dim = 1  # DataSet = mosi

    def forward(self, x):
        for i in range(self.max_feature_layers):
            if hasattr(self, "text_id"):
                x[self.text_id], _ = self.text_model[i](x[self.text_id])
                x[self.text_id], _ = self.text_model[i + 1](x[self.text_id])


            # 生成音频和图像进行聚合块以后的结点
            audio_visual_feature = self.audio_visual_model(
                x[self.audio_id], x[self.visual_id]
            )

            if hasattr(self, "audio_id"):
                x[self.audio_id], _ = self.audio_model[i](x[self.audio_id])
                x[self.audio_id], _ = self.audio_model[i + 1](x[self.audio_id])

            if hasattr(self, "visual_id"):
                x[self.visual_id], _ = self.visual_model[i](x[self.visual_id])
                x[self.visual_id], _ = self.visual_model[i + 1](x[self.visual_id])

            # 生成text的结点
            x[self.text_id] = self.layer_norm(x[self.text_id]).permute(1, 0, 2)
            t =  x[self.text_id]
            text_feature = self.classifcation(x[self.text_id])  # [50, 12, 32]

            # 生成vision的结点
            x[self.visual_id] = self.layer_norm(x[self.visual_id]).permute(1, 0, 2)
            v = x[self.visual_id]
            vision_feature = self.classifcation(x[self.visual_id])  # [50, 12, 32]

            # 生成audio的结点
            x[self.audio_id] = self.layer_norm1(x[self.audio_id]).permute(1, 0, 2)
            a = self.linear(x[self.audio_id])
            audio_feature = self.classifcation1(x[self.audio_id])  # [50, 12, 32]
            audio_feature = self.linear(audio_feature)



            # nodes_features, edge_index, edge_type_list = create_two_edge_graph(
            #     audio_visual_feature, text_feature
            # )
            # nodes_features, edge_index, edge_type_list = create_three_edge_graph(
            #     vision_feature, audio_feature, text_feature
            # )
            v = self.dropout_v(v)
            a = self.dropout_a(a)
            t = self.dropout_t(t)
            nodes_features, edge_index, edge_type_list = create_three_edge_graph(
                v, a, t
            )
            print(nodes_features.shape)


            # print(torch.stack(edge_type).shape)
            # print("nodes_features", nodes_features.shape)
            nodes_features = nodes_features.view(
                -1,
                nodes_features.shape[2],
            )  # [1200,32]
            # print("nodes_features", nodes_features.shape)
            nodes_features = self.layer_norm(nodes_features)
            # print("nodes_features", nodes_features.shape)

            # 这里需要点特征，边的邻接图，边类型
            graph_out = self.gcn(nodes_features, edge_index, edge_type_list)  # [1200, 8]
            # graph_out = self.gcn(nodes_features, edge_index)

            # graph_out = graph_out.reshape(-1, self.embed_dim)
            # graph_out = torch.narrow(graph_out, 0, 0, length_num)
            graph_out = self.activation(graph_out)
            # text_feature = self.classifcation(text_feature)


            # print("graph_out", graph_out.shape)
            # text_feature = text_feature.view(
            #     -1,
            #     text_feature.shape[2],
            # )
            # text_feature = self.layer_norm(text_feature)
            # text_feature = self.text_mlp(text_feature)

            # result = text_feature + graph_out

            # result = self.activation(result)

            l_av = self.cross_transformer(audio_visual_feature, text_feature, text_feature)

            av_l = self.cross_transformer(
                text_feature, audio_visual_feature, audio_visual_feature
            )

            l_result = self.classifcation(av_l)
            av_result = self.classifcation(l_av)

            # result1 = result1[-1]
            # result2 = audio_visual_feature[-1]
            # l_result = l_result[-1]
            # av_result = av_result[-1]

            # result = graph_out[-1] + text_feature[-1] + audio_visual_feature[-1] + l_result[-1] +  av_result[-1]

            result = graph_out[-1] + text_feature[-1] 
            print( text_feature)
            print( graph_out)
            print( graph_out[-1].shape)
            print(text_feature[-1].shape)
            # print("result.shape=", result.shape)
            # if result.shape[0]  == 1200:
            #     result = result.view(4, -1)
            # if result.shape[0]  == 300:
            #     result = result.view(3, -1)
            # if result.shape[0]  == 200:
            #     result = result.view(2, -1)
            # result = self.fcn800(result)
            # result = self.finalW(result)
            # result = result[-1]
            result = self.multimodal_classifier(result)
        return result
