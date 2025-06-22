import torch
import torch.nn as nn
from graph_builder import *
from modules.GNN import GNN
from networks import *
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torchstat import stat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# sys.path.append('E:\GitHub\MSAF-master')
# from MSAF import MSAF
from modules.transformer import TransformerEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP_block(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class MLP_Communicator(nn.Module):
    def __init__(self, token, channel, hidden_size, depth=1):
        super(MLP_Communicator, self).__init__()
        self.depth = depth
        self.token_mixer = nn.Sequential(
            Rearrange('b n d -> b d n'),
            MLP_block(input_size=channel, hidden_size=hidden_size),
            Rearrange('b n d -> b d n')
        )
        self.channel_mixer = nn.Sequential(
            MLP_block(input_size=token, hidden_size=hidden_size)
        )

    def forward(self, x):
        for _ in range(self.depth):
            x = x + self.token_mixer(x)
            x = x + self.channel_mixer(x)
        return x


# The probability of dropping a block
class BlockDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(BlockDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p: float = p

    def forward(self, X):
        if self.training:
            blocks_per_mod = [x.shape[1] for x in X]
            mask_size = torch.Size([X[0].shape[0], sum(blocks_per_mod)])
            binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
            mask = binomial.sample(mask_size).to(X[0].device) * (1.0 / (1 - self.p))
            mask_shapes = [list(x.shape[:2]) + [1] * (x.dim() - 2) for x in X]
            grouped_masks = torch.split(mask, blocks_per_mod, dim=1)
            grouped_masks = [m.reshape(s) for m, s in zip(grouped_masks, mask_shapes)]
            X = [x * m for x, m in zip(X, grouped_masks)]
            return X, grouped_masks
        return X, None

class MSAFBlock(nn.Module):
    def __init__(self, in_channels, block_channel, block_dropout=0., lowest_atten=0., reduction_factor=4):
        super(MSAFBlock, self).__init__()
        self.block_channel = block_channel
        self.in_channels = in_channels
        self.lowest_atten = lowest_atten
        self.num_modality = len(in_channels)
        self.reduced_channel = self.block_channel // reduction_factor
        self.block_dropout = BlockDropout(p=block_dropout) if 0 < block_dropout < 1 else None
        self.joint_features = nn.Sequential(
            nn.Linear(self.block_channel, self.reduced_channel),
            nn.BatchNorm1d(self.reduced_channel),
            nn.ReLU(inplace=True)
        )
        self.num_blocks = [math.ceil(ic / self.block_channel) for ic in
                           in_channels]  # number of blocks for each modality
        self.last_block_padding = [ic % self.block_channel for ic in in_channels]
        self.dense_group = nn.ModuleList([nn.Linear(self.reduced_channel, self.block_channel)
                                          for i in range(sum(self.num_blocks))])
        self.soft_attention = nn.Softmax(dim=0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # X: a list of features from different modalities
    def forward(self, X):
        # m1 = torch.rand(4, 32, 64, 64, 50)
        # m2 = torch.rand(4, 16, 32, 32, 53)
        # m1 = torch.rand(128, 50, 32)
        # m2 = torch.rand(128, 100, 32)
        bs_ch = [x.size()[:2] for x in X]
        for bc, ic in zip(bs_ch, self.in_channels):
            assert bc[1] == ic, "X shape and in_channels are different. X channel {} but got {}".format(str(bc[1]),
                                                                                                        str(ic))
        # pad channel if block_channel non divisible
        padded_X = [F.pad(x, (0, pad_size)) for pad_size, x in zip(self.last_block_padding, X)]

        # reshape each feature map: [batch size, N channels, ...] -> [batch size, N blocks, block channel, ...]
        desired_shape = [[x.shape[0], nb, self.block_channel] + list(x.shape[2:]) for x, nb in
                         zip(padded_X, self.num_blocks)]
        reshaped_X = [torch.reshape(x, ds) for x, ds in zip(padded_X, desired_shape)]

        if self.block_dropout:
            reshaped_X, masks = self.block_dropout(reshaped_X)

        # element wise sum of blocks then global ave pooling on channel
        elem_sum_X = [torch.sum(x, dim=1) for x in reshaped_X]
        gap = [F.adaptive_avg_pool1d(sx.view(list(sx.size()[:2]) + [-1]), 1) for sx in elem_sum_X]

        # combine GAP over modalities and generate attention values
        gap = torch.stack(gap).sum(dim=0)  # / (self.num_modality - 1)
        gap = torch.squeeze(gap, -1)
        gap = self.joint_features(gap)
        atten = self.soft_attention(torch.stack([dg(gap) for dg in self.dense_group])).permute(1, 0, 2)
        atten = self.lowest_atten + atten * (1 - self.lowest_atten)

        # apply attention values to features
        atten_shapes = [list(x.shape[:3]) + [1] * (x.dim() - 3) for x in reshaped_X]
        grouped_atten = torch.split(atten, self.num_blocks, dim=1)
        grouped_atten = [a.reshape(s) for a, s in zip(grouped_atten, atten_shapes)]
        if self.block_dropout and self.training:
            reshaped_X = [x * m * a for x, m, a in zip(reshaped_X, masks, grouped_atten)]
        else:
            reshaped_X = [x * a for x, a in zip(reshaped_X, grouped_atten)]
        X = [x.reshape(org_x.shape) for x, org_x in zip(reshaped_X, X)]

        return X


class MSAF(nn.Module):
    def __init__(self, in_channels, block_channel, block_dropout, lowest_atten=0., reduction_factor=4,
                 split_block=1):
        super(MSAF, self).__init__()
        self.num_modality = len(in_channels)
        self.split_block = split_block
        self.blocks = nn.ModuleList([MSAFBlock(in_channels, block_channel, block_dropout, lowest_atten,
                                               reduction_factor) for i in range(split_block)])

    # X: a list of features from different modalities
    def forward(self, X):
        if self.split_block == 1:
            ret = self.blocks[0](X)  # only 1 MSAF block
        else:
            # split into multiple time segments, assumes at dim=2
            segment_shapes = [[x.shape[2] // self.split_block] * self.split_block for x in X]
            for x, seg_shape in zip(X, segment_shapes):
                seg_shape[-1] += x.shape[2] % self.split_block
            segmented_x = [torch.split(x, seg_shape, dim=2) for x, seg_shape in zip(X, segment_shapes)]

            # process segments using MSAF blocks
            ret_segments = [self.blocks[i]([x[i] for x in segmented_x]) for i in range(self.split_block)]

            # put segments back together
            ret = [torch.cat([r[m] for r in ret_segments], dim=2) for m in range(self.num_modality)]

        return ret


class MGCN(nn.Module):
    def __init__(self, model_param):
        super(MGCN, self).__init__()
        self.g_dim = 32
        self.h1_dim = 64 #256
        self.h2_dim = 32
        self.fsn_len = 8
        self.max_feature_layers = (
            1  # number of layers in unimodal models before classifier
        )



        # self.audio_visual_model = BottleAttentionNet()

        self.embed_dim = 32
        # self.embed_dim1 = 64
        # self.cross_transformer = TransformerEncoder(
        #     embed_dim=self.embed_dim,
        #     num_heads=8,
        #     layers=4,
        #     attn_dropout=0.5,
        #     attn_mask=False,
        # )
        # self.classifcation = TransformerEncoder(
        #     embed_dim=self.embed_dim,
        #     num_heads=8,
        #     layers=4,
        #     attn_dropout=0.5,
        #     attn_mask=False,
        # )


        # self.classifcation = TransformerEncoder(
        #     embed_dim=self.embed_dim,
        #     num_heads=8,
        #     layers=4,
        #     attn_dropout=0.5,
        #     attn_mask=False,
        # )

        # self.classifcation_low = TransformerEncoder(
        #     embed_dim=self.embed_dim,
        #     num_heads=4,
        #     layers=2,
        #     attn_dropout=0.5,
        #     attn_mask=False,
        # )

        self.classifcation_high = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=4,
            layers=4,
            attn_dropout=0.5,
            attn_mask=False,
        )


        # self.dropout_a = nn.Dropout3d(p=0.95)
        # self.dropout_v = nn.Dropout3d(p=0.65)
        # self.dropout_t = nn.Dropout3d(p=0.4)
        # self.dropout_a = nn.Dropout3d(p=0.6)
        # self.dropout_v = nn.Dropout3d(p=0.6)
        # self.dropout = nn.Dropout(p=0.5)

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

        self.fsn = nn.Parameter(torch.randn(self.fsn_len, 128, 32, requires_grad=True))



        self.multimodal_classifier = nn.Sequential(nn.Linear(self.embed_dim, 1))
        # self.label_dim = 1  # DataSet = mosi

    def forward(self, x):
        for i in range(self.max_feature_layers):
            if hasattr(self, "text_id"):
                x[self.text_id],_  = self.text_model[i](x[self.text_id]) #(128,50, 768)
                x[self.text_id],_  = self.text_model[i+1](x[self.text_id])#(128,50, 32)

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
            t = self.layer_norm(x[self.text_id])
            v = self.layer_norm(x[self.visual_id])
            a = self.layer_norm(x[self.audio_id])


            t = t.permute(1, 0, 2)
            a = a.permute(1, 0, 2)
            v = v.permute(1, 0, 2)
            fsn =  self.fsn[:,:t.size(1),:]
            nodes_features, edge_index, edge_type_list = create_two_edge_weight_graph_dist(
              v , a, fsn
            ) 

            nodes_features_new = nodes_features.view(
                -1,
                nodes_features.shape[2],
            )  # [12800,32]

            
            nodes_features_new = nodes_features_new.cuda()
            edge_index = edge_index.cuda()
            edge_type_list = edge_type_list.cuda()
            graph_out = self.gcn(nodes_features_new, edge_index, edge_type_list)
            graph_out_new = graph_out.view(
                -1,
                t.shape[1],
                t.shape[2],
            )  #[104,128,32]
            graph_out_fsn_av = self.classifcation_high(graph_out_new)
            if self.fsn_len > 0:
                graph_out_fsn = graph_out_fsn_av[0:self.fsn_len,:,:]
            else:
                graph_out_fsn = graph_out_fsn_av

            # graph_out_fsn_av = torch.cat((graph_out_fsn, a, v), dim=0)
            # graph_out_fsn = graph_out_fsn_av[0:self.fsn_len,:,:]
            t_low = self.classifcation_high(t)
            t_low = self.classifcation_high(t_low, graph_out_fsn, graph_out_fsn)
            t_high = self.classifcation_high(t_low)
            # block = MLP_Communicator(
            #     token=32,  # token 的大小
            #     channel=t.size(1),  # 通道的大小
            #     hidden_size=64,  # 隐藏层的大小
            #     depth=1  # 深度
            # ).to(device)
            result_1 = t_high
            result_2 = self.classifcation_high(t_high,graph_out_fsn_av,graph_out_fsn_av)
            result_3 = self.classifcation_high(t_high + result_2)
            result_4 = self.classifcation_high(graph_out_fsn_av,t_high,t_high)
            # result_4 = self.classifcation_high(graph_out_fsn)
            # result_3 = block(self.classifcation_high(graph_out_fsn_high,t_high,t_high))
            # result_4 = block(self.classifcation_high(t_high,graph_out_fsn_high,graph_out_fsn_high))
            T = result_1[-1]
            # G = result_2[-1]
            T1 = result_4[-1]
            G1 = result_3[-1]
            result = T + T1 + G1
            

            result_out = self.multimodal_classifier(result)
            # result = self.activation(result)

        return result, result_out



class CrossAttention(nn.Module):        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/crossvit.py
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1)  # 输入通道为1，输出通道为784，核大小为1         Add it myself
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        num = self.num_heads
        # B1C -> B1H(C/H) -> BH1(C/H)
        q = self.wq(x[:, 0:1, ...])
        q = q.reshape(B, 1, -1, C).permute(0, 2, 1, 3).cuda()
        
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, -1, C).permute(0, 2, 1, 3)


        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, -1, C).permute(0, 2, 1, 3)



        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)



        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C

        x = self.conv(x)           # Add it myself

        x = self.proj(x)

        x = self.proj_drop(x)

        return x