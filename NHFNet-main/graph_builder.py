import main_utils as util
import math
import numpy as np
import torch

import main_utils as util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ifZero(tensor):
    a, b = tensor.shape
    tmp = torch.zeros(a, b).to(device)
    return torch.equal(tmp, tensor)


def create_full_graph(vision_audio, text):
    features, edge_index_list, batch_edge_types = [], [], []
    for v_a, t in zip(vision_audio, text):
        edge_types = []
        vision_audio_node_index = torch.arange(v_a.shape[0])
        text_node_index = torch.arange(t.shape[0]) + len(vision_audio_node_index)
        edge_type_offset = 0
        create_edge_type_uni_modal(
            vision_audio_node_index, edge_types, edge_type_offset
        )
        edge_type_offset += 16
        create_edge_type_uni_modal(text_node_index, edge_types, edge_type_offset)
        edge_type_offset += 16
        create_edge_type_cross_modal(
            vision_audio_node_index, text_node_index, edge_types, edge_type_offset
        )
        # node_index = torch.cat((vision_audio_node_index, text_node_index))
        # # Construct a full graph where each node is connected to every other node
        # src, dst = torch.meshgrid(node_index, node_index)
        # src, dst = src.flatten(), dst.flatten()
        # edge_index = torch.stack((src, dst), 0)
        # edge_index_list.append(edge_index.to(device))
        # # Construct the node features
        node_features = torch.cat((v_a, t), 0)
        features.append(node_features.to(device))
        batch_edge_types.append(torch.stack(edge_types).permute(1, 0).to(device))
        # assert node_features.shape[0] == max(node_index) + 1

    vision_audio_node = torch.arange(vision_audio.shape[1])
    text_node = torch.arange(text.shape[1]) + len(vision_audio_node)
    node_index = torch.cat((vision_audio_node, text_node))
    # Construct a full graph where each node is connected to every other node
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index = torch.stack((src, dst), 0).to(device)
    return features, edge_index, batch_edge_types


def create_full_edge_graph(vision_audio, text):
    features, edge_type_list = [], []
    num_edges = text.shape[0] + vision_audio.shape[0]
    edge_type_offset = 0
    for i, v_a in enumerate(vision_audio): #[58,4,32]
        features.append(v_a.to(device))
        num_edges = text.shape[0] 
        edge_type = torch.ones(num_edges, dtype=torch.long) * i  # Assign a unique edge type for each example
        edge_type_list.append(edge_type)

    for j, t in enumerate(text): #[50,4,32]
        features.append(t.to(device))
        num_edges = vision_audio.shape[0] 
        edge_type_offset = j + 1
        edge_type = torch.ones(num_edges, dtype=torch.long) * edge_type_offset  # Assign a unique edge type for each example
        edge_type_list.append(edge_type)

    vision_audio_node = torch.arange(vision_audio.shape[0])
    text_node = torch.arange(text.shape[0])  + len(vision_audio_node)
    node_index = torch.cat((vision_audio_node, text_node))
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index = torch.stack((src, dst), 0).to(device)
    # Construct a full graph where each node is connected to every other node
    # src, dst = torch.meshgrid(vision_audio_node, text_node)
    # src, dst = src.flatten(), dst.flatten()
    # edge_index_v_a_to_t = torch.stack((src, dst), 0).to(device) #[2,2900]

    # src, dst = torch.meshgrid(text_node, vision_audio_node)
    # src, dst = src.flatten(), dst.flatten()
    # edge_index_t_to_v_a = torch.stack((src, dst), 0).to(device) #[2,2900]

    # src, dst = torch.meshgrid(text_node, text_node)
    # src, dst = src.flatten(), dst.flatten()
    # edge_index_t = torch.stack((src, dst), 0).to(device) #[2,58*58]

    # src, dst = torch.meshgrid(vision_audio_node, vision_audio_node)
    # src, dst = src.flatten(), dst.flatten()
    # edge_index_v_a = torch.stack((src, dst), 0).to(device) #[2,50*50]

    # edge_index = torch.cat((edge_index_v_a_to_t, edge_index_t_to_v_a, edge_index_t, edge_index_v_a), 1)  #[2,5800]
    # edge_index = torch.cat((edge_index, edge_index_t, edge_index_v_a), 1)  #[2,5800]

    edge_type_list = torch.cat(edge_type_list) #[5800]
    nodes_features = torch.stack(features) #[100,4,32] 

    return nodes_features, edge_index, edge_type_list


def create_two_edge_graph(vision_audio, text):
    features, edge_type_list = [], []
    num_edges = text.shape[0] + vision_audio.shape[0]
    edge_type_offset = 0
    for i, v_a in enumerate(vision_audio): #[58,4,32]
        features.append(v_a.to(device))
        edge_type_offset += i
        edge_type = torch.ones(num_edges, dtype=torch.long) * edge_type_offset  # Assign a unique edge type for each example
        edge_type_list.append(edge_type)

    for j, t in enumerate(text): #[50,4,32]
        features.append(t.to(device))
        edge_type_offset = j + 1
        edge_type = torch.ones(num_edges, dtype=torch.long) * edge_type_offset  # Assign a unique edge type for each example
        edge_type_list.append(edge_type)

    vision_audio_node = torch.arange(vision_audio.shape[0])
    text_node = torch.arange(text.shape[0])  + len(vision_audio_node)
    node_index = torch.cat((vision_audio_node, text_node))
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index = torch.stack((src, dst), 0).to(device)

    edge_type_list = torch.cat(edge_type_list)  # [5800]
    edge_type_list = torch.tensor(edge_type_list, dtype=torch.float)
    nodes_features = torch.stack(features) #[100,4,32] 

    return nodes_features, edge_index, edge_type_list

def create_two_edge_weight_graph(vision, audio):
    features, edge_type_list = [], []
    nodes = torch.cat((vision, audio), dim=0)
    for _, v in enumerate(vision): #[128,50,32]
        features.append(v.to(device))

    for _, a in enumerate(audio): #[128,50,32]
        features.append(a.to(device))

    vision_node = torch.arange(vision.shape[0]) #128
    audio_node = torch.arange(audio.shape[0])  + len(vision_node) #128

    for i, nodes_x in enumerate(nodes):
        for j, nodes_y in enumerate(nodes):
            # if i in vision_node and j in vision_node:
            #     edge_type = 0
            # elif i in audio_node and j in audio_node:
            #     edge_type = 0
            # else:  
            edge_type = torch.dist(nodes_x, nodes_y)
            edge_type_list.append(edge_type)

    node_index = torch.cat((vision_node, audio_node))
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index = torch.stack((src, dst), 0).to(device) #[2,256*256]

    # edge_type_list = torch.cat(edge_type_list)  # []
    edge_type_list = torch.tensor(edge_type_list, dtype=torch.float)#[256*256]
    edge_type_list_1 = util.data_normal(edge_type_list)
    # edge_type_list_2 = edge_type_list_1 * -1 + 1
    edge_type_list_index = edge_type_list_1 < 0.4
    index = np.where(edge_type_list_index)
    edge_index_fin = np.delete(edge_index,index,axis=1) 
    edge_type_list_fin = np.delete(edge_type_list_1,index) 

    nodes_features = torch.stack(features) #[256,50,32] 

    return nodes_features, edge_index_fin, edge_type_list_fin

def create_two_edge_weight_graph_new(vision, audio, fsn_len):
    features, edge_type_list = [], []
    fsn = torch.zeros(fsn_len, audio.size(1),audio.size(2))
    # init.kaiming_uniform_(fsn, mode='fan_in', nonlinearity='relu')
    nodes = torch.cat((fsn, vision, audio), dim=0)
    features = np.split(nodes, nodes.size(0), axis=0)
    features = [np.squeeze(arr, axis=0) for arr in features]
    features_index_num = [len(features) for _ in range(len(features))]
    fsn_node = torch.arange(fsn.shape[0]) 
    vision_node = len(fsn_node) + torch.arange(vision.shape[0]) #54
    audio_node = torch.arange(audio.shape[0])  + len(vision_node) 


    for i, nodes_x in enumerate(features):
        for j, nodes_y in enumerate(features):
                # edge_type_1 = util.atom_calculate_edge_weight(nodes_x.flatten(), nodes_y.flatten())
            edge_type = torch.dist(nodes_x, nodes_y)
            edge_type_list.append(edge_type)

    node_index = torch.cat((fsn_node, vision_node, audio_node))
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index = torch.stack((src, dst), 0).to(device) #[2,104*104]

    # edge_type_list = torch.cat(edge_type_list)  # []
    edge_type_list = torch.tensor(edge_type_list, dtype=torch.float)#[104*104]
    edge_type_list_1 = util.data_normal(edge_type_list)
    # edge_type_list_2 = edge_type_list_1 * -1 + 1
    edge_type_list_index = edge_type_list_1  < 0.2  #2543 删除距离小于0.3的长度
    index = np.where(edge_type_list_index)
    index_list = list(index[0])
    features_delete_index = [value // len(features) for value in index_list]
    for index in features_delete_index:
       features_index_num[index] -= 1


    edge_type_list_index = edge_type_list_1  > 0.8  #2543 删除距离小于0.3的长度
    index_2 = np.where(edge_type_list_index)
    index_list_2 = list(index_2[0])
    features_delete_index_2 = [value // len(features) for value in index_list_2]
    for index in features_delete_index_2:
       features_index_num[index] -= 1

    edge_index_fin = np.delete(edge_index,index,axis=1) 
    edge_type_list_fin = np.delete(edge_type_list_1,index) 

    nodes_features = torch.stack(features,dim=0) #[104,128,32] 

    return nodes_features, edge_index_fin, edge_type_list_fin


def create_two_edge_weight_graph_new(text, vision, audio, fsn_len):
    features, edge_type_list = [], []
    fsn = torch.randn(fsn_len, text.size(1), text.size(2)).cuda()
    # init.kaiming_uniform_(fsn, mode='fan_in', nonlinearity='relu')
    nodes = torch.cat((fsn, vision, audio), dim=0)
    features = np.split(nodes, nodes.size(0), axis=0)
    features = [np.squeeze(arr, axis=0) for arr in features]
    features_index_num = [len(features) for _ in range(len(features))]
    fsn_node = torch.arange(fsn.shape[0])
    av_node = vision.shape[0] + audio.shape[0] 


    for nodes_x in features:
        for nodes_y in features:
            edge_type = torch.dist(nodes_x, nodes_y)
            edge_type_list.append(edge_type)

    edge_type_list_value = torch.tensor(edge_type_list, dtype=torch.float)#[104*104]
    edge_type_list_1 = util.data_normal(edge_type_list_value)
    edge_type_list_1 = edge_type_list_1.cpu()
    # edge_type_list_2 = edge_type_list_1 * -1 + 1
    edge_type_list_index = edge_type_list_1  < 0.2  #2543 删除距离小于0.2的长度
    index = np.where(edge_type_list_index)
    index_list = list(index[0])
    features_delete_index = [value // len(features) for value in index_list]
    for index in features_delete_index:
        features_index_num[index] -= 1


    edge_type_list_index = edge_type_list_1  > 0.6 #2543 删除距离大于0.8的长度
    index_2 = np.where(edge_type_list_index)
    index_list_2 = list(index_2[0])
    features_delete_index_2 = [value // len(features) for value in index_list_2]
    for index in features_delete_index_2:
        features_index_num[index] -= 1

    delete_node_idx = [
        id
        for id, value in enumerate(features_index_num)
        if value < (av_node // 10) and id >= fsn_len
    ]
    for index in sorted(delete_node_idx, reverse=True):
        features.pop(index)


    edge_type_list_fin = []
    for nodes_x in features:
        for nodes_y in features:
            edge_type = torch.dist(nodes_x, nodes_y)
            edge_type_list_fin.append(edge_type)

    node_index = torch.tensor(list(range(len(features))))
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index_fin = torch.stack((src, dst), 0).to(device) #[2,104*104]
    edge_type_list_fin = torch.tensor(edge_type_list_fin, dtype=torch.float)
    edge_type_list_fin = util.data_normal(edge_type_list_fin)
    nodes_features = torch.stack(features,dim=0) #[104,128,32] 
    print(nodes_features.size())

    return nodes_features, edge_index_fin, edge_type_list_fin

def create_two_edge_weight_graph_new_SimCos(text, vision, audio, fsn_len):
    features, edge_type_list = [], []
    fsn = torch.zeros(fsn_len, text.size(1), text.size(2)).cuda()
    # init.kaiming_uniform_(fsn, mode='fan_in', nonlinearity='relu')
    nodes = torch.cat((fsn, vision, audio), dim=0)
    features = np.split(nodes, nodes.size(0), axis=0)
    features = [torch.tensor(np.squeeze(arr, axis=0)).cuda() for arr in features]
    features_index_num = [len(features) for _ in range(len(features))]
    av_node = vision.shape[0] + audio.shape[0]

    for nodes_x in features:
        for nodes_y in features:
            # 计算余弦相似度   
            edge_type = torch.dist(nodes_x, nodes_y)
            edge_type_list.append(edge_type)


    edge_type_list = torch.tensor(edge_type_list, dtype=torch.float).cuda()

    edge_type_list_index = edge_type_list < 0.1 # 删除距离小于0.2的长度
    index = torch.where(edge_type_list_index)[0]
    index_list = index.tolist()
    features_delete_index = [value // len(features) for value in index_list]
    for index in features_delete_index:
        features_index_num[index] -= 1

    edge_type_list_index = edge_type_list > 0.9 # 删除距离大于0.6的长度
    index = torch.where(edge_type_list_index)[0]
    index_list = index.tolist()
    features_delete_index = [value // len(features) for value in index_list]
    for index in features_delete_index:
        features_index_num[index] -= 1

    delete_node_idx = [
        id
        for id, value in enumerate(features_index_num)
        if value < (av_node // 10) and id >= fsn_len
    ]
    for index in sorted(delete_node_idx, reverse=True):
        features.pop(index)

    edge_type_list_fin = []
    for nodes_x in features:
        for nodes_y in features:
            cos = torch.nn.functional.cosine_similarity(nodes_x.view(1, -1), nodes_y.view(1, -1))  
            edge_type_list_fin.append(cos.item())

    node_index = torch.tensor(list(range(len(features)))).cuda()
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index_fin = torch.stack((src, dst), 0).cuda()  # [2,104*104]
    edge_type_list_fin = torch.tensor(edge_type_list_fin, dtype=torch.float).cuda()
    nodes_features = torch.stack(features, dim=0).cuda()  # [104,128,32]
    print(nodes_features.size())
    
    return nodes_features, edge_index_fin, edge_type_list_fin


def create_two_edge_weight_graph_dist(vision, audio, fsn):
    features, edge_type_list = [], []
    # fsn = torch.randn(fsn_len, text.size(1), text.size(2)).cuda()
    fsns = np.split(fsn, fsn.size(0), axis=0)
    fsns_list = [torch.tensor(np.squeeze(arr, axis=0)).cuda() for arr in fsns]
    nodes = torch.cat((vision, audio), dim=0)
    features = np.split(nodes, nodes.size(0), axis=0)
    features_list = [torch.tensor(np.squeeze(arr, axis=0)).cuda() for arr in features]

    edge_index = []
    for index_x,nodes_x in enumerate(fsns_list):
        for index_y,nodes_y in enumerate(features_list):
            # edge_type = torch.dist(nodes_x, nodes_y)
            # edge_type_list.append(edge_type)

            # normalized_tensor_1 = nodes_x / nodes_x.norm(dim=-1, keepdim=True)
            # normalized_tensor_2 = nodes_y / nodes_y.norm(dim=-1, keepdim=True)
            # cos = (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1).mean()
            # cos = torch.mean(cos).item()
            # cos  = cos if cos > 0 else 0
            # edge_type_list.append(cos)
            index_y = index_y + 8
            cos = torch.nn.functional.cosine_similarity(nodes_x.view(1, -1), nodes_y.view(1, -1)).item()
            cos  = cos if cos > 0 else 0
            if cos > 0:
                edge_index.append([index_x,index_y])
                edge_type_list.append(cos)


    # node_index_1 = torch.tensor(list(range(len(features_list)))).cuda()
    # node_index_2 = torch.tensor(list(range(len(fsns_list)))).cuda()
    # src, dst = torch.meshgrid(node_index_1, node_index_2)
    # src, dst = src.flatten(), dst.flatten()
    # edge_index_fin = torch.stack((src, dst), 0).cuda()  # [2,104*104]
    edge_index_fin = torch.tensor(edge_index).t()
    edge_type_list = torch.tensor(edge_type_list, dtype=torch.float).cuda()
    nodes_features = torch.stack(fsns_list+features_list, dim=0).cuda()  # [104,128,32]
    # print(nodes_features.size())
    # print(edge_index_fin.size())
    
    return nodes_features, edge_index_fin, edge_type_list

def create_two_edge_weight_graph_newDelete(vision, audio, fsn_len):
    features, edge_type_list = [], []
    fsn = torch.zeros(fsn_len, audio.size(1),audio.size(2))
    # init.kaiming_uniform_(fsn, mode='fan_in', nonlinearity='relu')
    nodes = torch.cat((fsn, vision, audio), dim=0)
    features = np.split(nodes, nodes.size(0), axis=0)
    features = [np.squeeze(arr, axis=0) for arr in features]
    features_index_num = [len(features) for _ in range(len(features))]
    fsn_node = torch.arange(fsn.shape[0]) 
    vision_node = len(fsn_node) + torch.arange(vision.shape[0]) #54
    audio_node = torch.arange(audio.shape[0])  + len(vision_node) 
    av_node = vision.shape[0] + audio.shape[0]

    for i, nodes_x in enumerate(features):
        for j, nodes_y in enumerate(features):
                # edge_type_1 = util.atom_calculate_edge_weight(nodes_x.flatten(), nodes_y.flatten())
            edge_type = torch.dist(nodes_x, nodes_y)
            edge_type_list.append(edge_type)

    # node_index = torch.cat((fsn_node, vision_node, audio_node))
    # src, dst = torch.meshgrid(node_index, node_index)
    # src, dst = src.flatten(), dst.flatten()
    # edge_index = torch.stack((src, dst), 0).to(device) #[2,104*104]

    # edge_type_list = torch.cat(edge_type_list)  # []
    edge_type_list_value = torch.tensor(edge_type_list, dtype=torch.float)#[104*104]
    edge_type_list_1 = util.data_normal(edge_type_list_value)
    # edge_type_list_2 = edge_type_list_1 * -1 + 1
    edge_type_list_index = edge_type_list_1  < 0.4  #2543 删除距离小于0.2的长度
    index = np.where(edge_type_list_index)
    index_list = list(index[0])
    features_delete_index = [value // len(features) for value in index_list]
    for index in features_delete_index:
       features_index_num[index] -= 1


    edge_type_list_index = edge_type_list_1  > 0.6 #2543 删除距离大于0.8的长度
    index_2 = np.where(edge_type_list_index)
    index_list_2 = list(index_2[0])
    features_delete_index_2 = [value // len(features) for value in index_list_2]
    for index in features_delete_index_2:
       features_index_num[index] -= 1

    delete_node_idx = []
    for id,value in enumerate(features_index_num):
        if value < (av_node//10) and id >= fsn_len:
            delete_node_idx.append(id)
    
    for index in sorted(delete_node_idx, reverse=True):
        features.pop(index)
    
    
    edge_type_list_fin = []
    for i, nodes_x in enumerate(features):
        for j, nodes_y in enumerate(features):
                # edge_type_1 = util.atom_calculate_edge_weight(nodes_x.flatten(), nodes_y.flatten())
            edge_type = torch.dist(nodes_x, nodes_y)
            edge_type_list_fin.append(edge_type)

    node_index = torch.tensor([i for i in range(len(features))])
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index_fin = torch.stack((src, dst), 0).to(device) #[2,104*104]
    edge_type_list_fin = torch.tensor(edge_type_list_fin, dtype=torch.float)
    edge_type_list_fin = util.data_normal(edge_type_list_fin)
    # edge_index_fin = np.delete(edge_index,delete_index_list,axis=1) 
    # edge_type_list_fin = np.delete(edge_type_list_1,delete_index_list)
    # edge_index_fin = np.delete(edge_index,index,axis=1) 
    # edge_index_fin = np.delete(edge_index_fin,index_2,axis=1) 
    # edge_type_list_fin = np.delete(edge_type_list_1,index) 
    # edge_type_list_fin = np.delete(edge_type_list_fin,index_2) 

    nodes_features = torch.stack(features,dim=0) #[104,128,32] 

    return nodes_features, edge_index_fin, edge_type_list_fin

def create_three_edge_weight_graph_newDelete(text, vision, audio, fsn_len):
    features, edge_type_list = [], []
    fsn = torch.zeros(fsn_len, text.size(1),text.size(2)).cuda()
    # init.kaiming_uniform_(fsn, mode='fan_in', nonlinearity='relu')
    nodes = torch.cat((fsn, text, vision, audio), dim=0)
    features = np.split(nodes, nodes.size(0), axis=0)
    features = [np.squeeze(arr, axis=0) for arr in features]
    features_index_num = [len(features) for _ in range(len(features))]
    fsn_node = torch.arange(fsn.shape[0])
    vision_node = len(fsn_node) + torch.arange(vision.shape[0]) #54
    audio_node = torch.arange(audio.shape[0])  + len(vision_node)
    text_node = torch.arange(text.shape[0])  + len(audio_node)
    avt_node = vision.shape[0] + audio.shape[0] + text_node.shape[0]


    for nodes_x in features:
        for nodes_y in features:
            edge_type = torch.dist(nodes_x, nodes_y)
            edge_type_list.append(edge_type)

    edge_type_list_value = torch.tensor(edge_type_list, dtype=torch.float)#[104*104]
    edge_type_list_1 = util.data_normal(edge_type_list_value)
    edge_type_list_1 = edge_type_list_1.cpu()
    # edge_type_list_2 = edge_type_list_1 * -1 + 1
    edge_type_list_index = edge_type_list_1  < 0.4  #2543 删除距离小于0.2的长度
    index = np.where(edge_type_list_index)
    index_list = list(index[0])
    features_delete_index = [value // len(features) for value in index_list]
    for index in features_delete_index:
       features_index_num[index] -= 1


    edge_type_list_index = edge_type_list_1  > 0.6 #2543 删除距离大于0.8的长度
    index_2 = np.where(edge_type_list_index)
    index_list_2 = list(index_2[0])
    features_delete_index_2 = [value // len(features) for value in index_list_2]
    for index in features_delete_index_2:
       features_index_num[index] -= 1

    delete_node_idx = [
        id
        for id, value in enumerate(features_index_num)
        if value < (avt_node // 10) and id >= fsn_len
    ]
    for index in sorted(delete_node_idx, reverse=True):
        features.pop(index)


    edge_type_list_fin = []
    for nodes_x in features:
        for nodes_y in features:
            edge_type = torch.dist(nodes_x, nodes_y)
            edge_type_list_fin.append(edge_type)

    node_index = torch.tensor(list(range(len(features))))
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index_fin = torch.stack((src, dst), 0).to(device) #[2,104*104]
    edge_type_list_fin = torch.tensor(edge_type_list_fin, dtype=torch.float)
    edge_type_list_fin = util.data_normal(edge_type_list_fin)
    nodes_features = torch.stack(features,dim=0) #[104,128,32] 

    return nodes_features, edge_index_fin, edge_type_list_fin





def create_two_edge_weight_graph_no_Delete(vision, audio, fsn_len):
    features, edge_type_list = [], []
    fsn = torch.zeros(fsn_len, audio.size(1),audio.size(2))
    # init.kaiming_uniform_(fsn, mode='fan_in', nonlinearity='relu')
    nodes = torch.cat((fsn, vision, audio), dim=0)
    features = np.split(nodes, nodes.size(0), axis=0)
    features = [np.squeeze(arr, axis=0) for arr in features]
    fsn_node = torch.arange(fsn.shape[0]) 
    vision_node = len(fsn_node) + torch.arange(vision.shape[0]) #54
    audio_node = torch.arange(audio.shape[0])  + len(vision_node) 


    for i, nodes_x in enumerate(features):
        for j, nodes_y in enumerate(features):
                # edge_type_1 = util.atom_calculate_edge_weight(nodes_x.flatten(), nodes_y.flatten())
            edge_type = torch.dist(nodes_x, nodes_y)
            edge_type_list.append(edge_type)

    node_index = torch.cat((fsn_node, vision_node, audio_node))
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index_fin = torch.stack((src, dst), 0).to(device) #[2,104*104]
    edge_type_list_fin = torch.tensor(edge_type_list, dtype=torch.float)
    edge_type_list_fin = util.data_normal(edge_type_list_fin)
    nodes_features = torch.stack(features,dim=0) #[104,128,32] 

    return nodes_features, edge_index_fin, edge_type_list_fin

def create_two_edge_weight_graph_no_jhk(vision, audio):
    features, edge_type_list = [], []
    nodes = torch.cat((vision, audio), dim=0)
    features = np.split(nodes, nodes.size(0), axis=0)
    features = [np.squeeze(arr, axis=0) for arr in features]

    vision_node = torch.arange(vision.shape[0]) #54
    audio_node = torch.arange(audio.shape[0])  + len(vision_node) 


    for i, nodes_x in enumerate(features):
        for j, nodes_y in enumerate(features):
            if i == j:
                edge_type = 1
            else:
                # edge_type_1 = util.atom_calculate_edge_weight(nodes_x.flatten(), nodes_y.flatten())
                edge_type = torch.dist(nodes_x, nodes_y)
            edge_type_list.append(edge_type)

    node_index = torch.cat((vision_node, audio_node))
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index = torch.stack((src, dst), 0).to(device) #[2,104*104]

    # edge_type_list = torch.cat(edge_type_list)  # []
    edge_type_list = torch.tensor(edge_type_list, dtype=torch.float)#[104*104]
    edge_type_list_1 = util.data_normal(edge_type_list)
    edge_type_list_2 = edge_type_list_1 * -1 + 1
    edge_type_list_index = edge_type_list_2 < 0.3  #2543
    index = np.where(edge_type_list_index)
    edge_index_fin = np.delete(edge_index,index,axis=1) 
    edge_type_list_fin = np.delete(edge_type_list_2,index) 

    nodes_features = torch.stack(features,dim=0) #[104,128,32] 

    return nodes_features, edge_index_fin, edge_type_list_fin


def create_two_new_edge_weight_graph(vision, audio):
    features, edge_type_list = [], []
    nodes = torch.cat((vision, audio), dim=0)
    for _, v in enumerate(vision): #[128,50,32]
        features.append(v.to(device))

    for _, a in enumerate(audio): #[128,50,32]
        features.append(a.to(device))


    for i, nodes_x in enumerate(nodes):
        for j, nodes_y in enumerate(nodes):
            x = nodes_x.view(-1)
            y = nodes_y.view(-1)
            edge_type = util.atom_calculate_edge_weight(x, y)
            edge_type_list.append(edge_type)

    vision_node = torch.arange(vision.shape[0]) #128
    audio_node = torch.arange(audio.shape[0])  + len(vision_node) #128
    node_index = torch.cat((vision_node, audio_node))
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index = torch.stack((src, dst), 0).to(device) #[2,256*256]

    # edge_type_list = torch.cat(edge_type_list)  # []
    edge_type_list = torch.tensor(edge_type_list, dtype=torch.float)#[256*256]
    nodes_features = torch.stack(features) #[256,50,32] 

    return nodes_features, edge_index, edge_type_list


def create_three_edge_weight_graph(text, vision, audio):
    features, edge_type_list = [], []
    nodes = torch.cat((text, vision, audio), dim=0)

    for _, node in enumerate(nodes):
        features.append(node.to(device))

    vision_node = torch.arange(vision.shape[0]) #128
    audio_node = torch.arange(audio.shape[0])  + len(vision_node) #128
    text_node = torch.arange(text.shape[0])  + len(vision_node)+ len(audio_node) #128

    node_index = torch.cat((text_node, vision_node, audio_node))
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index = torch.stack((src, dst), 0).to(device) #[2,256*256]

    # 根据i，j定位二维数组的里值并删除
    for i, nodes_x in enumerate(nodes):
        for j, nodes_y in enumerate(nodes):
            edge_type = torch.dist(nodes_x, nodes_y)
            edge_type_list.append(edge_type)


    edge_type_list = torch.tensor(edge_type_list, dtype=torch.float)#[256*256]
    edge_type_list_1 = util.data_normal(edge_type_list)
    # edge_type_list_2 = edge_type_list_1 * -1 + 1
    edge_type_list_index = edge_type_list_1 < 0.5
    index = np.where(edge_type_list_index)
    edge_index_fin = np.delete(edge_index,index,axis=1) 
    edge_type_list_fin = np.delete(edge_type_list_1,index)
    nodes_features = torch.stack(features) #[256,50,32] 

    return nodes_features, edge_index_fin, edge_type_list_fin


def create_three_edge_graph(vision, audio, text):
    features, edge_type_list = [], []
    edge_type_offset = 0
    num_edges = text.shape[0] + audio.shape[0] + vision.shape[0]
    for i, v in enumerate(vision): #[50,4,32]
        features.append(v.to(device))
        edge_type_offset += i
        edge_type = torch.ones(num_edges, dtype=torch.long) * edge_type_offset  # Assign a unique edge type for each example
        edge_type_list.append(edge_type)

    for j, a in enumerate(audio): #[50,4,32]
        features.append(a.to(device))
        edge_type_offset += j + 1
        edge_type = torch.ones(num_edges, dtype=torch.long) * edge_type_offset  # Assign a unique edge type for each example
        edge_type_list.append(edge_type)

    for k, t in enumerate(text): #[50,4,32]
        features.append(t.to(device))
        edge_type_offset += k + 1
        edge_type = torch.ones(num_edges, dtype=torch.long) * edge_type_offset  # Assign a unique edge type for each example
        edge_type_list.append(edge_type)

    vision_node = torch.arange(vision.shape[0])
    audio_node = torch.arange(audio.shape[0])  + len(vision_node)
    text_node = torch.arange(text.shape[0])  + len(vision_node) + len(audio_node)
    node_index = torch.cat((vision_node, audio_node, text_node))
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index = torch.stack((src, dst), 0).to(device)
    edge_type_list = torch.cat(edge_type_list) #[5800]
    edge_type_list = torch.tensor(edge_type_list, dtype=torch.float)
    nodes_features = torch.stack(features) #[100,4,32] 

    return nodes_features, edge_index, edge_type_list

def create_three_weight_edge_graph(vision, audio, text):
    edge_type_offset = 0
    features, edge_type_list = [], []
    for i, v in enumerate(vision):  # [50,4,32]
        if not ifZero(v.to(device)):
            features.append(v.to(device))

    for j, a in enumerate(audio):  # [50,4,32]
        if not ifZero(a.to(device)):
            features.append(a.to(device))

    for k, t in enumerate(text):  # [50,4,32]
        if not ifZero(t.to(device)):
            features.append(t.to(device))

    edge_type_offset += vision.shape[0]
    for i, v in enumerate(vision):  # [50,4,32]
        if not ifZero(v.to(device)):
            edge_type = (
                torch.ones(len(features), dtype=torch.long) * edge_type_offset 
            )  # Assign a unique edge type for each example
            edge_type_list.append(edge_type)

    edge_type_offset += audio.shape[0]
    for j, a in enumerate(audio):  # [50,4,32]
        if not ifZero(a.to(device)):
            edge_type = (
                torch.ones(len(features), dtype=torch.long) * edge_type_offset 
            )  # Assign a unique edge type for each example
            edge_type_list.append(edge_type)

    edge_type_offset += text.shape[0]
    for k, t in enumerate(text):  # [50,4,32]
        if not ifZero(t.to(device)):
            edge_type = (
                torch.ones(len(features), dtype=torch.long) * edge_type_offset 
            )  # Assign a unique edge type for each example
            edge_type_list.append(edge_type)

    # vision_node = torch.arange(vision.shape[0])
    # audio_node = torch.arange(audio.shape[0]) + len(vision_node)
    # text_node = torch.arange(text.shape[0]) + len(vision_node) + len(audio_node)
    # node_index = torch.cat((vision_node, audio_node, text_node))
    node_index = torch.arange(len(features))
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index = torch.stack((src, dst), 0).to(device)
    edge_type_list = torch.cat(edge_type_list)  # [5800]
    edge_type_list = torch.tensor(edge_type_list, dtype=torch.float)
    nodes_features = torch.stack(features)  # [100,4,32]

    return nodes_features, edge_index, edge_type_list


def create_cross_modal_graph(x, x_in):
    features, edge_type_list = [], []
    edge_type_offset = 0
    num_edges = x.shape[0] + x_in.shape[0] 
    for i, v in enumerate(x): #[50,4,32]
        features.append(v.to(device))
        edge_type_offset += i
        edge_type = torch.ones(num_edges, dtype=torch.long) * edge_type_offset  # Assign a unique edge type for each example
        edge_type_list.append(edge_type)

    for j, a in enumerate(x_in): #[50,4,32]
        features.append(a.to(device))
        edge_type_offset += j + 1
        edge_type = torch.ones(num_edges, dtype=torch.long) * edge_type_offset  # Assign a unique edge type for each example
        edge_type_list.append(edge_type)


    x_node = torch.arange(x.shape[0])
    x_in_node = torch.arange(x_in.shape[0])  + len(x_node)
    node_index = torch.cat((x_node, x_in_node))
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index = torch.stack((src, dst), 0).to(device)
    edge_type_list = torch.cat(edge_type_list) #[5800]
    nodes_features = torch.stack(features) #[100,4,32] 

    return nodes_features, edge_index, edge_type_list



def create_three_diff_edge_graph(vision, audio, text):
    features, edge_type_list = [], []
    edge_type_offset = 0
    num_edges = text.shape[0] + audio.shape[0] + vision.shape[0]
    for i, v in enumerate(vision): #[50,4,32]
        features.append(v.to(device))
        edge_type = torch.ones(num_edges, dtype=torch.long) * 0.35  # Assign a unique edge type for each example
        edge_type_list.append(edge_type)

    for j, a in enumerate(audio): #[50,4,32]
        features.append(a.to(device))
        edge_type = torch.ones(num_edges, dtype=torch.long) * 0.05  # Assign a unique edge type for each example
        edge_type_list.append(edge_type)

    for k, t in enumerate(text): #[50,4,32]
        features.append(t.to(device))
        edge_type = torch.ones(num_edges, dtype=torch.long) * 0.6  # Assign a unique edge type for each example
        edge_type_list.append(edge_type)

    vision_node = torch.arange(vision.shape[0])
    audio_node = torch.arange(audio.shape[0])  + len(vision_node)
    text_node = torch.arange(text.shape[0])  + len(vision_node) + len(audio_node)
    node_index = torch.cat((vision_node, audio_node, text_node))
    src, dst = torch.meshgrid(node_index, node_index)
    src, dst = src.flatten(), dst.flatten()
    edge_index = torch.stack((src, dst), 0).to(device)
    edge_type_list = torch.cat(edge_type_list) #[5800]
    nodes_features = torch.stack(features) #[100,4,32] 

    return nodes_features, edge_index, edge_type_list

def create_graph(vision_audio, text, all_to_all=True):
    batch_x, batch_edge_index, batch_edge_types = [], [], []
    for va, t in zip(vision_audio, text):
        vision_audio_node_index = torch.tensor(va.shape[0]).long()
        text_node_index = torch.arange(t.shape[0]) + len(vision_audio_node_index)

        # Constructing node features and types
        node_features = torch.cat((va, t), 0)
        batch_x.append(node_features.to(device))

        # Constructing time aware dynamic graph edge index and types
        edge_index_list, edge_types = [], []
        edge_type_offset = 0
        # build uni-modal
        build_graph_uni_modal(
            vision_audio_node_index,
            edge_index_list,
            edge_types,
            edge_type_offset,
            all_to_all=all_to_all,
        )
        edge_type_offset += 2

        # text
        build_graph_uni_modal(
            text_node_index,
            edge_index_list,
            edge_types,
            edge_type_offset,
            all_to_all=all_to_all,
        )
        edge_type_offset += 2

        # build cross-modal
        # v_a - t
        build_graph_cross_modal(
            vision_audio_node_index,
            text_node_index,
            edge_index_list,
            edge_types,
            edge_type_offset,
        )
        edge_type_offset += 4

        try:
            edge_index = torch.cat(edge_index_list, 1)
        except Exception:
            import ipdb

            ipdb.set_trace()

        edge_types = torch.cat(edge_types)
        batch_edge_index.append(edge_index.to(device))
        batch_edge_types.append(edge_types.long().to(device))
    return batch_x, batch_edge_index, batch_edge_types


def empty_tensor_list(num=1):
    return [torch.empty(2, 0) for _ in range(num)]


def create_edge_type_uni_modal(seq, edge_types, edge_type_offset):
    if len(seq) > 0:
        edge = torch.stack((seq, seq))
        for i, _ in enumerate(list(range(seq.shape[0] * seq.shape[0]))):
            if edge.shape[0] != 0:
                edge_types.append(i + edge_type_offset + torch.zeros(edge.shape[0]))


def create_edge_type_cross_modal(seq1, seq2, edge_types, edge_type_offset):
    if len(seq1) > 0 and len(seq2) > 0:
        edge = torch.stack((seq1, seq2))
        for i, _ in enumerate(list(range(seq1.shape[0] * seq2.shape[0] * 2))):
            if edge.shape[0] != 0:
                edge_types.append(i + edge_type_offset + torch.zeros(edge.shape[0]))


def build_graph_uni_modal(
    seq, edge_index_list, edge_types, edge_type_offset, all_to_all=True
):
    if len(seq) > 0:
        current = torch.stack((seq, seq))
        if len(seq) > 1:
            if all_to_all:
                f_src, f_tgt = seq[:-1], seq[1:]
                f_src, f_tgt = torch.meshgrid(f_src, f_tgt)
                future = torch.stack((f_src.flatten(), f_tgt.flatten()))
                future_mask = (future[1, :] - future[0, :]) > 0
                future = future[:, future_mask]

                p_src, p_tgt = seq[1:], seq[:-1]
                p_src, p_tgt = torch.meshgrid(p_src, p_tgt)
                past = torch.stack((p_src.flatten(), p_tgt.flatten()))
                past_mask = (past[1, :] - past[0, :]) < 0
                past = past[:, past_mask]
            else:
                future = torch.stack((seq[:-1], seq[1:]))
                past = torch.stack((seq[1:], seq[:-1]))

        else:
            future = torch.empty(2, 0)
            past = torch.empty(2, 0)
    else:
        current, past, future = empty_tensor_list(3)
    for ei in enumerate([current, past, future]):
        if ei.shape[1] != 0:
            edge_index_list.append(ei)
            edge_types.append(edge_type_offset + torch.zeros(ei.shape[1]))


def build_graph_cross_modal(seq1, seq2, edge_index_list, edge_types, edge_type_offset):
    if len(seq1) > 0 and len(seq2) > 0:
        longer = seq1 if len(seq1) > len(seq2) else seq2
        shorter = seq1 if len(seq1) <= len(seq2) else seq2
        (
            ei_longer,
            ei_shorter,
        ) = build_graph(shorter, longer)
    else:
        (
            ei_longer,
            ei_shorter,
        ) = empty_tensor_list(2)

    for i, ei in enumerate(
        [
            ei_longer,
            ei_shorter,
        ]
    ):
        if ei.shape[1] != 0:
            edge_index_list.append(ei)
            edge_types.append(i // 3 + edge_type_offset + torch.zeros(ei.shape[1]))


def build_graph(seq, longest, window_factor=2):
    N, M = len(seq), len(longest)
    assert N <= M
    if M == 1:
        inds1 = torch.arange(1).reshape(1, 1)

    else:
        # To calculate the smallest window size and the corresponding stride, we have the following equation
        # (M - W) / S + 1 = N, where W is the window size, S is the stride size
        # so, W = M - (N - 1) * S
        # therefore, the corresponding S = M // (N - 1), and then we can compute the W
        if N > 1:
            stride = M // (N - 1)
            stride = math.floor((stride + 1) / window_factor)
            stride = max(stride, 2)
            window = M - (N - 1) * stride
            if window < 2:
                window = 2
                stride = (M - window) // (N - 1)
        else:
            stride = M
            window = M

        inds1 = torch.arange(window).repeat(len(seq), 1)
        offset = torch.arange(len(seq)).reshape(-1, 1).repeat(1, window) * stride
        inds1 = inds1 + offset

    nodes_within_view = longest[inds1]

    # build the edge_index for the current cross modality clique
    seq_t = seq.reshape(-1, 1).repeat(1, nodes_within_view.shape[1]).flatten()
    nodes_within_view_f = nodes_within_view.flatten()

    # bi-directional edges
    ei_current = torch.stack((seq_t, nodes_within_view_f))
    ei_current_reverse = torch.stack((nodes_within_view_f, seq_t))

    all_inds = torch.arange(M).repeat(len(seq), 1)

    # build the FUTURE edges for the current cross modality clique (from seq to longest),
    # the reverse order of which is the PAST edges from longest to seq
    ei_future = []
    future_cutoff, _ = torch.max(inds1, dim=1)
    future_node_mask = all_inds > future_cutoff.reshape(-1, 1)
    for node, f_mask in zip(seq, future_node_mask):
        tgt = longest[f_mask]
        src = node.repeat(len(tgt))
        ei_future_i = torch.stack((src, tgt))
        ei_future.append(ei_future_i)
    ei_future = torch.cat(ei_future, dim=-1)
    ei_future_reverse = torch.roll(ei_future, shifts=[1], dims=[0])
    # future_xm_edges.append(ei_future)
    # past_xm_edges.append(ei_future_reverse)

    # build the PAST edges for the current cross modality clique (from seq to longest),
    # the reverse order of which is the FUTURE edges from longest to seq
    ei_past = []
    past_cutoff, _ = torch.min(inds1, dim=1)
    past_node_mask = all_inds < past_cutoff.reshape(-1, 1)
    for node, p_mask in zip(seq, past_node_mask):
        tgt = longest[p_mask]
        src = node.repeat(len(tgt))
        ei_past_i = torch.stack((src, tgt))
        ei_past.append(ei_past_i)
    ei_past = torch.cat(ei_past, dim=-1)
    ei_past_reverse = torch.roll(ei_past, shifts=[1], dims=[0])
    # edge_types = ("O", "I", "O", "I", "O","I")
    # edge_times = ("C", "Cr", "P", "F", "F", "P")
    return (
        ei_current,
        ei_current_reverse,
        ei_past,
        ei_past_reverse,
        ei_future,
        ei_future_reverse,
    )



