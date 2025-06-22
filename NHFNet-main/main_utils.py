import time
import math
import time

from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt_sne
import mpl_toolkits.mplot3d as p3d
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import pandas as pd
import seaborn as sns
from modules.dataset import *
from torchstat import stat

def get_data(args, dataset, split='train'):
    alignment = 'a' if args.aligned else 'na'
    data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data

def convert2class(out):
    if out < 0:
        return -1
    elif out > 0:
        return 1
    else:
        return 0

def convert2classnew(out):
    if out < 0:
        return -1
    elif out >= 0:
        return 1
    


def convert7class(out):
    return np.clip(np.rint(out), -3, 3)


def acc_2_f_score(y_pred, y_true):
    exclude_zero=False
    test_preds = y_pred
    test_truth = y_true
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])
    pred_cls = []
    true_cls = []
    for y_p, y_t in zip(y_pred, y_true):
        if y_t == 0:
            continue
        pred_cls.append(convert2class(y_p))
        true_cls.append(convert2class(y_t))

    cm = confusion_matrix(true_cls, pred_cls, labels=[-1, 1])
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    f_score = f1_score(true_cls, pred_cls, labels=[-1, 1], average="weighted")
    # f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')   
    # binary_truth = (test_truth[non_zeros] > 0)
    # binary_preds = (test_preds[non_zeros] > 0) 
    # acc = accuracy_score(binary_truth, binary_preds)

    return f_score, acc, cm

def acc_2_f_score_new(y_pred, y_true):
    pred_cls = []
    true_cls = []
    for y_p, y_t in zip(y_pred, y_true):
        pred_cls.append(convert2classnew(y_p))
        true_cls.append(convert2classnew(y_t))

    cm = confusion_matrix(true_cls, pred_cls, labels=[-1, 1])
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    f_score = f1_score(true_cls, pred_cls, labels=[-1, 1], average="weighted")

    return f_score, acc, cm


def acc_7(y_pred, y_true):
    pred_cls = []
    true_cls = []
    for y_p, y_t in zip(y_pred, y_true):
        pred_cls.append(convert7class(y_p))
        true_cls.append(convert7class(y_t))

    cm = confusion_matrix(true_cls, pred_cls, labels=[-3, -2, -1, 0, 1, 2, 3])
    acc = np.sum(np.diag(cm)) / np.sum(cm)

    return acc, cm


def train(
    get_X,
    log_interval,
    model,
    device,
    train_loader,
    optimizer,
    scheduler,
    loss_func,
    epoch,
    lr,
    verbose=True,
):
    # set model as training mode
    model.train()

    losses = []
    all_y_true = []
    all_y_pred = []
    all_features = []
    init_feature = []
    N_count = 0  # counting total trained sample in one epoch
    for batch_idx, sample in enumerate(train_loader):
        # distribute data to device
        # print(sample)
        # print(optimizer.param_groups[0]['lr'])
        start_time = time.time()
        X, n = get_X(device, sample)
        # macs, params = profile(model, inputs=(X,))
        # print(macs)
        y_true = sample["label"].to(device)  # .view(-1, )
        features, output = model(X)
        N_count += n
        optimizer.zero_grad()

        # print("output", output.shape)
        # print("y_true", y_true.shape) (128,1)
        loss = loss_func(output, y_true)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        all_features.append(features.detach().cpu().numpy())
        all_y_pred.append(output.detach().cpu().numpy())
        all_y_true.append(y_true.cpu().numpy())
        end_time = time.time()

        # show information
        if (batch_idx + 1) % log_interval == 0 and verbose:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.2f}".format(
                    epoch + 1,
                    N_count,
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    end_time - start_time
                )
            )

    all_y_pred = np.concatenate(all_y_pred)
    all_features = np.concatenate(all_features)
    all_y_true = np.concatenate(all_y_true)
    # init_feature = np.concatenate(init_feature)
    # init_feature = np.mean(init_feature, axis=1)
    # # 计算 t-SNE 嵌入

    # color_map = {-3:"#194f97",-2:"#76da91",-1: '#bd6b08', 1: '#c82d31', 2:"#625ba1",3:"#00686b"}
    # colors = [color_map[3 if label > 2 else (2 if label > 1 else (1 if label>0 else (-1 if label>-1 else (-2 if label > -2 else -3))))] for label in all_y_pred]

    # tsne = TSNE(n_components=2, random_state=42)
    # X_tsne = tsne.fit_transform(init_feature)
    # fig, ax = plt_sne.subplots(figsize=(10, 10))
    # scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors)
    # plt_sne.savefig(os.path.join("T-SEN",'tsne_2D_epoch_%s_%s.png') % (str(lr) , str(epoch+1)) )
    # plt_sne.close()


    # # 绘制3d 图
    #  # 调整画布宽高比为2:1
    # tsne = TSNE(n_components=3, random_state=42)
    # X_tsne = tsne.fit_transform(all_features)
    # fig = plt_sne.figure(figsize=(12, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=colors, marker='^')

    # plt_sne.savefig(os.path.join("T-SEN",'tsne_3D_epoch_%s_%s.png') % (str(lr) , str(epoch+1)) )
    # plt_sne.close()

    # print(all_y_pred.shape)
    # print(all_y_true.shape)
    acc7, _ = acc_7(all_y_pred, all_y_true)
    f1, acc2, _ = acc_2_f_score(all_y_pred, all_y_true)
    corr, _ = pearsonr(all_y_pred.squeeze(), all_y_true.squeeze())

    scores = [acc7 * 100, acc2 * 100, f1 * 100, corr * 100]
    print(
        "Train Epoch: {} Acc 7: {:.2f}%; Acc 2: {:.2f}%; F1 score: {:.2f}%; Corr: {:.2f}%\n".format(
            epoch + 1, *scores
        )
    )

    return np.mean(losses), scores


def validation(get_X, model, device, loss_func, val_loader, epoch, lr, print_cm=False):
    # set model as testing mode
    model.eval()

    test_loss = []
    all_y_true = []
    all_y_pred = []
    all_features= []
    init_feature = []
    samples = 0

    with torch.no_grad():
        for sample in val_loader:
            # distribute data to device
            init_feature.append(sample['bert50'].float())
            X, _ = get_X(device, sample)
            y_true = sample["label"].to(device)
            features, output = model(X)
            loss = loss_func(output, y_true)
            test_loss.append(loss.item())  # sum up batch loss
            all_y_true.append(y_true.cpu().numpy())
            all_y_pred.append(output.cpu().numpy())
            all_features.append(features.detach().cpu().numpy())

            samples += len(y_true)

    all_y_pred = np.concatenate(all_y_pred)
    all_y_true = np.concatenate(all_y_true)
    all_features = np.concatenate(all_features)
    # init_feature = np.concatenate(init_feature)
    # init_feature = np.mean(init_feature, axis=1)
    # color_map = {-3:"#194f97",-2:"#76da91",-1: '#bd6b08', 1: '#c82d31', 2:"#625ba1",3:"#00686b", 0 :"#E7DAD2"}
    # colors = [color_map[3 if label > 2 else (2 if label > 1 else (1 if label>0 else (0 if label==0 else (-1 if label>-1 else (-2 if label > -2 else -3)))))] for label in all_y_pred]
    # tsne = TSNE(n_components=2, random_state=42)
    # X_tsne = tsne.fit_transform(all_features)
    # fig, ax = plt_sne.subplots(figsize=(10, 10))
    # scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors)
    # plt_sne.savefig(os.path.join("T-SEN",'tsne_2D_epoch_%s_%s.png') % (str(lr) , str(epoch+1)) )
    # plt_sne.close()

    acc7, cm7 = acc_7(all_y_pred, all_y_true)
    if print_cm:
        print("7 class sentiment confusion matrix")
        print(cm7)
    corr, _ = pearsonr(all_y_pred.squeeze(), all_y_true.squeeze())
    f1, acc2, cm2 = acc_2_f_score(all_y_pred, all_y_true)
    f1_new, acc2_new, cm2 = acc_2_f_score_new(all_y_pred, all_y_true)
    
    test_loss = np.mean(test_loss) 
    # test_score = [acc7 * 100, acc2 * 100, acc2_new * 100, f1 * 100, f1_new * 100, corr * 100, test_loss * 100]
    test_score = [acc7 * 100, acc2 * 100, f1 * 100, corr * 100, test_loss * 100]

    # print info
    print(
        "\nTest set ({:d} samples): Average MAE loss: {:.4f}".format(samples, test_loss)
    )
    # print(
    #     "Acc 7: {:.2f}%; Acc 2: {:.2f}%; Acc 2_new: {:.2f}%;F1 score: {:.2f}%; F1 score_new: {:.2f}%;Corr: {:.2f}%\n".format(
    #         *test_score
    #     )
    # )
    print(
        "Acc 7: {:.2f}%; Acc 2: {:.2f}%;F1 score: {:.2f}% ;Corr: {:.2f}%\n".format(
            *test_score
        )
    )

    return test_loss, test_score


# def save_model(args, model):
#     torch.save(model, f'../pre_trained_models/{args.model}.pt')


# def load_model(args):
#     model = torch.load(f'../pre_trained_models/{args.model}.pt')
#     return model


def cossim(x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)


def atom_calculate_edge_weight(x, y):
        f = cossim(x, y)
        if f >1 and f <1.05:
            f = 1
        elif f< -1 and f>-1.05:
            f = -1
        elif f>=1.05 or f<=-1.05:
            f = f
        return  1-math.acos(f) / math.pi



def data_normal(arr):
    d_min = arr.min()
    if d_min < 0:
        arr += torch.abs(d_min)
        d_min = arr.min()
    d_max = arr.max()
    dst = d_max - d_min
    norm_data = (arr - d_min).true_divide(dst)
    return norm_data

def data_z_score_normal(arr):
    mean = torch.mean(arr)
    std_dev = torch.std(arr)

    #  对张量进行Z-score归一化
    normalized_x = (arr - mean) / std_dev
    return normalized_x