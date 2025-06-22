import sys
import os
import argparse
import numpy as np

sys.path.insert(0, "D:\GitHubCode\MBGCN\CMU-MultimodalSDK-master")
from mmsdk import mmdatasdk as md

# Using BERT from https://github.com/shehzaadzd/pytorch-pretrained-BERT
# pip install pytorch-pretrained-bert
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from bert_utils import convert_examples_to_features


def myavg(intervals, features):
    return np.average(features, axis=0)


if __name__ == "__main__":
    DATA_PATH = "./CMU_MOSEI"
    CSD_PATH = os.path.join(DATA_PATH, "csd")
    TRAIN_PATH = os.path.join(DATA_PATH, "train")
    VAL_PATH = os.path.join(DATA_PATH, "val")
    TEST_PATH = os.path.join(DATA_PATH, "test")
    DATASET = md.cmu_mosei
    # DATASET = md.cmu_mosi

    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)

    try:
        os.mkdir(CSD_PATH)
    except OSError as error:
        print(error)

    try:
        os.mkdir(TRAIN_PATH)
    except OSError as error:
        print(error)

    try:
        os.mkdir(VAL_PATH)
    except OSError as error:
        print(error)

    try:
        os.mkdir(TEST_PATH)
    except OSError as error:
        print(error)

    # Downloading the dataset

    try:
        md.mmdataset(DATASET.highlevel, CSD_PATH)
    except RuntimeError:
        print("High-level features have been downloaded previously.")

    try:
        md.mmdataset(DATASET.raw, CSD_PATH)
    except RuntimeError:
        print("Raw data have been downloaded previously.")

    try:
        md.mmdataset(DATASET.labels, CSD_PATH)
    except RuntimeError:
        print("Labels have been downloaded previously.")

    # Loading the dataset
    # All fields are listed here:
    # https://github.com/A2Zadeh/CMU-MultimodalSDK/blob/master/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei.py
    # Label format [sentiment, happy, sad, anger, surprise, disgust, fear]

    visual_field = 'CMU_MOSEI_VisualFacet42'
    acoustic_field = 'CMU_MOSEI_COVAREP'
    text_field = 'CMU_MOSEI_TimestampedWords'
    label_field = 'CMU_MOSEI_Labels'
    # visual_field = "CMU_MOSI_Visual_Facet_42"
    # acoustic_field = "CMU_MOSI_COVAREP"
    # text_field = "CMU_MOSI_TimestampedWordVectors"
    # label_field = "CMU_MOSI_Opinion_Labels"

    features = [text_field, visual_field, acoustic_field]

    recipe = {feat: os.path.join(CSD_PATH, feat) + ".csd" for feat in features}
    dataset = md.mmdataset(recipe)
    # dataset.align(text_field, collapse_functions=[myavg])

    label_recipe = {label_field: os.path.join(CSD_PATH, label_field + ".csd")}
    dataset.add_computational_sequences(label_recipe, destination=None)
    # dataset.align(label_field)
    dataset.align(text_field, collapse_functions=[myavg])

    # Creating BERT features
    computational_sequences = dataset.computational_sequences

    # s = computational_sequences[text_field].data['03bSnISJMiM']['features'][:]
    # print(s)

    train, val, test = dataset.get_tensors(
        seq_len=50,
        non_sequences=[label_field],
        direction=False,
        folds=[
            DATASET.standard_folds.standard_train_fold,
            DATASET.standard_folds.standard_valid_fold,
            DATASET.standard_folds.standard_test_fold,
        ],
    )

    print("Creating BERT features...")
