import sys
import os
import argparse
import copy
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


def bert_features(model, tokenizer, data, batch_size=1):
    in_features = convert_examples_to_features(data, seq_length=50, tokenizer=tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in in_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in in_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()

    bert = []
    for input_ids, input_mask, example_indices in tqdm(eval_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        all_encoder_layers, _ = model(
            input_ids, token_type_ids=None, attention_mask=input_mask
        )
        bert.append(all_encoder_layers[-1].detach().cpu().numpy())

    return np.concatenate(bert, axis=0)


def myavg(intervals, features):
    return np.average(features, axis=0)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", type=str, help="dataset directory", default="NHFNet-main/CMU_MOSI"
    )
    parser.add_argument(
        "--align", action="store_true", default=False, help="If Aligned"
    )
    args = parser.parse_args()

    DATA_PATH = args.datadir
    CSD_PATH = os.path.join(DATA_PATH, "csd")
    TRAIN_PATH = os.path.join(DATA_PATH, "train_un")
    VAL_PATH = os.path.join(DATA_PATH, "valNew")
    TEST_PATH = os.path.join(DATA_PATH, "testNew_un")
    # DATASET = md.cmu_mosei
    DATASET = md.cmu_mosi

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

    # try:
    #     os.mkdir(VAL_PATH)
    # except OSError as error:
    #     print(error)

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

    # visual_field = 'CMU_MOSEI_VisualFacet42'
    # acoustic_field = 'CMU_MOSEI_COVAREP'
    # text_field = 'CMU_MOSEI_TimestampedWords'
    # label_field = 'CMU_MOSEI_Labels'
    visual_field = "CMU_MOSI_Visual_Facet_42"
    acoustic_field = "CMU_MOSI_COVAREP"
    text_field = 'CMU_MOSI_TimestampedWords'
    label_field = "CMU_MOSI_Opinion_Labels"

    features = [text_field, visual_field, acoustic_field]

    recipe = {feat: os.path.join(CSD_PATH, feat) + ".csd" for feat in features}
    dataset = md.mmdataset(recipe)
    dataset_ori = md.mmdataset(recipe)
    # dataset.align(text_field, collapse_functions=[myavg])

    label_recipe = {label_field: os.path.join(CSD_PATH, label_field + ".csd")}
    dataset.add_computational_sequences(label_recipe, destination=None)
    dataset_ori.add_computational_sequences(label_recipe, destination=None)
  
    dataset.align(label_field)

    # Creating BERT features
    print("Creating BERT features...")
    data = dataset.computational_sequences
    train_segments = []
    # valid_segments = []
    test_segments = []

    for key in data[features[0]].keys():
        if key in data[features[1]].keys() and key in data[features[2]].keys():
            video_file_name = key.split("[")[0]
            sentence = data[features[0]][key]["features"].T.astype(str)
            sentence = " ".join(list(sentence[0]))
            if video_file_name in DATASET.standard_folds.standard_train_fold:
                train_segments.append(sentence)
            # elif video_file_name in DATASET.standard_folds.standard_valid_fold:
            #     valid_segments.append(sentence)
            elif video_file_name in DATASET.standard_folds.standard_test_fold:
                test_segments.append(sentence)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    model = BertModel.from_pretrained("bert-base-uncased")
    model.to(device)
    # print(get_n_params(model))

    train_bert = bert_features(model, tokenizer, train_segments)
    # valid_bert = bert_features(model, tokenizer, valid_segments)
    test_bert = bert_features(model, tokenizer, test_segments)

    np.save(os.path.join(TRAIN_PATH, "bert50.npy"), train_bert)
    # np.save(os.path.join(VAL_PATH, "bert50.npy"), valid_bert)
    np.save(os.path.join(TEST_PATH, "bert50.npy"), test_bert)
    
    # print("BERT features saved ", train_bert.shape, valid_bert.shape, test_bert.shape)

    # Train/dev/test split for non BERT features and labels
    if args.align:
        print("正在ALign")
        train, test = dataset.get_tensors(
            seq_len=50,
            non_sequences=[label_field],
            direction=False,
            folds=[
                DATASET.standard_folds.standard_train_fold,
                # DATASET.standard_folds.standard_valid_fold,
                DATASET.standard_folds.standard_test_fold,
            ],
        )
    else:
        print("非对齐")
        train50,  test50 = dataset.get_tensors(
            seq_len=50,
            non_sequences=[label_field],
            direction=False,
            folds=[
                DATASET.standard_folds.standard_train_fold,
                # DATASET.standard_folds.standard_valid_fold,
                DATASET.standard_folds.standard_test_fold,
            ],
        )
        train375, test375 = dataset.get_tensors(
            seq_len=375,
            non_sequences=[label_field],
            direction=False,
            folds=[
                DATASET.standard_folds.standard_train_fold,
                # DATASET.standard_folds.standard_valid_fold,
                DATASET.standard_folds.standard_test_fold,
            ],
        )

        train500, test500 = dataset.get_tensors(
            seq_len=425,
            non_sequences=[label_field],
            direction=False,
            folds=[
                DATASET.standard_folds.standard_train_fold,
                # DATASET.standard_folds.standard_valid_fold,
                DATASET.standard_folds.standard_test_fold,
            ],
        )

    print("Split: label field, visual field, acoustic field")
    if args.align:
        print(
            "Train:",
            train[text_field].shape,
            train[visual_field].shape,
            train[acoustic_field].shape,
        )
        print(
            "Test:",
            test[text_field].shape,
            test[visual_field].shape,
            test[acoustic_field].shape,
        )

    else:
        print(
            "Train:",
            train50[text_field].shape,
            train500[visual_field].shape,
            train375[acoustic_field].shape,
        )
        print(
            "Test:",
            test50[text_field].shape,
            test500[visual_field].shape,
            test375[acoustic_field].shape,
        )

    print("Saving features...")
    if args.align:
        for split, path in zip([train,  test], [TRAIN_PATH, TEST_PATH]):
            for f, n in zip(
                [visual_field, acoustic_field, label_field, text_field],
                ["visual", "audio", "label"],
            ):
                np.save(os.path.join(path, n + "50.npy"), split[f])

    else:

        for split, path in zip(
            [train50, test50], [TRAIN_PATH,  TEST_PATH]
        ):
            for f, n in zip([label_field], ["label"]):
                np.save(os.path.join(path, n + "50.npy"), split[f])


        for split, path in zip(
            [train375, test375], [TRAIN_PATH,  TEST_PATH]
        ):
            for f, n in zip([acoustic_field], ["audio"]):
                np.save(os.path.join(path, n + "375.npy"), split[f])

        
        for split, path in zip(
            [train500, test500], [TRAIN_PATH,  TEST_PATH]
        ):
            for f, n in zip([visual_field], ["visual"]):
                np.save(os.path.join(path, n + "425.npy"), split[f])


