import sys
import os
import argparse
import numpy as np
sys.path.insert(0, 'E:\GitHub\CMU-MultimodalSDK-master')
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
        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
        bert.append(all_encoder_layers[-1].detach().cpu().numpy())

    return np.concatenate(bert, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='dataset directory', default='CMU_MOSEI')
    parser.add_argument('--align', action='store_true', default=False, help='If Aligned')
    args = parser.parse_args()

    DATA_PATH = args.datadir
    CSD_PATH = os.path.join(DATA_PATH, 'csd')
    TRAIN_PATH = os.path.join(DATA_PATH, 'train')
    VAL_PATH = os.path.join(DATA_PATH, 'val')
    TEST_PATH = os.path.join(DATA_PATH, 'test')
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

    # visual_field = 'CMU_MOSEI_VisualFacet42'
    # acoustic_field = 'CMU_MOSEI_COVAREP'
    # text_field = 'CMU_MOSEI_TimestampedWords'
    # label_field = 'CMU_MOSEI_Labels'
    visual_field = 'CMU_MOSI_Visual_Facet_42'
    acoustic_field = 'CMU_MOSI_COVAREP'
    text_field = 'CMU_MOSI_TimestampedWords'
    label_field = 'CMU_MOSI_Opinion_Labels'

    features = [
        text_field,
        visual_field,
        acoustic_field
    ]

    recipe = {feat: os.path.join(CSD_PATH, feat) + '.csd' for feat in features}
    dataset = md.mmdataset(recipe)

    label_recipe = {label_field: os.path.join(CSD_PATH, label_field + '.csd')}
    dataset.add_computational_sequences(label_recipe, destination=None)
    if args.align:
        print("align!!!!!!!!!!!")
        dataset.align(label_field)
    else:
        print("unalign!!!!!!!!!!!")

    # Creating BERT features
    print("Creating BERT features...")
    data = dataset.computational_sequences
    train_segments = []
    valid_segments = []
    test_segments = []

    for key in data[features[0]].keys():
        if key in data[features[1]].keys() and key in data[features[2]].keys():
            video_file_name = key.split("[")[0]
            if args.align:
                sentence = data[features[0]][key]['features'].T.astype(str)
                sentence = ' '.join(list(sentence[0]))
            else:
                text_list = []
                for s in list(dataset[text_field][key]['features']):
                    text_list.append(list(s))
                sentence = np.array(text_list).T.astype(str)
                sentence = ' '.join(list(sentence[0]))
            if video_file_name in DATASET.standard_folds.standard_train_fold:
                train_segments.append(sentence)
            elif video_file_name in DATASET.standard_folds.standard_valid_fold:
                valid_segments.append(sentence)
            elif video_file_name in DATASET.standard_folds.standard_test_fold:
                test_segments.append(sentence)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)

    train_bert = bert_features(model, tokenizer, train_segments)
    valid_bert = bert_features(model, tokenizer, valid_segments)
    test_bert = bert_features(model, tokenizer, test_segments)

    np.save(os.path.join(TRAIN_PATH, "bert50.npy"), train_bert)
    np.save(os.path.join(VAL_PATH, "bert50.npy"), valid_bert)
    np.save(os.path.join(TEST_PATH, "bert50.npy"), test_bert)
    #
    # print("BERT features saved ", train_bert.shape, valid_bert.shape, test_bert.shape)

    # Train/dev/test split for non BERT features and labels
    train, val, test = dataset.get_tensors(seq_len=50, non_sequences=[label_field], direction=False,
                                           folds=[DATASET.standard_folds.standard_train_fold,
                                                  DATASET.standard_folds.standard_valid_fold,
                                                  DATASET.standard_folds.standard_test_fold])

    print("Split: label field, visual field, acoustic field")
    print("Train:", train[label_field].shape, train[visual_field].shape, train[acoustic_field].shape)
    print("Val:", val[label_field].shape, val[visual_field].shape, val[acoustic_field].shape)
    print("Test:", test[label_field].shape, test[visual_field].shape, test[acoustic_field].shape)

    print("Saving features...")
    for split, path in zip([train, val, test], [TRAIN_PATH, VAL_PATH, TEST_PATH]):
        for f, n in zip([visual_field, acoustic_field, label_field], ['visual', 'audio', 'label']):
            label = []
            if n == 'label' and not args.align:
                for s in split[f]:
                    for k in s[:]:
                        label.append(list(k))
                label1 = np.array(label)
                label1 = label1.reshape(label1.shape[0], label1.shape[1], 1)
                np.save(os.path.join(path, n + "50.npy"), label1)
                print(label1.shape)
                print(label1)
                print(100 * '-')
            else:
                if n == 'label':
                    print(split[f].shape)
                    print(split[f])
                    print(100*'-')
                np.save(os.path.join(path, n + "50.npy"), split[f])
