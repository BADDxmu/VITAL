import numpy as np
import torch
import json
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, precision_score, f1_score, matthews_corrcoef, confusion_matrix
import argparse
import warnings
warnings.filterwarnings("ignore")
def parser():
    ap = argparse.ArgumentParser(description='VIATL for the recommendation dataset')
    ap.add_argument('--device', default='cuda:7')

    # For Training
    ap.add_argument('--epoch', type=int, default=50, help='Number of epochs. Default is 50.')
    ap.add_argument('--batch_size', type=int, default=16, help='Batch size. Default is 16.')
    ap.add_argument('--dropout_rate', type=int, default=0.2, help='Dropout Rate. Default is 0.2.')
    ap.add_argument('--num_workers', default=0, type=int)
    ap.add_argument('--dim', default=128, type=int)
    
    ap.add_argument('--neg_times', default=1)
    ap.add_argument('--train_test_split_dict_path', default='./datasets/VITAL/PePPI/train_test_split_set.json', type=str, help='Train/Test Split Json Path')
    ap.add_argument('--dir_feature_dict', default='./datasets/VITAL/PePPI/feature_dict/', type=str,help='Feature Dict Path')
    ap.add_argument('--dir_log', default='./logs/training_log.txt', type=str,help='Saved Log Path')
    ap.add_argument('--dir_ckpt', default='./ckpts/', type=str,help='Saved Checkpoint Path')

    ap.add_argument('--max_len_x', default=100, type=int)
    ap.add_argument('--max_len_y', default=100, type=int)

    ap.add_argument('--is_save_model', default=True, type=bool, help='Is save Model or Not')
    ap.add_argument('--is_save_log', default=False, type=bool, help='Is save Log or Not')

    # For Prediction
    ap.add_argument('--feature_dict_path', type=str, help='Path to the feature dictionary JSON file')
    ap.add_argument('--ckpt_path', type=str, help='Path to the model checkpoint file')
    ap.add_argument('--batch_input_csv', type=str, help='Path to CSV file for batch prediction (requires pair_id and feature_dict_path columns)')
    ap.add_argument('--output', type=str, default='prediction_result.json', help='Output file path for prediction results (default: prediction_result.json)')
    ap.add_argument('--ASM_output_path', type=str, default='./output/ASM/', help='Output file path for ASM matrices')
    ap.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = ap.parse_args()
    return args

def cls_scores(labels, preds):
    # Flatten the arrays to ensure compatibility with sklearn metrics
    labels_flat = labels.reshape(-1)
    preds_flat = np.round(preds).reshape(-1)
    # Calculate various metrics
    acc = accuracy_score(labels_flat, preds_flat)
    AUC = roc_auc_score(labels_flat, preds.reshape(-1))
    AUPR = average_precision_score(labels_flat, preds.reshape(-1))
    # AUC = 0
    # AUPR = 0
    recall = recall_score(labels_flat, preds_flat)
    precision = precision_score(labels_flat, preds_flat)
    f1 = f1_score(labels_flat, preds_flat, average='binary', pos_label=1)
    MCC = matthews_corrcoef(labels_flat, preds_flat)
    
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(labels_flat, preds_flat).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # Success rate
    success_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    # specificity = 0
    # success_rate = 0
    
    return acc, AUC, AUPR, precision, recall, f1, MCC, specificity, success_rate

def cls_scores_long(labels, preds):
    acc = accuracy_score(labels, np.round(preds))
    AUC = 0
    AUPR = 0
    recall = recall_score(labels, np.round(preds))
    precision = precision_score(labels, np.round(preds))
    f1 = f1_score(labels, np.round(preds), average = 'binary',pos_label=1)
    return acc, AUC, AUPR, recall, precision, f1

def reg_scores(labels, preds):
    # For regression, consider using regression metrics such as MSE or MAE instead of classification metrics.
    mse = np.mean((labels - preds)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(labels - preds))
    return rmse, mae

def BPR_Loss(pred, label):
    reverse_label = (~(label.bool())).long()
    pred_pos = pred*label
    pred_neg = pred*reverse_label
    score_diff = pred_pos.mean()-pred_neg.mean()
    loss = -torch.log(torch.sigmoid(score_diff))
    return loss
def load_checkpoint(filepath):
    ckpt = torch.load(filepath, map_location='cuda:4')
    model = ckpt['model']
    model.load_state_dict(ckpt['model_state_dict'])
    # for parameter in model.parameters():
    #     print(parameter.requires_grad)
    #     parameter.requires_grad=False
    # model.eval()
    return model

def load_full_and_seg_dataset(dir,times):
    with open(dir+'train/idx_pos', 'r') as file:
            pair_idxs = file.readlines()#[:100]
        # label = [id.split('\n')[0].split('\t')[1] for id in pair_idxs]
    with open(dir+f'train/idx_neg_{times}times', 'r') as file:
            pair_idxs_neg = file.readlines()#[:100]

    with open(dir+'test/idx_pos', 'r') as file:
            pair_idxs_test = file.readlines()#[:100]
        # label = [id.split('\n')[0].split('\t')[1] for id in pair_idxs]
    with open(dir+f'test/idx_neg_{times}times', 'r') as file:
            pair_idxs_neg_test = file.readlines()#[:100]

    pair_idxs = [id.strip() for id in pair_idxs]
    label = [1 for _ in pair_idxs]

    pair_idxs_neg = [id.strip() for id in pair_idxs_neg]
    label_neg = [0 for _ in pair_idxs_neg]

    pair_idxs_test = [id.strip() for id in pair_idxs_test]
    label_test = [1 for _ in pair_idxs_test]

    pair_idxs_neg_test = [id.strip() for id in pair_idxs_neg_test]
    label_neg_test = [0 for _ in pair_idxs_neg_test]

    pair_idxs.extend(pair_idxs_neg)
    label.extend(label_neg)

    pair_idxs_test.extend(pair_idxs_neg_test)
    label_test.extend(label_neg_test)
    print(dir,times)
    return pair_idxs, label, pair_idxs_test, label_test


def load_full_and_seg_dataset_swing(dir):
    with open(dir+'train/idx_pos', 'r') as file:
            pair_idxs = file.readlines()#[:100]
    with open(dir+f'train/idx_neg', 'r') as file:
            pair_idxs_neg = file.readlines()#[:100]

    # with open(dir+'test/idx_pos', 'r') as file:
    #         pair_idxs_test = file.readlines()#[:100]
    #     # label = [id.split('\n')[0].split('\t')[1] for id in pair_idxs]
    # with open(dir+f'test/idx_neg', 'r') as file:
    #         pair_idxs_neg_test = file.readlines()#[:100]

    folds = load_idxs('/data1/qwwang/PepPI/MILD/datasets/MILD/data_10mer_20250409/PePI/no_folds.json')
    pair_idxs_test = folds['test']['test_idx']
    label_test = folds['test']['test_label']

    pair_idxs = [id.strip() for id in pair_idxs]
    label = [1 for _ in pair_idxs]

    pair_idxs_neg = [id.strip() for id in pair_idxs_neg]
    label_neg = [0 for _ in pair_idxs_neg]

    # pair_idxs_test = [id.strip() for id in pair_idxs_test]
    # label_test = [1 for _ in pair_idxs_test]

    # pair_idxs_neg_test = [id.strip() for id in pair_idxs_neg_test]
    # label_neg_test = [0 for _ in pair_idxs_neg_test]

    pair_idxs.extend(pair_idxs_neg)
    label.extend(label_neg)

    # pair_idxs_test.extend(pair_idxs_neg_test)
    # label_test.extend(label_neg_test)
    print(dir)
    return pair_idxs, label, pair_idxs_test, label_test


def load_idxs(dir):
    with open(dir,'r') as f:
        folds = json.load(f)
    return folds

def random_split(X, Y, fold=5):
	skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=1)
	train_idx_list, test_idx_list = [], []
	for train_index, test_index in skf.split(X, Y):
		train_idx_list.append(train_index)
		test_idx_list.append(test_index)
	return train_idx_list, test_idx_list