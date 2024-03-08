import sys, json, os, argparse, time
from operator import itemgetter
import shutil
import os.path as osp
from datetime import datetime
import operator
from tqdm import trange
import numpy as np
import torch
import torch.nn.functional as F
from models.get_model import get_arch
from utils.get_loaders import get_train_val_cls_loaders, modify_dataset, modify_loader, get_combo_loader

# from utils.evaluation import evaluate_multi_cls
from utils.evaluation_1 import evaluate_multi_cls
from utils.model_saving_loading import save_model, str2bool, load_model
from utils.reproducibility import set_seeds

from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple
import torchvision.transforms as tr
import torchvision.transforms as transf
import torchvision

from torch.utils.data import DataLoader


# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='bit_resnext50_1', help='architecture')
parser.add_argument('--resume_path', type=str,  help='Path to saved model')
parser.add_argument('--n_classes', type=int, default=12, help='binary disease detection (1) or multi-class (5)')


args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
model_name = args.model_name
n_classes = args.n_classes

#train_data = torchvision.datasets.ImageFolder(root='dataset/data_3split/train/')
#val_data = torchvision.datasets.ImageFolder(root='dataset/data_3split/val/')
#val_data_isic = torchvision.datasets.ImageFolder(root='dataset_ham_5classes/')



model_name = args.model_name
print('* Instantiating a {} model'.format(model_name))
model, mean, std = get_arch(model_name, n_classes=n_classes)
model, stats = load_model(model, args.resume_path, device='cpu')

# model, stats, optimizer_state_dict = load_model(model, args.resume_path, device=device)


# Augmentation of dataset

train_transform = tr.Compose([
    transf.Resize((100,100)),
    transf.ToTensor(),
    transf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = tr.Compose([
    transf.Resize((100,100)),
    transf.ToTensor(),
    transf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_data = torchvision.datasets.ImageFolder(root='dataset/data_3split/train/', transform=train_transform)
print("train_data:",train_data.class_to_idx)
val_data = torchvision.datasets.ImageFolder(root='dataset/data_3split/val/', transform= val_transform)
print("val_data:",val_data.class_to_idx)
test_data = torchvision.datasets.ImageFolder(root='dataset/data_3split/test/', transform= val_transform)
print("test_data:",test_data.class_to_idx)

# Define a custom target transform function
def remap_labels(label):
    # Example remapping
    if label == 3:
        return 6 #convert melanoma label from 3 to 6
    elif label == 4:
        return 7 #convert nevus from 4 to 7
    else:
        return 0
target_transform = remap_labels
val_data_isic = torchvision.datasets.ImageFolder(root='exp_5class/dataset_ham_5classes/', transform= val_transform, target_transform=target_transform)
print("val_data_isic:",val_data_isic.class_to_idx)






train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=128, shuffle=True )
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=128, shuffle=True )
test_loader_isic = torch.utils.data.DataLoader(dataset=val_data_isic, batch_size=128, shuffle=True )

'''for img,lbl in test_loader_isic:
    print(lbl)
    break



quit()'''
model = model.to(device) 



def inference(loader, model):
    probs_all, preds_all, labels_all = [], [], []
    for i_batch, batch in enumerate(loader):
        inputs, labels = batch[0].to(device), batch[1].squeeze().to(device)
        logits = model(inputs)
        probs = logits.softmax(dim=1)
        preds = np.argmax(probs.detach().cpu().numpy(), axis=1)
        probs_all.extend(probs.detach().cpu().numpy())
        preds_all.extend(preds)
        labels_all.extend(labels.cpu().numpy())
    return np.stack(preds_all), np.stack(probs_all), np.stack(labels_all), 0

with torch.no_grad():
    tr_preds, tr_probs, tr_labels, tr_loss =inference(train_loader, model)
    vl_preds, vl_probs, vl_labels, vl_loss = inference(val_loader, model)
    ts_preds, ts_probs, ts_labels, ts_loss = inference(test_loader, model)
    ts_preds_isic, ts_probs_isic, ts_labels_isic, ts_loss_isic = inference(test_loader_isic, model)


tr_auc, tr_k, tr_mcc, tr_f1, tr_gm, tr_bacc, tr_auc_all, tr_f1_all, tr_acc_all = evaluate_multi_cls(tr_labels, tr_preds, tr_probs)
vl_auc, vl_k, vl_mcc, vl_f1, vl_gm, vl_bacc, vl_auc_all, vl_f1_all, vl_acc_all = evaluate_multi_cls(vl_labels, vl_preds, vl_probs)
ts_auc, ts_k, ts_mcc, ts_f1, ts_gm, ts_bacc, ts_auc_all, ts_f1_all, ts_acc_all = evaluate_multi_cls(ts_labels, ts_preds, ts_probs)  
ts_auc_isic, ts_k_isic, ts_mcc_isic, ts_f1_isic, ts_gm_isic, ts_bacc_isic, ts_auc_all_isic, ts_f1_all_isic, ts_acc_all_isic = evaluate_multi_cls(ts_labels_isic, ts_preds_isic, ts_probs_isic) 


print('Train||Val Loss: {:.4f}||{:.4f} - K: {:.2f}||{:.2f} - mAUC: {:.2f}||{:.2f} - MCC: {:.2f}||{:.2f} - BACC: {:.2f}||{:.2f}- F1-score: {:.2f}||{:.2f} - GM: {:.2f}||{:.2f}'.format(
    tr_loss, vl_loss, 100 * tr_k, 100 * vl_k, 100 * tr_auc, 100 * vl_auc, 100 * tr_mcc, 100 * vl_mcc, 100 * tr_bacc, 100 * vl_bacc, 100*tr_f1,100*vl_f1, 100*tr_gm,100*vl_gm))
print('Train||Test Loss: {:.4f}||{:.4f} - K: {:.2f}||{:.2f} - mAUC: {:.2f}||{:.2f} - MCC: {:.2f}||{:.2f} - BACC: {:.2f}||{:.2f}- F1-score: {:.2f}||{:.2f} - GM: {:.2f}||{:.2f}'.format(
    tr_loss, ts_loss, 100 * tr_k, 100 * ts_k, 100 * tr_auc, 100 * ts_auc, 100 * tr_mcc, 100 * ts_mcc, 100 * tr_bacc, 100 * ts_bacc, 100*tr_f1,100*ts_f1, 100*tr_gm,100*ts_gm))
print('Train||Test Loss ISIC: {:.4f}||{:.4f} - K: {:.2f}||{:.2f} - mAUC: {:.2f}||{:.2f} - MCC: {:.2f}||{:.2f} - BACC: {:.2f}||{:.2f}- F1-score: {:.2f}||{:.2f} - GM: {:.2f}||{:.2f}'.format(
    tr_loss, ts_loss_isic, 100 * tr_k, 100 * ts_k_isic, 100 * tr_auc, 100 * ts_auc_isic, 100 * tr_mcc, 100 * ts_mcc_isic, 100 * tr_bacc, 100 * ts_bacc_isic, 100*tr_f1,100*ts_f1_isic, 100*tr_gm,100*ts_gm_isic))



import csv
# Define column names
fields = ["path", "tr_auc", "tr_k", "tr_mcc", "tr_f1", "tr_gm", "tr_bacc", "tr_auc_all", "tr_f1_all", "tr_acc_all", "vl_auc", "vl_k", "vl_mcc", "vl_f1", "vl_gm", "vl_bacc", "vl_auc_all", "vl_f1_all", "vl_acc_all", "ts_auc", "ts_k", "ts_mcc", "ts_f1", "ts_gm", "ts_bacc", "ts_auc_all", "ts_f1_all", "ts_acc_all", "ts_auc_isic", "ts_k_isic", "ts_mcc_isic", "ts_f1_isic", "ts_gm_isic", "ts_bacc_isic", "ts_auc_all_isic", "ts_f1_all_isic", "ts_acc_all_isic"  ]
data_values = [str(args.resume_path), tr_auc, tr_k, tr_mcc, tr_f1, tr_gm, tr_bacc, tr_auc_all, tr_f1_all, tr_acc_all, vl_auc, vl_k, vl_mcc, vl_f1, vl_gm, vl_bacc, vl_auc_all, vl_f1_all, vl_acc_all, ts_auc, ts_k, ts_mcc, ts_f1, ts_gm, ts_bacc, ts_auc_all, ts_f1_all, ts_acc_all, ts_auc_isic, ts_k_isic, ts_mcc_isic, ts_f1_isic, ts_gm_isic, ts_bacc_isic, ts_auc_all_isic, ts_f1_all_isic, ts_acc_all_isic  ]

# Specify the CSV file name
csv_filename = "results.csv"

row_values = []
# Write data to CSV file
with open(csv_filename, 'a', newline='\n') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write column headers
    #writer.writerow(fields)
    
    # Write data rows
    for i in range(len(data_values)):
        row_values.append(str(data_values[i]))
    #print(row_values)
    writer.writerow(row_values)

print("CSV file saved successfully.")
