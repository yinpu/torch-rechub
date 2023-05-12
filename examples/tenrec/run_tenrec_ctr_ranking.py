import sys

sys.path.append("../..")

import pandas as pd
import torch
import os
import random
import pickle
import numpy as np
from utils import *
from torch_rechub.models.ranking.afm import AFM
from sklearn.model_selection import train_test_split
from torch_rechub.trainers import CTRTrainer
from sklearn.preprocessing import LabelEncoder
from torch_rechub.basic.features import DenseFeature, SparseFeature
from tqdm import tqdm

def set_seed(seed, re=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    if re:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def check_files_exist(file_list):
    for file in file_list:
        if not os.path.isfile(file):
            return False
    return True

def sample_data(df):
    p_df = df[df.click.isin([1])]
    n_df = df[df.click.isin([0])]
    del df
    n_df = n_df.sample(n=len(p_df)*2)
    #df = p_df.append(n_df)
    df = pd.concat([p_df, n_df])
    del p_df, n_df
    df = df.sample(frac=1)
    return df

def get_tenrec_ctr_data(dataset_path, batch_size):
    file_prefix = os.path.splitext(dataset_path)[0] # 获取文件名前缀
    file_list = [file_prefix + ".train.tfrecord", 
                 file_prefix + ".train.index", 
                 file_prefix + ".valid.tfrecord",
                 file_prefix + ".valid.index", 
                 file_prefix + ".test.tfrecord",
                 file_prefix + ".test.index",
                 file_prefix + ".feat_nunique_dict.pkl"]
    if  not check_files_exist(file_list):
        df = pd.read_csv(dataset_path, 
                         usecols=["user_id", "item_id", "click", "video_category", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"])
        df['video_category'] = df['video_category'].astype(str)
        df = sample_data(df)
        sparse_features = ["user_id", "item_id", "video_category", "gender", "age", "hist_1", "hist_2",
                        "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]
        lbe = LabelEncoder()
        df['click'] = lbe.fit_transform(df['click'])
        feat_nunique_dict, feat_dtype = {}, {'click':'int'}
        for feat in tqdm(sparse_features):
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])
            feat_nunique_dict[feat] = df[feat].nunique()
            feat_dtype[feat] = 'int'
        train_df, test_df = train_test_split(df, test_size=0.1)
        train_df, valid_df = train_test_split(train_df, test_size=0.1111)
        dataframe_to_tfrecord(train_df, file_prefix + ".train.tfrecord", file_prefix + ".train.index", feat_dtype)
        dataframe_to_tfrecord(valid_df, file_prefix + ".valid.tfrecord", file_prefix + ".valid.index", feat_dtype)
        dataframe_to_tfrecord(test_df, file_prefix + ".test.tfrecord", file_prefix + ".test.index", feat_dtype)
        with open(file_prefix + ".feat_nunique_dict.pkl", "wb") as f:
            pickle.dump(feat_nunique_dict, f)
            
    with open(file_prefix + ".feat_nunique_dict.pkl", "rb") as f:
        feat_nunique_dict = pickle.load(f)
    train_dataloader = create_tfrecord_dataloader(file_prefix + ".train.tfrecord", file_prefix + ".train.index", 
                                                  batch_size, label_name='click')
    test_dataloader = create_tfrecord_dataloader(file_prefix + ".test.tfrecord", file_prefix + ".test.index", 
                                                  batch_size, label_name='click')
    valid_dataloader = create_tfrecord_dataloader(file_prefix + ".valid.tfrecord", file_prefix + ".valid.index", 
                                                  batch_size, label_name='click')
    sparse_feas = [SparseFeature(name=fea, vocab_size=fea_nunique, embed_dim=32, ) for fea, fea_nunique in feat_nunique_dict.items()]
    dense_feas = []
    return  dense_feas, sparse_feas, train_dataloader, test_dataloader, valid_dataloader
    
             
def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    set_seed(seed)
    
    dense_feas, sparse_feas, train_dataloader, val_dataloader, test_dataloader= get_tenrec_ctr_data(dataset_path, batch_size)
    if model_name == "afm":
        #embedding_size = 32
        model = AFM(linear_features=dense_feas+sparse_feas, afm_features=sparse_feas, attention_dim=8)

    ctr_trainer = CTRTrainer(model, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=3, device=device, model_path=save_dir)
    #scheduler_fn=torch.optim.lr_scheduler.StepLR,scheduler_params={"step_size": 2,"gamma": 0.8},
    ctr_trainer.fit(train_dataloader, val_dataloader)
    auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data/ctr_data_1M.csv")
    parser.add_argument('--model_name', default='afm')
    parser.add_argument('--epoch', type=int, default=20)  #100
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=131072)  #4096
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')  #cuda:0
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed)
    

