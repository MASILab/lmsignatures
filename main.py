"""Entry point"""

import os
import numpy as np
import math
import pandas as pd
from pathlib import Path
import argparse
import random
import gc
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate # https://github.com/pytorch/pytorch/issues/11372
from monai.transforms import ToTensor

from utils import seed_everything, load_config, EarlyStopper, ConstantLRSchedule, WarmupConstantSchedule, WarmupLinearSchedule, WarmupCosineSchedule
from datasets import PaddedLongNLSTDataset, PaddedLongC2VDataset, PaddedLongICADataset
from TDEncoder import FeatViT, ICAFeatViT, MAE, ICAFeatTD, IcaRoiTD
from MLP import MLP
from Experiment import Experiment


CONFIG_DIR = "/home/local/VANDERBILT/litz/github/MASILab/LMcurves/configs"
SCHEDULES = {
    "Constant": ConstantLRSchedule,
    "WarmupConstant": WarmupConstantSchedule,
    "Linear": WarmupLinearSchedule,
    "Cosine": WarmupCosineSchedule,
}
model_classes = {
    "FeatViT": FeatViT,
    "ICAFeatViT": ICAFeatViT,
    "ICAFeatTD": ICAFeatTD,
    "IcaRoiTD": IcaRoiTD,
    "MLP": MLP,
}

def get_dataset(dataset, pids, prep_dir, labelf, pbb_dir, codef=None, embf=None, ehr_dim=2000, time_length=2, transform=None):
    map = {
        'PaddedLongNLSTDataset': lambda: PaddedLongNLSTDataset(pids, prep_dir, labelf, pbb_dir, ehr_dim=ehr_dim, time_length=time_length, img_transform=transform),
        'PaddedLongC2VDataset': lambda: PaddedLongC2VDataset(pids, prep_dir, labelf, pbb_dir, codef, embf, time_length=time_length, img_transform=transform),
        'PaddedLongICADataset': lambda: PaddedLongICADataset(pids, prep_dir, labelf, pbb_dir, ehr_dim=ehr_dim, time_length=time_length, img_transform=transform),
    }
    return map[dataset]()

############################################################################################################
# Single phase Training
############################################################################################################
def single_train(config, config_id):
    seed_everything(config["random_seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    feat_dir = config["feat_dir"]
    label = pd.read_csv(config["labels"], dtype={'pid':str})
    try:
        label = label[label['test_set']==False] # training set
    except KeyError:
        pass
    pids = label['pid'].unique().tolist()

    log_dir = os.path.join(config["root_dir"], "logs", config_id)
    checkpoint_dir = os.path.join(config["root_dir"], "checkpoints", config_id)
    model_dir = os.path.join(config["root_dir"], "models", config_id)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Training/Validation split, stratified by label
    labelpids = label.groupby('pid', as_index=False).max()
    val_pids = labelpids.groupby('lung_cancer', group_keys=False).apply(lambda x: x.sample(frac=config["val_fraction"]))['pid']
    train_pids = labelpids.drop(val_pids.index)['pid']
    train_pids, val_pids = train_pids.tolist(), val_pids.tolist()

    # Data
    train_transforms = ToTensor()
    train_dataset = get_dataset(config['dataset'], train_pids, feat_dir, config['labels'], config['pbb_dir'], config['codes'], config['icd10_embedding'], 
            ehr_dim=config['code_dim'], time_length=config['time_length'], transform=train_transforms)
    val_dataset = get_dataset(config['dataset'], val_pids, feat_dir, config['labels'], config['pbb_dir'], config['codes'], config['icd10_embedding'], 
            ehr_dim=config['code_dim'], time_length=config['time_length'], transform=train_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config['batch_size']), num_workers=4, pin_memory=True, shuffle=True)

    # Model
    model_class = model_classes[config["model_class"]]
    if config["model_class"]=="FeatViT":
        model = model_class(
            num_feat=1,
            feat_dim=config['feat_dim'],
            code_dim=100,
            num_classes=2,
            dim=config['embedding_dim'],
            depth=config['depth'],
            heads=config['heads'],
            mlp_dim=config['mlp_dim'],
            qkv_bias=config['qkv_bias'],
            time_encoding=config['time_enc'],
            dropout=config['dropout'],
        ).to(device)
    else:
        model = model_class(
            feat_dim=config['feat_dim'],
            code_dim=config['code_dim'],
            num_classes=2,
            dim=config['embedding_dim'],
            depth=config['depth'],
            heads=config['heads'],
            mlp_dim=config['mlp_dim'],
            qkv_bias=config['qkv_bias'],
            time_encoding=config['time_enc'],
            dim_head=config['dim_head'],
            dropout=config['dropout'],
        ).to(device)


    # freeze image embedding
    if config['freeze_img_emb']:
        model.freeze_img_emb()
    if config['freeze_ehr_emb']:
        model.freeze_ehr_emb()

    # loss and opt
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95))

    if config["pretrained_mae"]:
        print(f"From pretrained at {config['pretrained_mae']}")
        model.load_from_mae(torch.load(config['pretrained_mae']))

    if config["pretrained_model"]:
        print(f"From pretrained at {config['pretrained_model']}")
        model.load_state_dict(torch.load(config['pretrained_model']))
        
    if config["checkpoint"]:
        print(f"Resuming training of {config_id} from {config['checkpoint']}")
        checkpoint = torch.load(os.path.join(checkpoint_dir, config['checkpoint']))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_metric = checkpoint['best_metric']
        start_epoch = checkpoint['epoch'] + 1
        last_global_step = checkpoint['global_step']
    else:
        best_metric = torch.tensor(1e5)
        start_epoch = 0
        last_global_step = 0

    # epochs
    epoch_range = (start_epoch, config["epochs"])

    # scheduler
    n_batches = math.ceil(len(train_dataset)/int(config["batch_size"]))
    print(f"Total steps: {config['epochs']*n_batches}")
    if config["schedule"]=="Constant":
        scheduler = SCHEDULES[config["schedule"]](optimizer, last_epoch=start_epoch*n_batches-1)
    elif config["schedule"]=="WarmupConstant":
        scheduler = SCHEDULES[config["schedule"]](optimizer, warmup_steps=config["warmup_steps"],
                                                last_epoch=start_epoch*n_batches-1)
    else:
        scheduler = SCHEDULES[config["schedule"]](optimizer, warmup_steps=config["warmup_steps"],
                                                t_total=config["epochs"]*n_batches, last_epoch=start_epoch*n_batches-1)

    stopper = EarlyStopper(config["stop_agg"], config["stop_delta"])
    writer = SummaryWriter(log_dir=log_dir)        

    print(f"Using {train_dataset.__class__.__name__} on {model.__class__.__name__}")     
    experiment = Experiment(config, config_id, None, model, epoch_range, last_global_step, train_loader, val_loader, device, criterion, 
        optimizer, stopper, scheduler, writer, model_dir, checkpoint_dir, best_metric)
    experiment.train()

############################################################################################################
# CV Training
############################################################################################################
def cv_train(config, config_id, num_folds):
    seed_everything(config["random_seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    feat_dir = config["feat_dir"]
    label = pd.read_csv(config["labels"], dtype={'pid':str})

    # split into k folds
    label = label.groupby('pid', as_index=False)['lung_cancer'].max()
    skf = StratifiedKFold(n_splits=int(num_folds), shuffle=True, random_state=config['random_seed'])
    cv_splits = skf.split(label['pid'], label['lung_cancer']) # generator for cv split indexes

    # shuffled = label['pid'].unique().tolist()
    # random.shuffle(shuffled)
    # cv_splits = np.array_split(shuffled, int(num_folds))

    for k, (infold_idx, _) in enumerate(cv_splits):

        log_dir = os.path.join(config["root_dir"], "logs", config_id, f"fold{k}")
        checkpoint_dir = os.path.join(config["root_dir"], "checkpoints", config_id, f"fold{k}")
        model_dir = os.path.join(config["root_dir"], "models", config_id, f"fold{k}")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Training/Validation split
        split = label.iloc[infold_idx]
        val_pids = split.groupby('lung_cancer', group_keys=False).apply(lambda x: x.sample(frac=config["val_fraction"]))['pid']
        train_pids = split.drop(val_pids.index)['pid']
        train_pids, val_pids = train_pids.tolist(), val_pids.tolist()

        # Data
        train_transforms = ToTensor()
        train_dataset = get_dataset(config['dataset'], train_pids, feat_dir,config['labels'], config['pbb_dir'], config['codes'], config['icd10_embedding'], 
            ehr_dim=config['code_dim'], time_length=config['time_length'], transform=train_transforms)
        val_dataset = get_dataset(config['dataset'], val_pids, feat_dir,config['labels'], config['pbb_dir'], config['codes'], config['icd10_embedding'], 
            ehr_dim=config['code_dim'], time_length=config['time_length'], transform=train_transforms)

        train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=int(config['batch_size']), num_workers=4, pin_memory=True, shuffle=True)
        # Model
        model_class = model_classes[config["model_class"]]
        if config["model_class"]=="FeatViT":
            model = model_class(
                num_feat=1,
                feat_dim=config['feat_dim'],
                code_dim=100,
                num_classes=2,
                dim=config['embedding_dim'],
                depth=config['depth'],
                heads=config['heads'],
                mlp_dim=config['mlp_dim'],
                qkv_bias=config['qkv_bias'],
                time_encoding=config['time_enc'],
                dim_head=config['dim_head'],
                dropout=config['dropout'],
            ).to(device)
        else:
            model = model_class(
                feat_dim=config['feat_dim'],
                code_dim=config['code_dim'],
                num_classes=2,
                dim=config['embedding_dim'],
                depth=config['depth'],
                heads=config['heads'],
                mlp_dim=config['mlp_dim'],
                qkv_bias=config['qkv_bias'],
                time_encoding=config['time_enc'],
                dim_head=config['dim_head'],
                dropout=config['dropout'],
            ).to(device)

        # freeze image embedding
        if config['freeze_img_emb']:
            model.freeze_img_emb()
        if config['freeze_ehr_emb']:
            model.freeze_ehr_emb()

        # loss and opt
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95))

        if config["pretrained_mae"]:
            print(f"From pretrained at {config['pretrained_mae']}")
            model.load_from_mae(torch.load(config['pretrained_mae']))

        if config["pretrained_model"]:
            print(f"From pretrained at {config['pretrained_model']}")
            model.load_state_dict(torch.load(config['pretrained_model']))
            
        if config["checkpoint"]:
            print(f"Resuming training of {config_id} from {config['checkpoint']}")
            checkpoint = torch.load(os.path.join(checkpoint_dir, config['checkpoint']))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_metric = checkpoint['best_metric']
            start_epoch = checkpoint['epoch'] + 1
            last_global_step = checkpoint['global_step']
        else:
            best_metric = torch.tensor(1e5)
            start_epoch = 0
            last_global_step = 0

        # epochs
        epoch_range = (start_epoch, config["epochs"])

        # scheduler
        n_batches = math.ceil(len(train_dataset)/int(config["batch_size"]))
        print(f"Total steps: {config['epochs']*n_batches}")
        if config["schedule"]=="Constant":
            scheduler = SCHEDULES[config["schedule"]](optimizer, last_epoch=start_epoch*n_batches-1)
        elif config["schedule"]=="WarmupConstant":
            scheduler = SCHEDULES[config["schedule"]](optimizer, warmup_steps=config["warmup_steps"],
                                                    last_epoch=start_epoch*n_batches-1)
        else:
            scheduler = SCHEDULES[config["schedule"]](optimizer, warmup_steps=config["warmup_steps"],
                                                    t_total=config["epochs"]*n_batches, last_epoch=start_epoch*n_batches-1)

        stopper = EarlyStopper(config["stop_agg"], config["stop_delta"])
        writer = SummaryWriter(log_dir=log_dir)   

        print(f"Using {train_dataset.__class__.__name__} on {model.__class__.__name__}")     
        print(f"Start Training Fold {k} =============")
        experiment = Experiment(config, config_id, k, model, epoch_range, last_global_step, train_loader, val_loader, device, criterion, 
            optimizer, stopper, scheduler, writer, model_dir, checkpoint_dir, best_metric)
        experiment.train()
        
        # clear mem for next batch
        model = None
        gc.collect()

############################################################################################################
# Pretraining
############################################################################################################

def pretrain(config, config_id):
    seed_everything(config["random_seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    feat_dir = config["feat_dir"]
    label = pd.read_csv(config["labels"], dtype={'pid':str})
    pids = label['pid'].unique().tolist()

    log_dir = os.path.join(config["root_dir"], "logs", config_id)
    checkpoint_dir = os.path.join(config["root_dir"], "checkpoints", config_id)
    model_dir = os.path.join(config["root_dir"], "models", config_id)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Training/Validation split
    val_size = int(np.floor(config["val_fraction"] * len(pids)))
    random.shuffle(pids)
    val_pids, train_pids = pids[:val_size], pids[val_size:]

    # Data
    train_transforms = ToTensor()
    train_dataset = get_dataset(config['dataset'], train_pids, feat_dir,config['labels'], config['pbb_dir'], config['codes'], config['icd10_embedding'], 
        ehr_dim=config['code_dim'], time_length=config['time_length'], transform=train_transforms)
    val_dataset = get_dataset(config['dataset'], val_pids, feat_dir,config['labels'], config['pbb_dir'], config['codes'], config['icd10_embedding'], 
        ehr_dim=config['code_dim'], time_length=config['time_length'], transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config['batch_size']), num_workers=4, pin_memory=True, shuffle=True)

    # Model
    encoder = FeatViT(
        num_feat=1,
        feat_dim=config['feat_dim'],
        code_dim=100,
        num_classes=2,
        dim=config['embedding_dim'],
        depth=config['depth'],
        heads=config['heads'],
        mlp_dim=config['mlp_dim'],
        qkv_bias=config['qkv_bias'],
        time_encoding=config['time_enc'],
        dropout=config['dropout'],
    ).to(device)
    model = MAE(encoder=encoder, masking_ratio=config['masking_ratio'], decoder_dim=config['embedding_dim'], decoder_depth=4).to(device)

    # loss and opt
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95))

    if config["checkpoint"]:
        print(f"Resuming training of {config_id} from {config['checkpoint']}")
        checkpoint = torch.load(os.path.join(checkpoint_dir, config['checkpoint']))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_metric = checkpoint['best_metric']
        start_epoch = checkpoint['epoch'] + 1
        last_global_step = checkpoint['global_step']
    else:
        best_metric = 1e5
        start_epoch = 0
        last_global_step = 0

    # epochs
    epoch_range = (start_epoch, config["epochs"])

    # scheduler
    n_batches = math.ceil(len(train_dataset)/int(config["batch_size"]))
    print(f"Total steps: {config['epochs']*n_batches}")
    if config["schedule"]=="Constant":
        scheduler = SCHEDULES[config["schedule"]](optimizer, last_epoch=start_epoch*n_batches-1)
    elif config["schedule"]=="WarmupConstant":
        scheduler = SCHEDULES[config["schedule"]](optimizer, warmup_steps=config["warmup_steps"],
                                                last_epoch=start_epoch*n_batches-1)
    else:
        scheduler = SCHEDULES[config["schedule"]](optimizer, warmup_steps=config["warmup_steps"],
                                                t_total=config["epochs"]*n_batches, last_epoch=start_epoch*n_batches-1)

    stopper = EarlyStopper(config["stop_agg"], config["stop_delta"])
    writer = SummaryWriter(log_dir=log_dir)        

    experiment = Experiment(config, config_id, None, model, epoch_range, last_global_step, train_loader, val_loader, device, None, 
            optimizer, stopper, scheduler, writer, model_dir, checkpoint_dir, best_metric)
    experiment.pretrain_train()
    

############################################################################################################
# Single Testing
############################################################################################################
def single_test(config, config_id):
    seed_everything(config["random_seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    feat_dir = config["feat_dir"]
    label = pd.read_csv(config["labels"], dtype={'pid':str})
    try:
        label = label[label['test_set']==True] # training set
    except KeyError:
        pass
    test_pids = label['pid'].unique().tolist()

    # split into k folds
    log_dir = os.path.join(config["root_dir"], "logs", config_id)
    checkpoint_dir = os.path.join(config["root_dir"], "checkpoints", config_id)
    model_dir = os.path.join(config["root_dir"], "models", config_id)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Data
    transforms = ToTensor()

    test_dataset = get_dataset(config['dataset'], test_pids, feat_dir,config['labels'], config['pbb_dir'], config['codes'], config['icd10_embedding'], 
            ehr_dim=config['code_dim'], time_length=config['time_length'], transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True)
    
    # Model
    model_class = model_classes[config["model_class"]]
    if config["model_class"]=="FeatViT":
        model = FeatViT(
            num_feat=1,
            feat_dim=config['feat_dim'],
            code_dim=100,
            num_classes=2,
            dim=config['embedding_dim'],
            depth=config['depth'],
            heads=config['heads'],
            mlp_dim=config['mlp_dim'],
            qkv_bias=config['qkv_bias'],
            time_encoding=config['time_enc'],
            dim_head=config['dim_head'],
            dropout=config['dropout'],
        ).to(device)
    elif config['model_class']=="ICAFeatViT":
        model = model_class(
            feat_dim=config['feat_dim'],
            code_dim=config['code_dim'],
            num_classes=2,
            dim=config['embedding_dim'],
            depth=config['depth'],
            heads=config['heads'],
            mlp_dim=config['mlp_dim'],
            qkv_bias=config['qkv_bias'],
            time_encoding=config['time_enc'],
            dim_head=config['dim_head'],
            dropout=config['dropout'],
        ).to(device)
    else:
        model = model_class(
            num_feat=6,
            nod_dim=config['feat_dim'],
            code_dim=100,
            feat_dim=64,
            num_classes=2
        ).to(device)

    model_path = os.path.join(model_dir, "best_model.pth")
    model.load_state_dict(torch.load(model_path))

    print(f"Using {test_dataset.__class__.__name__} on {model.__class__.__name__}")   
    print(f"Single testing {len(test_dataset)} subjects")
    experiment = Experiment(model=model, val_loader=test_loader, device=device, model_dir=model_dir)
    experiment.test()


############################################################################################################
# CV Testing
############################################################################################################
def cv_test(config, config_id, num_folds):
    seed_everything(config["random_seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    feat_dir = config["feat_dir"]
    label = pd.read_csv(config["labels"], dtype={'pid':str})

    # split into k folds
    label = label.groupby('pid', as_index=False)['lung_cancer'].max()
    skf = StratifiedKFold(n_splits=int(num_folds), shuffle=True, random_state=config['random_seed'])
    cv_splits = skf.split(label['pid'], label['lung_cancer']) # generator for cv split indexes

    for k, (_, outfold_idx) in enumerate(cv_splits):
        out_fold = label.iloc[outfold_idx]['pid'].tolist()

        log_dir = os.path.join(config["root_dir"], "logs", config_id, f"fold{k}")
        checkpoint_dir = os.path.join(config["root_dir"], "checkpoints", config_id, f"fold{k}")
        model_dir = os.path.join(config["root_dir"], "models", config_id, f"fold{k}")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Data
        transforms = ToTensor()
        test_dataset = get_dataset(config['dataset'], out_fold, feat_dir, config['labels'], config['pbb_dir'], config['codes'], config['icd10_embedding'], 
            ehr_dim=config['code_dim'], time_length=config['time_length'], transform=transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True)
        
        # Model
        model_class = model_classes[config["model_class"]]
        if config["model_class"]=="FeatViT":
            model = FeatViT(
                num_feat=1,
                feat_dim=config['feat_dim'],
                code_dim=100,
                num_classes=2,
                dim=config['embedding_dim'],
                depth=config['depth'],
                heads=config['heads'],
                mlp_dim=config['mlp_dim'],
                qkv_bias=config['qkv_bias'],
                time_encoding=config['time_enc'],
                dim_head=config['dim_head'],
                dropout=config['dropout'],
            ).to(device)
        else:
            model = model_class(
                feat_dim=config['feat_dim'],
                code_dim=config['code_dim'],
                num_classes=2,
                dim=config['embedding_dim'],
                depth=config['depth'],
                heads=config['heads'],
                mlp_dim=config['mlp_dim'],
                qkv_bias=config['qkv_bias'],
                time_encoding=config['time_enc'],
                dim_head=config['dim_head'],
                dropout=config['dropout'],
            ).to(device)

        model_path = os.path.join(model_dir, "best_model.pth")
        model.load_state_dict(torch.load(model_path))

        print(f"Using {test_dataset.__class__.__name__} on {model.__class__.__name__}")   
        print(f"Testing {len(test_dataset)} subjects, fold {k}")
        experiment = Experiment(model=model, val_loader=test_loader, device=device, model_dir=model_dir)
        experiment.test()

        # clear mem for next batch
        model = None
        gc.collect()
        
############################################################################################################
# Infer
############################################################################################################

def infer(config, config_id):
    seed_everything(config["random_seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    feat_dir = config["feat_dir"]
    label = pd.read_csv(config["labels"], dtype={'pid':str})
    try:
        label = label[label['test_set']==True] # training set
    except KeyError:
        pass

    pids = label['pid'].unique().tolist()
    random.shuffle(pids)

    # UNCOMMENT: Test a specific fold
    # shuffled = label['pid'].unique().tolist()
    # random.shuffle(shuffled)
    # cv_splits = np.array_split(shuffled, int(3))
    # pids = cv_splits[2]

    model_dir = os.path.join(config["root_dir"], "models", config_id)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Data
    time_range = (0, 1)
    transforms = ToTensor()
    test_dataset = get_dataset(config['dataset'], pids, feat_dir, config['labels'], config['codes'], config['icd10_embedding'],
        ehr_dim=config['code_dim'],agg_years=3, time_length=config['time_length'], transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True)
    
    # Model
    model_class = model_classes[config["model_class"]]
    if config["model_class"]=="FeatViT":
        model = FeatViT(
            num_feat=6,
            feat_dim=config['feat_dim'],
            code_dim=100,
            num_classes=2,
            dim=config['embedding_dim'],
            depth=config['depth'],
            heads=config['heads'],
            mlp_dim=config['mlp_dim'],
            qkv_bias=config['qkv_bias'],
            time_encoding=config['time_enc'],
            dim_head=config['dim_head'],
            dropout=config['dropout'],
        ).to(device)
    elif config['model_class']=="ICAFeatViT":
        model = model_class(
            feat_dim=config['feat_dim'],
            code_dim=config['code_dim'],
            num_classes=2,
            dim=config['embedding_dim'],
            depth=config['depth'],
            heads=config['heads'],
            mlp_dim=config['mlp_dim'],
            qkv_bias=config['qkv_bias'],
            time_encoding=config['time_enc'],
            dim_head=config['dim_head'],
            dropout=config['dropout'],
        ).to(device)
    else:
        model = model_class(
            num_feat=6,
            nod_dim=config['feat_dim'],
            code_dim=100,
            feat_dim=64,
            num_classes=2
        ).to(device)

    # model_path = os.path.join(model_dir, "best_model.pth")
    # infer from checkpoint or pretrained model
    if config['checkpoint']:
        print(f"From checkpoint at {config['checkpoint']}")
        checkpoint = torch.load(config['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(config["pretrained_model"]))

    experiment = Experiment(model=model, val_loader=test_loader, device=device, model_dir=model_dir)
    experiment.test()


############################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--single_train', action='store_true', default=False)
    parser.add_argument('--single_test', action='store_true', default=False)
    parser.add_argument('--cv_train', action='store_true', default=False)
    parser.add_argument('--cv_test', action='store_true', default=False)
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--folds', type=str)
    args = parser.parse_args()

    config = load_config(CONFIG_DIR, args.config)
    if args.single_train:
        single_train(config, args.config)
    if args.single_test:
        single_test(config, args.config)
    if args.cv_train:
        cv_train(config, args.config, args.folds)
    if args.cv_test:
        cv_test(config, args.config, args.folds)
    if args.pretrain:
        pretrain(config, args.config)
    if args.infer:
        infer(config, args.config)