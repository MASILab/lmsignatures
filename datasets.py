import os
import sys
import glob
import json
import pandas as pd
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None # suppress setting as a copy warning
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
sys.path.append('/home/local/VANDERBILT/litz/github/MASILab/ICD9CMtoICD10CM/')
from ICD9to10 import ICD9to10

from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset
from einops import rearrange

sys.path.append("/home/local/VANDERBILT/litz/github/MASILab/DSB2017")
from layers import nms

from torchvision.transforms import ToTensor


def load_icd10_emb(embf):
    with open(embf, 'r') as f:
        emb_file = json.load(f)
    emb_dict = {
        k : [float(x) for x in emb_file['SG_CO_embeddings_100'][k]]
        for k in emb_file['SG_CO_embeddings_100'].keys()
    }
    return emb_dict, len(emb_dict['A01.00'])

############################################################################################################
# Code2Vec
############################################################################################################

class CSLiVUFeatDataset(Dataset):
    def __init__(self, pids, feat_dir, labelf, codef, embf, agg_years=1, feat_dim=64, topk=5, img_transform=ToTensor(), tfidf=False) -> None:
        super().__init__()
        self.pids = pids
        self.feat_dir = feat_dir
        self.feat_dim = feat_dim
        self.topk = topk
        self.img_transform = img_transform
        self.label = pd.read_csv(labelf, dtype={'pid':str})
        self.code2vec = True if codef else False
        # load embeddings
        emb_dict, self.ehr_dim = load_icd10_emb(embf)

        if self.code2vec:
            code = pd.read_csv(codef, dtype={'mcl_id':str})

            # merge label and codes
            label = self.label.copy()
            label = code.merge(label[['pid', 'id', 'shifted_scan_date']], left_on='mcl_id', right_on='pid')
            label['shifted_icd_date'] = pd.to_datetime(label['shifted_icd_date'])
            label['shifted_scan_date'] = pd.to_datetime(label['shifted_scan_date'])
            aggtime = timedelta(days=365*int(agg_years))
            label = label[(label['shifted_icd_date'] >= label['shifted_scan_date'] - aggtime) & (label['shifted_icd_date'] < label['shifted_scan_date'])]

            # convert ICD9 -> ICD10
            converter = ICD9to10('/home/local/VANDERBILT/litz/github/MASILab/ICD9CMtoICD10CM/icd9to10dictionary.txt')
            icd10 = pd.Series(converter.convert(label['ICD_CODE'].tolist()))
            icd10 = icd10.fillna(label['ICD_CODE'].reset_index(drop=True))
            label['icd10_code'] = icd10.tolist()

            # filter label file for pids and valid ICD10
            label = label[label['mcl_id'].isin(pids)] 
            label = label[label['icd10_code'].isin(emb_dict.keys())] # may be some scans dropped if no icd codes during agg period
            emb_label = label.groupby(['mcl_id'], as_index=False).max()
            self.emb_label = emb_label.sort_values(by=['mcl_id'], ascending=False, ignore_index=True) # index matches order of emb matrix

            # compute subject level embeddings https://github.com/crownpku/text2vec/blob/master/wv_wrt_tfidf.md
            scanids = self.emb_label['id'].tolist() 
            corpus = [' '.join(label[label['id']==i]['icd10_code'].tolist()) for i in scanids]
            # tfidf weighting
            if tfidf: 
                vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r'[^\s]+')
            else:
                vectorizer = CountVectorizer(lowercase=False, token_pattern=r'[^\s]+')
            X = vectorizer.fit_transform(corpus) # nxd, n=len(scanids), d=unique codes
            # divide sum of tf-idf
            d = X.shape[1]
            X_sum = np.tile(np.sum(X, axis=1), d) # divide each sample emb by their sum tf-idf
            W = np.zeros((len(vectorizer.vocabulary_), self.ehr_dim)) # dxw, w=emb dim
            for c, i in vectorizer.vocabulary_.items():
                W[i] = emb_dict[c]
            self.emb = X/X_sum @ W # nxw, n <= 2xsubjects


    def __getitem__(self, index):
        pid = self.pids[index]
        pid_rows = self.label[self.label['pid']==pid].sort_values(by=['session'], ascending=False)
        pid_row = pid_rows.iloc[0]
        
        # get code embeddings
        if self.code2vec:
            emb_row = self.emb_label[self.emb_label['pid']==pid]
            code_emb = np.zeros((1, self.ehr_dim))
            if len(emb_row) > 0:
                emb_i = emb_row.index.tolist()
                code_emb = self.emb[emb_i]
            else:
                code_emb = np.zeros((1, self.ehr_dim)) # dropped scans are zero vectors
            code_emb = torch.from_numpy(code_emb).to(torch.float32)
        else:
            code_emb = torch.zeros(size=(1, self.ehr_dim), dtype=torch.float32)

        # nodule embeddings
        fname = pid_row['id']
        feat = np.load(os.path.join(self.feat_dir, f"{fname}.npy"))[:self.topk]
        feat = self.img_transform(feat)

        # ROIs
        samplerows = samplerows.sort_values(by='session', ascending=False)
        seq = torch.zeros((self.topk, self.patch_size, self.patch_size, self.patch_size), dtype=torch.float32)
        fnames = samplerows['id'].tolist()
        pbb = None
        for t, fname in enumerate(fnames):
            tmp = np.load(os.path.join(self.prep_dir, f"{fname}_clean.npy"))
            
            # find ROI of the latest scan and extract same region from all previous
            if t==0:
                pbb = np.load(os.path.join(self.pbb_dir, f"{fname}_pbb.npy"))
                pbb = pbb[pbb[:,0]>-1]
                pbb = nms(pbb,0.05)
            
            # crop ROIs and if less than topk ROIs, leave rest of patches as zero
            conf_list = pbb[:,0]
            chosenid = conf_list.argsort()[::-1][:self.topk]
            croplist = np.zeros([self.topk,self.crop_size[0],self.crop_size[1],self.crop_size[2]]).astype('float32')
            for k, cid in enumerate(chosenid):
                target=pbb[cid, 1:]
                crop = self.crop(tmp, target)
                crop = crop.astype(np.float32)
                if self.img_transform:
                    crop = self.img_transform(crop)
                croplist[k] = crop
            seq[t] = torch.from_numpy(croplist)

        # get label
        label = int(pid_row['lung_cancer'])

        return {"fnames": [fname], "imgs": seq, "ehr": code_emb, "label": label}

    def __len__(self):
        return len(self.pids)


class PaddedLongLiVUFeatDataset(Dataset):
    """Returns time_length randomly sampled cross sections with padding if time points are missing (superset of LIVU)"""
    def __init__(self,  pids, feat_dir, labelf, codef, embf, agg_years=1, feat_dim=64, time_length=2, topk=5, img_transform=ToTensor(), tfidf=False):
        super().__init__()
        self.feat_dir = feat_dir
        self.pids = pids
        self.feat_dir = feat_dir
        self.feat_dim = feat_dim
        self.time_length = time_length
        self.topk = topk
        self.img_transform = img_transform
        self.label = pd.read_csv(labelf, dtype={'pid':str})
        self.code2vec = True if codef else False
        
        # load ICD10 embeddings
        emb_dict, self.ehr_dim = load_icd10_emb(embf)

        if self.code2vec:
            code = pd.read_csv(codef, dtype={'mcl_id':str})
            # merge label and codes
            label = self.label.copy()
            label['session'] = label['session'] - label.groupby('pid')['session'].transform('min') # number each scan 0 or 1
            label = code.merge(label[['pid', 'id', 'session', 'shifted_scan_date', 'Duration']], left_on='mcl_id', right_on='pid')
            label['shifted_icd_date'] = pd.to_datetime(label['shifted_icd_date'])
            label['shifted_scan_date'] = pd.to_datetime(label['shifted_scan_date'])
            aggtime = timedelta(days=365*int(agg_years))
            label = label[(label['shifted_icd_date'] >= label['shifted_scan_date'] - aggtime) & (label['shifted_icd_date'] < label['shifted_scan_date'])]

            # convert ICD9 -> ICD10
            converter = ICD9to10('/home/local/VANDERBILT/litz/github/MASILab/ICD9CMtoICD10CM/icd9to10dictionary.txt')
            icd10 = pd.Series(converter.convert(label['ICD_CODE'].tolist()))
            icd10 = icd10.fillna(label['ICD_CODE'].reset_index(drop=True))
            label['icd10_code'] = icd10.tolist()

            # filter label file for pids and valid ICD10
            label = label[label['mcl_id'].isin(pids)] 
            label = label[label['icd10_code'].isin(emb_dict.keys())] # may be some scans dropped if no icd codes during agg period
            emb_label = label.groupby(['mcl_id', 'shifted_scan_date'], as_index=False).max()
            self.emb_label = emb_label.sort_values(by=['mcl_id', 'shifted_scan_date'], ascending=False, ignore_index=True) # index matches order of emb matrix

            # compute subject level embeddings https://github.com/crownpku/text2vec/blob/master/wv_wrt_tfidf.md
            scanids = self.emb_label['id'].tolist() 
            corpus = [' '.join(label[label['id']==i]['icd10_code'].tolist()) for i in scanids]
            # tfidf weighting
            if tfidf: 
                vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r'[^\s]+')
            else:
                vectorizer = CountVectorizer(lowercase=False, token_pattern=r'[^\s]+')
            X = vectorizer.fit_transform(corpus) # nxd, n=len(scanids), d=unique codes
            # divide sum of tf-idf
            d = X.shape[1]
            X_sum = np.tile(np.sum(X, axis=1), d) # divide each sample emb by their sum tf-idf
            W = np.zeros((len(vectorizer.vocabulary_), self.ehr_dim)) # dxw, w=emb dim
            for c, i in vectorizer.vocabulary_.items():
                W[i] = emb_dict[c]
            self.emb = X/X_sum @ W # nxw, n <= 2xsubjects
        
    def __getitem__(self, index):
        pid = self.pids[index]
        # returns descending scan date
        pid_rows = self.label[self.label['pid']==pid].sort_values(by=['session'], ascending=False)
        randidx = random.sample(range(len(pid_rows)), min(len(pid_rows), self.time_length))
        samplerows = pid_rows.iloc[randidx]
        samplerows['session_rank'] = samplerows['session'].rank(ascending=False).astype(int) - 1 # rank sessions in descending order

        # padding up to time_length. used to generate attention mask
        padding = np.zeros(self.time_length, dtype='float32')
        padding[:len(samplerows)] = 1 # 1 if value present, 0 if padding

        # get ICA expressions
        code_emb = np.zeros((self.time_length, self.ehr_dim))
        if self.code2vec:
            emb_rows = self.emb_label[self.emb_label['id'].isin(samplerows['id'])]
            emb_rows = self.emb_label.reset_index().merge(samplerows, on='id').set_index('index')
            code_emb[emb_rows['session_rank'].tolist()] = self.emb[emb_rows.index.tolist()] # stores embedding vectors in the order of sess rank
        code_emb = torch.from_numpy(code_emb).to(torch.float32)

        # ROIs
        samplerows = samplerows.sort_values(by='session', ascending=False)
        seq = torch.zeros((self.time_length, self.topk, self.patch_size, self.patch_size, self.patch_size), dtype=torch.float32)
        fnames = samplerows['id'].tolist()
        pbb = None
        for t, fname in enumerate(fnames):
            tmp = np.load(os.path.join(self.prep_dir, f"{fname}_clean.npy"))
            
            # find ROI of the latest scan and extract same region from all previous
            if t==0:
                pbb = np.load(os.path.join(self.pbb_dir, f"{fname}_pbb.npy"))
                pbb = pbb[pbb[:,0]>-1]
                pbb = nms(pbb,0.05)
            
            # crop ROIs and if less than topk ROIs, leave rest of patches as zero
            conf_list = pbb[:,0]
            chosenid = conf_list.argsort()[::-1][:self.topk]
            croplist = np.zeros([self.topk,self.crop_size[0],self.crop_size[1],self.crop_size[2]]).astype('float32')
            for k, cid in enumerate(chosenid):
                target=pbb[cid, 1:]
                crop = self.crop(tmp, target)
                crop = crop.astype(np.float32)
                if self.img_transform:
                    crop = self.img_transform(crop)
                croplist[k] = crop
            seq[t] = torch.from_numpy(croplist)
        
        # relative time distances in descending order
        times_padded = torch.zeros(self.time_length)
        times = samplerows['Duration'].tolist()
        times = [t-times[0] for t in times] # set latest scan t=0
        times = torch.Tensor(times)/30.33 # transform into fractional months
        times_padded[:len(samplerows)] = times

        label = int(pid_rows.iloc[0]['lung_cancer'])

        fnames_padded = ['' for i in range(self.time_length)]
        fnames_padded[:len(samplerows)] = samplerows['id'].tolist()

        return {"fnames": fnames_padded, "imgs": img, "ehr": code_emb, "padding": padding, "times": times_padded, "label": label}

    def __len__(self):
        return len(self.pids)

class LongImageVUFeatDataset(Dataset):
    """Returns two randomly sampled cross sections from a subject in ImageVU (superset of LIVU)"""
    def __init__(self, pids, feat_dir, labelf, codef, embf, agg_years=1, feat_dim=64, time_range=(0, 1), topk=5, img_transform=None, tfidf=False) -> None:
        super().__init__()
        self.pids = pids
        self.feat_dir = feat_dir
        self.feat_dim = feat_dim
        self.time_length = time_range[1] + 1
        self.topk = topk
        self.img_transform = img_transform
        self.label = pd.read_csv(labelf, dtype={'pid':str})
        code = pd.read_csv(codef, dtype={'mcl_id':str})

        # load ICD10 embeddings
        emb_dict, self.ehr_dim = load_icd10_emb(embf)

        # merge label and codes
        label = self.label.copy()
        label['session'] = label['session'] - label.groupby('pid')['session'].transform('min') # number each scan 0 or 1
        label = code.merge(label[['pid', 'id', 'session', 'shifted_scan_date', 'Duration']], left_on='mcl_id', right_on='pid')
        label['shifted_icd_date'] = pd.to_datetime(label['shifted_icd_date'])
        label['shifted_scan_date'] = pd.to_datetime(label['shifted_scan_date'])
        aggtime = timedelta(days=365*int(agg_years))
        label = label[(label['shifted_icd_date'] >= label['shifted_scan_date'] - aggtime) & (label['shifted_icd_date'] < label['shifted_scan_date'])]

        # convert ICD9 -> ICD10
        converter = ICD9to10('/home/local/VANDERBILT/litz/github/MASILab/ICD9CMtoICD10CM/icd9to10dictionary.txt')
        icd10 = pd.Series(converter.convert(label['ICD_CODE'].tolist()))
        icd10 = icd10.fillna(label['ICD_CODE'].reset_index(drop=True))
        label['icd10_code'] = icd10.tolist()

        # filter label file for pids and valid ICD10
        label = label[label['mcl_id'].isin(pids)] 
        label = label[label['icd10_code'].isin(emb_dict.keys())] # may be some scans dropped if no icd codes during agg period
        emb_label = label.groupby(['mcl_id', 'shifted_scan_date'], as_index=False).max()
        self.emb_label = emb_label.sort_values(by=['mcl_id', 'shifted_scan_date'], ascending=False, ignore_index=True) # index matches order of emb matrix

        # compute subject level embeddings https://github.com/crownpku/text2vec/blob/master/wv_wrt_tfidf.md
        scanids = self.emb_label['id'].tolist() 
        corpus = [' '.join(label[label['id']==i]['icd10_code'].tolist()) for i in scanids]
        # tfidf weighting
        if tfidf: 
            vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r'[^\s]+')
        else:
            vectorizer = CountVectorizer(lowercase=False, token_pattern=r'[^\s]+')
        X = vectorizer.fit_transform(corpus) # nxd, n=len(scanids), d=unique codes
        # divide sum of tf-idf
        d = X.shape[1]
        X_sum = np.tile(np.sum(X, axis=1), d) # divide each sample emb by their sum tf-idf or sum count
        W = np.zeros((len(vectorizer.vocabulary_), self.ehr_dim)) # dxw, w=emb dim
        for c, i in vectorizer.vocabulary_.items():
            W[i] = emb_dict[c]
        self.emb = X/X_sum @ W # nxw, n <= 2xsubjects

    def __getitem__(self, index):
        pid = self.pids[index]
        pid_rows = self.label[self.label['pid']==pid]

        # randomly sample two cross sections
        randidx = random.sample(range(len(pid_rows)), 2)
        # pid_rows = pid_rows.iloc[randidx]
        twosample_rows = pid_rows.iloc[randidx] 
        twosample_rows['session_rank'] = twosample_rows['session'].rank(ascending=False).astype(int) - 1 # rank sessions
        emb_rows = self.emb_label.reset_index().merge(twosample_rows, on='id').set_index('index') # len can be 0, 1, or 2

        code_emb = np.zeros((2, self.ehr_dim))
        code_emb[emb_rows['session_rank'].tolist()] = self.emb[emb_rows.index.tolist()] # stores embedding vectors in the order of sess rank
        code_emb = torch.from_numpy(code_emb).to(torch.float32)

        twosample_rows = twosample_rows.sort_values(by='session', ascending=False)
        seq = np.zeros((self.time_length, 5, self.feat_dim), dtype='float32')
        fnames = twosample_rows['id'].tolist()
        for t, fname in enumerate(fnames):
            feat = np.load(os.path.join(self.feat_dir, f"{fname}.npy"))[:self.topk]
            seq[t] = self.img_transform(feat)

        # ROIs
        twosample_rows = twosample_rows.sort_values(by='session', ascending=False)
        seq = torch.zeros((self.time_length, self.topk, self.patch_size, self.patch_size, self.patch_size), dtype=torch.float32)
        fnames = twosample_rows['id'].tolist()
        pbb = None
        for t, fname in enumerate(fnames):
            tmp = np.load(os.path.join(self.prep_dir, f"{fname}_clean.npy"))
            
            # find ROI of the latest scan and extract same region from all previous
            if t==0:
                pbb = np.load(os.path.join(self.pbb_dir, f"{fname}_pbb.npy"))
                pbb = pbb[pbb[:,0]>-1]
                pbb = nms(pbb,0.05)
            
            # crop ROIs and if less than topk ROIs, leave rest of patches as zero
            conf_list = pbb[:,0]
            chosenid = conf_list.argsort()[::-1][:self.topk]
            croplist = np.zeros([self.topk,self.crop_size[0],self.crop_size[1],self.crop_size[2]]).astype('float32')
            for k, cid in enumerate(chosenid):
                target=pbb[cid, 1:]
                crop = self.crop(tmp, target)
                crop = crop.astype(np.float32)
                if self.img_transform:
                    crop = self.img_transform(crop)
                croplist[k] = crop
            seq[t] = torch.from_numpy(croplist)
        
        # relative time distances in descending order
        times = twosample_rows['Duration'].tolist()
        times = [t-times[0] for t in times] # set latest scan t=0
        times = torch.Tensor(times)/30.33 # transform into fractional months

        return {"fnames": twosample_rows['id'].tolist(), "imgs": seq, "ehr": code_emb, "times": times}

    def __len__(self):
        return len(self.pids)

class CSNLSTFeatDataset(Dataset):
    """cross sectional NLST nodule features + zero code embeddings"""
    def __init__(self,  pids, feat_dir, labelf, feat_dim=64, topk=5, img_transform=ToTensor(), ehr_dim=100):
        super().__init__()
        self.feat_dir = feat_dir
        self.pids = pids
        self.label = pd.read_csv(labelf, dtype={'pid':str})
        self.feat_dim = feat_dim
        self.topk = topk
        self.img_transform = img_transform
        self.ehr_dim = ehr_dim
        
    def __getitem__(self, index):
        # returns (t c x y z)
        pid = self.pids[index]
        # most recent scan
        pid_row = self.label[self.label['pid']==pid].sort_values(by=['session'], ascending=False).iloc[0]
        
         # ROIs
        samplerows = samplerows.sort_values(by='session', ascending=False)
        seq = torch.zeros((self.topk, self.patch_size, self.patch_size, self.patch_size), dtype=torch.float32)
        fnames = samplerows['id'].tolist()
        pbb = None
        for t, fname in enumerate(fnames):
            tmp = np.load(os.path.join(self.prep_dir, f"{fname}_clean.npy"))
            
            # find ROI of the latest scan and extract same region from all previous
            if t==0:
                pbb = np.load(os.path.join(self.pbb_dir, f"{fname}_pbb.npy"))
                pbb = pbb[pbb[:,0]>-1]
                pbb = nms(pbb,0.05)
            
            # crop ROIs and if less than topk ROIs, leave rest of patches as zero
            conf_list = pbb[:,0]
            chosenid = conf_list.argsort()[::-1][:self.topk]
            croplist = np.zeros([self.topk,self.crop_size[0],self.crop_size[1],self.crop_size[2]]).astype('float32')
            for k, cid in enumerate(chosenid):
                target=pbb[cid, 1:]
                crop = self.crop(tmp, target)
                crop = crop.astype(np.float32)
                if self.img_transform:
                    crop = self.img_transform(crop)
                croplist[k] = crop
            seq[t] = torch.from_numpy(croplist)
        
        # get label
        label = int(pid_row['lung_cancer'])

        # zero code embeddings
        code_emb = torch.zeros(size=(1, self.ehr_dim), dtype=torch.float32)

        return {"fnames":[f"{pid_row['id']}.npy"], "imgs": seq, "ehr": code_emb, "label": label}
    
    def __len__(self):
        return len(self.pids)

class LongNLSTFeatDataset(Dataset):
    """cross sectional NLST nodule features + zero code embeddings"""
    def __init__(self,  pids, feat_dir, labelf, feat_dim=64, time_range=(0,1), topk=5, img_transform=ToTensor(), ehr_dim=100):
        super().__init__()
        self.feat_dir = feat_dir
        self.pids = pids
        self.label = pd.read_csv(labelf, dtype={'pid':str})
        self.time_length = time_range[1] + 1
        self.feat_dim = feat_dim
        self.topk = topk
        self.img_transform = img_transform
        self.ehr_dim = ehr_dim

    def __getitem__(self, index):
        # returns (t c x y z)
        pid = self.pids[index]
        # returns descending in scan date
        pid_rows = self.label[self.label['pid']==pid].sort_values(by=['session'], ascending=False)
        
        # ROIs
        samplerows = samplerows.sort_values(by='session', ascending=False)
        seq = torch.zeros((self.topk, self.patch_size, self.patch_size, self.patch_size), dtype=torch.float32)
        fnames = samplerows['id'].tolist()
        pbb = None
        for t, fname in enumerate(fnames):
            tmp = np.load(os.path.join(self.prep_dir, f"{fname}_clean.npy"))
            
            # find ROI of the latest scan and extract same region from all previous
            if t==0:
                pbb = np.load(os.path.join(self.pbb_dir, f"{fname}_pbb.npy"))
                pbb = pbb[pbb[:,0]>-1]
                pbb = nms(pbb,0.05)
            
            # crop ROIs and if less than topk ROIs, leave rest of patches as zero
            conf_list = pbb[:,0]
            chosenid = conf_list.argsort()[::-1][:self.topk]
            croplist = np.zeros([self.topk,self.crop_size[0],self.crop_size[1],self.crop_size[2]]).astype('float32')
            for k, cid in enumerate(chosenid):
                target=pbb[cid, 1:]
                crop = self.crop(tmp, target)
                crop = crop.astype(np.float32)
                if self.img_transform:
                    crop = self.img_transform(crop)
                croplist[k] = crop
            seq[t] = torch.from_numpy(croplist)
        
        # relative time distances in descending order
        times = pid_rows['Duration'].tolist()
        times = [t-times[0] for t in times] # set latest scan t=0
        times = torch.Tensor(times)/30.33 # transform into fractional months
        
        # get label
        label = int(pid_rows.iloc[0]['lung_cancer'])

        # zero code embeddings
        code_emb = torch.zeros(size=(self.time_length, self.ehr_dim), dtype=torch.float32)

        return {"fnames": pid_rows['filename'].tolist(), "imgs": seq, "ehr": code_emb, "times": times, "label": label}
    
    def __len__(self):
        return len(self.pids)

class PaddedLongNLSTROIDataset(Dataset):
    """Returns time_length randomly sampled cross sections with padding if time points are missing (superset of LIVU)"""
    def __init__(self,  pids, prep_dir, labelf, pbb_dir, feat_dim=64, ehr_dim=2000, time_length=2, topk=5, patch_size=128, img_transform=None):
        super().__init__()
        self.prep_dir = prep_dir
        self.pbb_dir = pbb_dir
        self.pids = pids
        self.label = pd.read_csv(labelf, dtype={'pid':str})
        self.time_length = time_length
        self.feat_dim = feat_dim
        self.ehr_dim = ehr_dim
        self.topk = topk
        self.img_transform = img_transform
        self.patch_size = patch_size
        self.crop_size = (patch_size, patch_size, patch_size)
        self.crop = simpleCrop(self.crop_size)
    
    def __getitem__(self, index):
        pid = self.pids[index]
        # returns descending in scan date
        pid_rows = self.label[self.label['pid']==pid].sort_values(by=['session'], ascending=False)
        randidx = random.sample(range(len(pid_rows)), min(len(pid_rows), self.time_length))
        samplerows = pid_rows.iloc[randidx]
        samplerows['session_rank'] = samplerows['session'].rank(ascending=False).astype(int) - 1 # rank sessions

        # padding up to time_length. used to generate attention mask
        padding = np.zeros(self.time_length, dtype='float32')
        padding[:len(samplerows)] = 1 # 1 if value present, 0 if padding

        # ROIs
        samplerows = samplerows.sort_values(by='session', ascending=False)
        seq = torch.zeros((self.time_length, self.topk, self.patch_size, self.patch_size, self.patch_size), dtype=torch.float32)
        fnames = samplerows['id'].tolist()
        pbb = None
        for t, fname in enumerate(fnames):
            tmp = np.load(os.path.join(self.prep_dir, f"{fname}_clean.npy"))
            
            # find ROI of the latest scan and extract same region from all previous
            if t==0:
                pbb = np.load(os.path.join(self.pbb_dir, f"{fname}_pbb.npy"))
                pbb = pbb[pbb[:,0]>-1]
                pbb = nms(pbb,0.05)
            
            # crop ROIs and if less than topk ROIs, leave rest of patches as zero
            conf_list = pbb[:,0]
            chosenid = conf_list.argsort()[::-1][:self.topk]
            croplist = np.zeros([self.topk,self.crop_size[0],self.crop_size[1],self.crop_size[2]]).astype('float32')
            for k, cid in enumerate(chosenid):
                target=pbb[cid, 1:]
                crop = self.crop(tmp, target)
                crop = crop.astype(np.float32)
                if self.img_transform:
                    crop = self.img_transform(crop)
                croplist[k] = crop
            seq[t] = torch.from_numpy(croplist)

        # relative time distances in descending order
        times_padded = torch.zeros(self.time_length)
        times = samplerows['Duration'].tolist()
        times = [t-times[0] for t in times] # set latest scan t=0
        times = torch.Tensor(times)/30.33 # transform into fractional months
        times_padded[:len(samplerows)] = times

        # get label
        label = int(samplerows.iloc[0]['lung_cancer'])

        # fnames padded
        fnames_padded = ['' for i in range(self.time_length)]
        fnames_padded[:len(samplerows)] = samplerows['id'].tolist()

        # zero code embeddings
        ica = torch.zeros(size=(self.time_length, self.ehr_dim), dtype=torch.float32)

        return {"fnames": fnames_padded, "imgs": seq, "ehr": ica, "padding": padding, "times": times_padded, "label": label}
    
    def __len__(self):
        return len(self.pids)


class PaddedLongLiVU_ROI_ICADataset(Dataset):
    """Returns time_length randomly sampled cross sections with padding if time points are missing (superset of LIVU)"""
    def __init__(self,  pids, prep_dir, labelf, pbb_dir=None, feat_dim=64, ehr_dim=2000, time_length=2, topk=5, patch_size=128, img_transform=None):
        super().__init__()
        self.prep_dir = prep_dir
        self.pbb_dir = pbb_dir
        self.pids = pids
        self.label = pd.read_csv(labelf, dtype={'pid':str})
        self.time_length = time_length
        self.feat_dim = feat_dim
        self.ehr_dim = ehr_dim
        self.topk = topk
        self.img_transform = img_transform
        self.patch_size = patch_size
        self.crop_size = (patch_size, patch_size, patch_size)
        self.crop = simpleCrop(self.crop_size)
        
    def __getitem__(self, index):
        pid = self.pids[index]
        # returns descending in scan date
        pid_rows = self.label[self.label['pid']==pid].sort_values(by=['session'], ascending=False)
        randidx = random.sample(range(len(pid_rows)), min(len(pid_rows), self.time_length))
        samplerows = pid_rows.iloc[randidx]
        samplerows['session_rank'] = samplerows['session'].rank(ascending=False).astype(int) - 1 # rank sessions

        # padding up to time_length. used to generate attention mask
        padding = np.zeros(self.time_length, dtype='float32')
        padding[:len(samplerows)] = 1 # 1 if value present, 0 if padding

        # img features
        samplerows = samplerows.sort_values(by='session', ascending=False)
        seq = torch.zeros((self.time_length, self.topk, self.patch_size, self.patch_size, self.patch_size), dtype=torch.float32)
        fnames = samplerows['id'].tolist()
        pbb = None
        for t, fname in enumerate(fnames):
            tmp = np.load(os.path.join(self.prep_dir, f"{fname}_clean.npy"))
            
            # find ROI of the latest scan and extract same region from all previous
            if t==0:
                pbb = np.load(os.path.join(self.pbb_dir, f"{fname}_pbb.npy"))
                pbb = pbb[pbb[:,0]>-1]
                pbb = nms(pbb,0.05)
            
            # crop ROIs and if less than topk ROIs, leave rest of patches as zero
            conf_list = pbb[:,0]
            chosenid = conf_list.argsort()[::-1][:self.topk]
            croplist = np.zeros([self.topk,self.crop_size[0],self.crop_size[1],self.crop_size[2]]).astype('float32')
            for k, cid in enumerate(chosenid):
                target=pbb[cid, 1:]
                crop = self.crop(tmp, target)
                crop = crop.astype(np.float32)
                if self.img_transform:
                    crop = self.img_transform(crop)
                croplist[k] = crop
            seq[t] = torch.from_numpy(croplist)
        
        # get ICA expressions
        ica = np.zeros((self.time_length, self.ehr_dim), dtype='float32')
        ica[:len(samplerows)] = samplerows.iloc[:,-self.ehr_dim:].to_numpy(dtype='float32')
        ica = torch.from_numpy(ica).to(torch.float32)
        
        # relative time distances in descending order
        times_padded = torch.zeros(self.time_length)
        times = samplerows['Duration'].tolist()
        times = [t-times[0] for t in times] # set latest scan t=0
        times = torch.Tensor(times)/30.33 # transform into fractional months
        times_padded[:len(samplerows)] = times

        label = int(pid_rows.iloc[0]['lung_cancer'])

        fnames_padded = ['' for i in range(self.time_length)]
        fnames_padded[:len(samplerows)] = samplerows['id'].tolist()

        return {"fnames": fnames_padded, "imgs": seq, "ehr": ica, "padding": padding, "times": times_padded, "label": label}

    def __len__(self):
        return len(self.pids)


class simpleCrop():
    """Cropping algorithm from Liao https://github.com/lfz/DSB2017/blob/master/data_classifier.py#L110"""
    
    def __init__(self,crop_size, scaleLim=[0.85,1.15], radiusLim=[6,100], stride=4, jitter_range=0.15, filling_value=160, phase='train'):
        self.crop_size = crop_size
        self.scaleLim = scaleLim
        self.radiusLim = radiusLim
        self.stride = stride
        self.jitter_range = jitter_range
        self.filling_value = filling_value
        self.phase = phase
        
    def __call__(self,imgs,target):
        crop_size = np.array(self.crop_size).astype('int')
        if self.phase=='train':
            jitter_range = target[3]*self.jitter_range
            jitter = (np.random.rand(3)-0.5)*jitter_range
        else:
            jitter = 0
        start = (target[:3]- crop_size/2 + jitter).astype('int')
        pad = [[0,0]]
        for i in range(3):
            if start[i]<0:
                leftpad = -start[i]
                start[i] = 0
            else:
                leftpad = 0
            if start[i]+crop_size[i]>imgs.shape[i+1]:
                rightpad = start[i]+crop_size[i]-imgs.shape[i+1]
            else:
                rightpad = 0
            pad.append([leftpad,rightpad])
        imgs = np.pad(imgs,pad,'constant',constant_values =self.filling_value)
        crop = imgs[:,start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]

        return crop

if __name__ == "__main__":
    root_dir = "/home/local/VANDERBILT/litz/github/MASILab/LMcurves"
    labelf = "/home/local/VANDERBILT/litz/github/MASILab/LMcurves/data/nlst/nlst_2scan.csv"
    codef = os.path.join(root_dir, "data/livu/livu_codes_nolc.csv")
    embf = os.path.join(root_dir, "code_embedding/embeddings.json")

    feat_dir = "/home/local/VANDERBILT/litz/data/nlst/DeepLungScreening/prep"
    pbb_dir = "/home/local/VANDERBILT/litz/data/nlst/DeepLungScreening/bbox"
    label = pd.read_csv(labelf, dtype={'pid':str})
    pids = label['pid'].tolist()
    
    dataset =PaddedLongNLSTROIDataset(pids, feat_dir, labelf, pbb_dir, time_length=3)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for batch in loader:
        print(batch['imgs'].shape)

        print(batch['padding'].shape)
        print(batch['times'].shape)
        print(len(batch['fnames']))
        print(batch['label'].shape)
        print('----------')

    print('done')
    