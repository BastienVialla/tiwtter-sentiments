
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import functools

class DomainDataset(Dataset):
    def __init__(self, file: str):
        super().__init__()
        df = pd.read_csv(file)
        self.labels = list(df['polarity'])
        texts = list(df['text'])
        self.texts = []
        for t in texts:
            if type(t) != str:
                continue
            self.texts.append(t)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {'label': self.labels[idx],
                'input': self.texts[idx]}

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.path = cfg['data_dir']
        self.tokenizer = tokenizer
        self.batch_size = cfg['batch_size']
        self.num_workers = cfg['num_workers']
    
    def _collate_fn(batch, tokenizer):
        enc = tokenizer([x['input'] for x in batch], padding=True, return_tensors='pt')['input_ids']
        inpt = enc.transpose(0, 1)
        y = torch.Tensor([x['label'] for x in batch])
        return inpt, y

    def setup(self, stage=None):
        self.ds_train = DomainDataset(self.path+'/train.csv')
        self.ds_valid = DomainDataset(self.path+'/valid.csv')

    def train_dataloader(self):
        return DataLoader(self.ds_train, 
                          batch_size=self.batch_size,
                          shuffle=self.cfg['train_shuffle'],
                          
                          num_workers=self.num_workers,
                          collate_fn=functools.partial(DataModule._collate_fn, 
                                                        tokenizer=self.tokenizer
                                                    )
                            )
    
    def val_dataloader(self):        
        return DataLoader(self.ds_valid, 
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=functools.partial(DataModule._collate_fn, 
                                                        tokenizer=self.tokenizer
                                                    )
                            )

