import torch
from torch import nn
import torch.nn.functional as F

def _init(modules):
    for module in modules:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class ClassifierHead(nn.Module):
    def __init__(self, hparams):
        super(ClassifierHead, self).__init__()
        self.dense1 = nn.Linear(hparams['hidden_dim'], hparams['hidden_dim'])
        self.activation = nn.Tanh()
        self.dp = nn.Dropout(hparams['final_dp'])
        self.dense2 = nn.Linear(hparams['hidden_dim'], 1)
        _init(self.modules())

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dp(x)
        x = self.dense2(x)
        return x

class LstmEncoder(nn.Module):
    def __init__(self, hparams):
        super(LstmEncoder, self).__init__()
        self.hparams = hparams

        self.embs = nn.Embedding(hparams['vocab_size'], hparams['embs_dim'])
        self.embs_norm = nn.LayerNorm(hparams['embs_dim'])
        self.embs_dp = nn.Dropout(hparams['embs_dp'])
    
        self.lstm = nn.LSTM(input_size=hparams['embs_dim'],
                                  hidden_size=hparams['hidden_dim'])
      
        _init(self.modules())
    
    def forward(self, src):
        x = self.embs(src)
        x = self.embs_norm(x)
        x = self.embs_dp(x)
        
        x, _ = self.lstm(x)

        return x

class Classifier(nn.Module):
    def __init__(self, hparams):
        super(Classifier, self).__init__()
        self.hparams = hparams
        self.pad_id = hparams['pad_id']
        self.mask_id = hparams['mask_id']
        self.char_dp = hparams['char_dp']
        
        self.encoder = LstmEncoder(hparams)

        self.fc = ClassifierHead(hparams)
        
        _init(self.modules())
    
    def forward(self, src):
        if self.training:
            padding_mask = torch.eq(src, self.pad_id)
            probability_matrix = torch.full(src.shape, self.char_dp, device=src.device).masked_fill_(padding_mask, 0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            src = src.masked_fill_(masked_indices, self.mask_id)

        x = self.encoder(src)
        return self.fc(x[-1, :, :])