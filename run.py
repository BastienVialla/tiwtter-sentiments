import os
from argparse import ArgumentParser
import yaml
import importlib
import torch
import pytorch_lightning as pl

from transformers import PreTrainedTokenizerFast

def cli_main(cfg):

    tokenizer = PreTrainedTokenizerFast.from_pretrained('./tokenizers/gpt2_2k')
    assert cfg['tokenizer']['vocab_size'] == len(tokenizer.get_vocab())

    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})

    cfg['model']['architecture']['pad_id'] = tokenizer.vocab['[PAD]']
    cfg['model']['architecture']['mask_id'] = tokenizer.vocab['[MASK]']

    mod_datamodule_ = importlib.import_module(
        f'src.datamodules.{cfg["datamodule_name"]}')
    datamodule: pl.LightningDataModule = mod_datamodule_.DataModule(
        cfg['datamodule'], tokenizer)

    mod_models_ = importlib.import_module(
        f'src.models.{cfg["pl_model_name"]}')
    model_class_ = getattr(mod_models_, cfg['pl_model_name'])
    model: pl.LightningModule = model_class_(cfg['model'])

    callbacks = []
    early_stopping = pl.callbacks.EarlyStopping(
        monitor=cfg['callbacks']['early_stopping']['monitor'],
        min_delta=cfg['callbacks']['early_stopping']['min_delta'],
        patience=cfg['callbacks']['early_stopping']['patience'])
    callbacks.append(early_stopping)

    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_logger)

    checkpoint = False
    # if 'checkpoint' in cfg:
    checkpoint = pl.callbacks.ModelCheckpoint(filename=cfg['callbacks']['checkpoint']['filename'],
                                              save_top_k=cfg['callbacks']['checkpoint']['save_top_k'],
                                              verbose=cfg['callbacks']['checkpoint']['verbose'],
                                              monitor=cfg['callbacks']['checkpoint']['monitor'],
                                              mode=cfg['callbacks']['checkpoint']['mode'])

    logger = None
    # if logger in cfg:
    logger = pl.loggers.TensorBoardLogger(cfg['callbacks']['tensorboard_logging']['dir'],
                                          name=cfg['callbacks']['tensorboard_logging']['name'])

    trainer = pl.Trainer(gpus=cfg['trainer']['gpus'],
                         gradient_clip_val=cfg['trainer']['gradient_clip_val'],
                         stochastic_weight_avg=cfg['trainer']['stochastic_weight_avg'],
                         callbacks=callbacks,
                         checkpoint_callback=checkpoint,
                         precision=cfg['trainer']['precision'],
                         check_val_every_n_epoch=cfg['trainer']['check_val_every_n_epoch'],
                         logger=logger)
    trainer.fit(model, datamodule)
    #trainer.checkpoint_callback.best_model_path
    model = model_class_.load_from_checkpoint(checkpoint_path=trainer.checkpoint_callback.best_model_path, hparams=cfg['model'])
    torch.save(model.model.state_dict(), f'./models/{cfg["model_name"]}.pt')
    torch.save(model.model.encoder.state_dict(), f'./models/{cfg["model_name"]}_enc.pt')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', help='Configuration file',
                        required=True, type=str)
    args = parser.parse_args()
    with open(f'{args.config}', 'r') as in_file:
        cfg = yaml.load(in_file, Loader=yaml.FullLoader)
    cli_main(cfg)
