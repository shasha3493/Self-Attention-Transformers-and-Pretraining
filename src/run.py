import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import argparse
random.seed(0)

import dataset
import model
import trainer
import utils


argp = argparse.ArgumentParser()
argp.add_argument('function', help="Choose pretrain, finetune, or evaluate")
argp.add_argument('variant', help="Choose vanilla or perceiver") 
argp.add_argument('--bottleneck_dim', type=int, default=32)
argp.add_argument('pretrain_corpus_path', default=None)
argp.add_argument('--reading_params_path',default=None)
argp.add_argument('--writing_params_path',default=None)
argp.add_argument('--finetune_corpus_path', default=None)
argp.add_argument('--eval_corpus_path', default=None)
argp.add_argument('--outputs_path', default=None)
argp.add_argument('--pretrain_lr', default=6e-3, type=float)
argp.add_argument('--finetune_lr', default=6e-4, type=float)
argp.add_argument('--tb_expt_name', help='debug string for tb log.',
                  default='run')
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# TensorBoard training log
writer = SummaryWriter(log_dir='expt/%s/%s_%s_%d_pt_lr_%f_ft_lr_%f' % (
    args.function,
    args.tb_expt_name,
    args.variant,
    args.bottleneck_dim,
    args.pretrain_lr,
    args.finetune_lr))

# Keep the block size 128
# Why is the pretraining corpus always required (even if we're not pretraining?)
# It's because we're using it as a hack to always have the same vocabulary
# (that is, the same mapping from character to integer, and we build the
# vocab from the pretraining corpus.)
block_size = 128
text = open(args.pretrain_corpus_path, encoding='utf-8').read()
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size) # builds the vocabulary from the pretrainig corpus and assigns id 

# We don't suggest you change these hyperparameters, as they're known to work.
# use them for both the vanilla and the perceiver models
mconf = model.GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
    n_layer=4, n_head=8, n_embd=256) # configuration for vanilla GPT network

"""
Don't change above here; write your code below
"""

# define models.
# note: models should moved to device defined on line 34.

if args.variant == 'vanilla':
    model = model.GPT(mconf).to(device) # Make some model here
elif args.variant == 'perceiver':
    # set mconf.perceiver, and mconf.bottleneck_dim parameters appropriately.
    mconf.perceiver = True
    mconf.bottleneck_dim = args.bottleneck_dim
    model = model.GPT(mconf).to(device) # Make some model here
else:
    raise ValueError("Unknown model variant")

# Perform pretraining, finetuning, or evaluation
if args.function == 'pretrain':
    assert args.writing_params_path is not None
    # - Given:
    #     1. A corpus specified in args.pretrain_corpus_path
    #     2. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. Pretrain the model on this corpus
    #     2. Save the resulting model in args.writing_params_path
    
    # - Make sure to use the following hyperparameters for pretraining:
    # Hyperparameters for pretraining:
    # max_epochs=650
    # batch_size=128
    # learning_rate=args.pretrain_lr
    # lr_decay=True
    # warmup_tokens=512*20
    # final_tokens=200*len(pretrain_dataset)*block_size
    # num_workers=4
    # writer=writer 
    
    assert args.pretrain_corpus_path is not None
    
    trainer_config = trainer.TrainerConfig(
            max_epochs=650,
            batch_size=128,
            learning_rate=args.pretrain_lr,
            lr_decay=True,
            warmup_tokens=512*20,
            final_tokens=200*len(pretrain_dataset)*block_size,
            num_workers=0,
            writer=writer
         )
    
    
    trainer = trainer.Trainer(model, pretrain_dataset, None, trainer_config)
    trainer.train()
    
    torch.save(model.state_dict(), args.writing_params_path)
    
elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None
    # - Given:
    #     1. A finetuning corpus specified in args.finetune_corpus_path
    #     2. A path args.reading_params_path containing pretrained model
    #         parameters, or None if finetuning without a pretrained model
    #     3. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. If args.reading_params_path is specified, load these parameters
    #         into the model
    #     2. Finetune the model on this corpus
    #     3. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters:
    #     Hyperparameters for finetuning WITHOUT a pretrained model:
    #         max_epochs=75
    #         batch_size=256
    #         learning_rate=args.finetune_lr
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #         writer=writer
    #     Hyperparameters for finetuning WITH a pretrained model:
    #         max_epochs=10
    #         batch_size=256
    #         learning_rate=args.finetune_lr
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #         writer=writer
    
    
    
    # different sets of hyperparameters depending on whether or not we use pretrained weights
    if args.reading_params_path is not None:
        model.load_state_dict(torch.load(args.reading_params_path))
        trainer_config = trainer.TrainerConfig(
            max_epochs=10,
            batch_size=256,
            learning_rate=args.finetune_lr,
            lr_decay=True,
            warmup_tokens=512*20,
            final_tokens=200*len(pretrain_dataset)*block_size,
            num_workers=0,
            writer=writer
        )
    else:
         trainer_config = trainer.TrainerConfig(
            max_epochs=75,
            batch_size=256,
            learning_rate=args.finetune_lr,
            lr_decay=True,
            warmup_tokens=512*20,
            final_tokens=200*len(pretrain_dataset)*block_size,
            num_workers=0,
            writer=writer
         )
    
    
    finetune_dataset = dataset.NameDataset(pretrain_dataset, open(args.finetune_corpus_path, encoding='utf-8').read()) # initialize fine tune dataset
    
    trainer = trainer.Trainer(model, finetune_dataset, None, trainer_config)
    trainer.train()
    
    torch.save(model.state_dict(), args.writing_params_path)
    
elif args.function == 'evaluate':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    model.load_state_dict(torch.load(args.reading_params_path))
    correct = 0
    total = 0
    # take val data and get the predictions
    with open(args.outputs_path, 'w', encoding='utf-8') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path, encoding='utf-8')):
            x = line.split('\t')[0]
            x = x + '⁇' # eg, Where was Bryan Dubreuiel born?⁇
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None,...].to(device) # convert character to token
            pred = utils.sample(model, x, 32, sample=False)[0] # generate next 32 characters
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred]) # join all the characters
            pred = completion.split('⁇')[1] # split by  mask token, we trained the model to generate answer between the first and next mask token
            predictions.append(pred)
            fout.write(pred + '\n') # write the prediction in th eoutput file
        total, correct = utils.evaluate_places(args.eval_corpus_path, predictions) # compare gt and predicted answers
    if total > 0:
      print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
    else:
        print('Predictions written to {}; no targets provided'
                .format(args.outputs_path))

