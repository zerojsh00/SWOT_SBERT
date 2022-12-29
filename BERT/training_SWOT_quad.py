import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import get_linear_schedule_with_warmup


from simple_ntc.bert_trainer import BertTrainer as Trainer
from simple_ntc.bert_dataset import TextClassificationDataset, TextClassificationCollator
from simple_ntc.utils import read_text

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--n_companies', type=int, required=True)

    # Recommended model list:
    # - kykim/bert-kor-base
    # - kykim/albert-kor-base
    # - beomi/kcbert-base
    # - beomi/kcbert-large
    p.add_argument('--pretrained_model_name', type=str, default='klue/bert-base')
    p.add_argument('--model_save_dir', type=str, default='./output')
    p.add_argument('--dataset_dir', type=str, default='../Dataset/SWOT_quad')

    p.add_argument('--use_albert', action='store_true')
    
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--n_epochs', type=int, default=7)

    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--warmup_ratio', type=float, default=.2) # At first, learn less.
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    # If you want to use RAdam, I recommend to use LR=1e-4.
    # Also, you can set warmup_ratio=0 since we use radam not to use warmup.
    p.add_argument('--use_radam', action='store_true')
    p.add_argument('--valid_ratio', type=float, default=.2)

    p.add_argument('--max_length', type=int, default=512)

    config = p.parse_args()

    return config


def get_loaders(fn, fn_v, tokenizer, valid_ratio=.2):
    # Get list of labels and list of texts.
    labels, texts = read_text(fn) # TRAIN DATASETS
    labels_v, texts_v = read_text(fn_v) # TRAIN DATASETS


    # Generate label to index map.
    unique_labels = list(set(labels))
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label

    # Convert label text to integer value.
    labels = list(map(label_to_index.get, labels))
    labels_v = list(map(label_to_index.get, labels_v))

    # Get dataloaders using given tokenizer as collate_fn.
    train_loader = DataLoader(
        TextClassificationDataset(texts[:], labels[:]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )
    valid_loader = DataLoader(
        TextClassificationDataset(texts_v[:], labels_v[:]),
        batch_size=config.batch_size,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )

    return train_loader, valid_loader, index_to_label


def get_optimizer(model, config):

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.lr,
        eps=config.adam_epsilon
    )

    return optimizer


def main(config):
    N_COMPANIES = config.n_companies
    MODEL_SAVE_DIR = config.model_save_dir
    DATASET_DIR = config.dataset_dir
    MODEL_SAVE_PATH = '{}/SWOT_BERT_{}.pt'.format(MODEL_SAVE_DIR, str(N_COMPANIES).zfill(4))
    TRAIN_DATASET = '{}/{}/train.tsv'.format(DATASET_DIR, str(N_COMPANIES).zfill(4))
    VALID_DATASET = '{}/{}/valid.tsv'.format(DATASET_DIR, str(N_COMPANIES).zfill(4))

    # Get pretrained tokenizer for BPE(Byte Pair Encoding).
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)
    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader, index_to_label = get_loaders(
        TRAIN_DATASET,
        VALID_DATASET,
        tokenizer,
        valid_ratio=config.valid_ratio
    )

    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    # Get pretrained model with specified softmax layer.
    model_loader = AlbertForSequenceClassification if config.use_albert else BertForSequenceClassification
    model = model_loader.from_pretrained(
        config.pretrained_model_name,
        num_labels=len(index_to_label)
    )
    optimizer = get_optimizer(model, config)

    # By default, model returns a hidden representation before softmax func.
    # Thus, we need to use CrossEntropyLoss, which combines LogSoftmax and NLLLoss.
    crit = nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    # Start train.
    trainer = Trainer(config)
    model = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
    )

    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, MODEL_SAVE_PATH)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
