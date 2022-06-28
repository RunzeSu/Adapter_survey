# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
from itertools import cycle
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import sys, os
sys.path.insert(1, '..')
from preprocess.preprocess import *
from preprocess import tokenization
from models.shared_model.modeling import BertConfig, BertForSequenceClassification, BertForMultiTask, BERTLayerNorm
from models.shared_model.optimization import BERTAdam
from adapters import AdapterController, AutoAdapterConfig
from adapters import MetaAdapterController, AdapterLayersHyperNetController, AdapterLayersOneHyperNetController
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"]="0"



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--multi",
                        default=False,
                        help="Whether to add adapter modules",
                        action='store_true')
    parser.add_argument("--hyperformer",
                        default=False,
                        help="Whether to add adapter modules",
                        action='store_true')
    parser.add_argument("--adapters",
                        default=None,
                        help="Defines a dictionary from adapters to the tasks.")
    parser.add_argument("--task_embeddings",
                        default=None,
                        help="Defines a dictionary from tasks to the tasks embeddings.")
    parser.add_argument("--adapter_config_name",
                        default="meta-adapter",
                        type=str,
                        help="config name for the adapter layers, should be selected.")
    parser.add_argument("--task_embedding_dim",
                        default=None,
                        type=int,
                        help="task embedding dimensions.")
    parser.add_argument("--add_layer_norm_before_adapter",
                        default=False,
                        help="whether to have layer-norm before adapter.",
                        action='store_true')
    parser.add_argument("--add_layer_norm_after_adapter",
                        default=False,
                        help="whether to have layer-norm after adapter.",
                        action='store_true')
    parser.add_argument("--hidden_dim",
                        default=128,
                        type=int,
                        help="defines the default hidden dimension for adapter layers.")
    parser.add_argument("--reduction_factor",
                        default=16,
                        type=int,
                        help="defines the default reduction factor for adapter layers.")
    parser.add_argument("--non_linearity",
                        default="swish",
                        type=str,
                        help="Defines nonlinearity for adapter layers.")
    parser.add_argument("--train_task_embeddings",
                        default=False,
                        help="If specified learns the tasks embeddings from given task seedings.",
                        action='store_true')
    parser.add_argument("--projected_task_embedding_dim",
                        default=64,
                        type=int,
                        help="Defines the task embedding dimension after projection layer.")
    parser.add_argument("--task_hidden_dim",
                        default=128,
                        type=int,
                        help="defines the hidden dimension for task embedding projector.")
    parser.add_argument("--conditional_layer_norm",
                        default=False,
                        help="Implements conditional layer norms modulated based on task embeddings.",
                        action='store_true')
    parser.add_argument("--train_adapters_blocks",
                        default=False,
                        help="If set, uses adapter blocks.",
                        action='store_true')
    parser.add_argument("--unique_hyper_net",
                        default=False,
                        help="If set, uses one hyper network to generates the adapter weights for all the layers.",
                        action='store_true')
    parser.add_argument("--efficient_unique_hyper_net",
                        default=False,
                        help="If set, uses one hyper network for all adapters in each layer.",
                        action='store_true')
    parser.add_argument("--unique_hyper_net_layer_norm",
                        default=False,
                        help="If set, applies a layer norm after computing the embeddings for the unique hyper-net.",
                        action='store_true')
    parser.add_argument("--unfreeze_layer_norms",
                        default=False,
                        help="unfreezes the layer norms.",
                        action='store_true')
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--optim",
                        default='normal',
                        help="Whether to split up the optimiser between adapters and not adapters.")
    parser.add_argument("--sample",
                        default='rr',
                        help="How to sample tasks, other options 'prop', 'sqrt' or 'anneal'")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--h_aug",
                        default="n/a",
                        help="Size of hidden state for adapters..")
    parser.add_argument("--tasks",
                        default="all",
                        help="Which set of tasks to train on.")
    parser.add_argument("--task_id",
                        default=1,
                        help="ID of single task to train on if using that setting.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--freeze",
                        default=False,
                        action='store_true',
                        help="Freeze base network weights")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")               
    args = parser.parse_args()
    print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "rte": RTEProcessor,
        "sts": STSProcessor,
        "sst": SSTProcessor,
        "qqp": QQPProcessor,
        "qnli": QNLIProcessor,
    }
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))
    
    print(device)
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    
    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    if args.tasks == 'inf':
        task_names =['qnli', 'mnli', 'rte']
        data_dirs = ['QNLI', 'MNLI', 'RTE']
    elif args.tasks == 'all':
        task_names =['cola', 'mrpc', 'mnli', 'rte', 'sts', 'sst', 'qqp', 'qnli']
        data_dirs = ['CoLA', 'MRPC', 'MNLI', 'RTE', 'STS-B', 'SST-2', 'QQP', 'QNLI']
    elif args.tasks == 'single':
        task_names = ['cola', 'mrpc', 'mnli', 'rte', 'sts', 'sst', 'qqp', 'qnli']
        data_dirs = ['CoLA', 'MRPC', 'MNLI', 'RTE', 'STS-B', 'SST-2', 'QQP', 'QNLI']
        task_names = [task_names[int(args.task_id)]]
        data_dirs = [data_dirs[int(args.task_id)]]
    if task_names[0] not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor_list = [processors[task_name]() for task_name in task_names]
    label_list = [processor.get_labels() for processor in processor_list]

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    num_tasks = len(task_names)
    if args.do_train:
        train_examples = [processor.get_train_examples(args.data_dir + data_dir) for processor, data_dir in zip(processor_list, data_dirs)]
        num_train_steps = int(
            len(train_examples[0]) / args.train_batch_size * args.num_train_epochs)
        if args.tasks == 'all':
            total_tr = 300 * num_tasks * args.num_train_epochs
        else:
            total_tr = int(0.5 * num_train_steps)

    if args.tasks == 'all':
        steps_per_epoch = args.gradient_accumulation_steps * 300 * num_tasks
    else:
        steps_per_epoch = int(num_train_steps/(2. * args.num_train_epochs))
    bert_config.num_tasks = num_tasks
    if args.h_aug is not 'n/a':
        bert_config.hidden_size_aug = int(args.h_aug)
        
    # Gets the adapter config and updates the specified parameters.
    if args.hyperformer:
        adapter_config = AutoAdapterConfig.get(args.adapter_config_name)
        adapter_config.input_dim = bert_config.hidden_size
        adapter_config.tasks = task_names
        adapter_config.task_to_adapter = {task:adapter for task, adapter in zip(adapter_config.tasks, args.adapters)} if args.adapters is not None else None
        # If this is a parametric task embedding this mapping makes sense, but in case we use any task embeddings,
        # then, we do not need any mapping as we use the pretrained task embeddings.
        adapter_config.task_to_embeddings = {task:embedding for task, embedding in zip(adapter_config.tasks, args.task_embeddings)}\
             if (args.task_embeddings is not None) else None
        extra_adapter_params = ("task_embedding_dim",
                                "add_layer_norm_before_adapter",
                                "add_layer_norm_after_adapter",
                                "reduction_factor",
                                "hidden_dim",
                                "non_linearity",
                                "train_task_embeddings",
                                "projected_task_embedding_dim",
                                "task_hidden_dim",
                                "conditional_layer_norm",
                                "train_adapters_blocks",
                                "unique_hyper_net",
                                "unique_hyper_net_layer_norm",
                                "efficient_unique_hyper_net")
        for p in extra_adapter_params:
            if hasattr(args, p) and hasattr(adapter_config, p):
                setattr(adapter_config, p, getattr(args, p))
            else:
                logger.warning(f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute")
        adapter_config.device = device
    else:
        adapter_config = None
     
    model = BertForMultiTask(bert_config, [len(labels) for labels in label_list], adapter_config = adapter_config)
    
    if args.init_checkpoint is not None:
        if args.multi:
            partial = torch.load(args.init_checkpoint, map_location='cpu')
            model_dict = model.bert.state_dict()
            update = {}
            for n, p in model_dict.items():
                if 'aug' in n or 'mult' in n:
                    update[n] = p
                    if 'pooler.mult' in n and 'bias' in n:
                        update[n] = partial['pooler.dense.bias']
                    if 'pooler.mult' in n and 'weight' in n:
                        update[n] = partial['pooler.dense.weight']
                else:
                    if 'task_emb' not in n:
                        update[n] = partial[n]
                    else:
                        update[n] = model_dict[n]
            model.bert.load_state_dict(update)
        elif args.hyperformer:
            partial = torch.load(args.init_checkpoint, map_location='cpu')
            model_dict = model.bert.state_dict()
            update = {}
            for n, p in model_dict.items():
                if 'adapter' in n:
                    update[n] = p
                else:
                    update[n] = partial[n]
            model.bert.load_state_dict(update)    
        else:
            model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
    
    if args.freeze:
        for n, p in model.bert.named_parameters():
            if 'aug' in n or 'classifier' in n or 'mult' in n or 'gamma' in n or 'beta' in n or "adapater" in n:
                continue
            p.requires_grad = False

    if args.hyperformer and bert_config.train_adapters:
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (MetaAdapterController, AdapterController)):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True
        if args.adapter_config_name == "meta-adapter":
            for param in model.task_embedding_controller.parameters():
                param.requires_grad = True
        if args.unique_hyper_net:
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, (AdapterLayersHyperNetController, AdapterController)):
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
        if args.efficient_unique_hyper_net:
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, (AdapterLayersOneHyperNetController)):
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True                       
        if args.unfreeze_layer_norms:
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, BERTLayerNorm):
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
    
    if args.optim == 'normal':
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
                ]
        optimizer = BERTAdam(optimizer_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=total_tr)
    else:
        no_decay = ['bias', 'gamma', 'beta']
        base = ['attn']
        optimizer_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in base)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in base)], 'weight_decay_rate': 0.0}
                ]
        optimizer = BERTAdam(optimizer_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=total_tr)
        optimizer_parameters_mult = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in base)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in base)], 'weight_decay_rate': 0.0}
                ]
        optimizer_mult = BERTAdam(optimizer_parameters_mult,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=total_tr)
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    if args.do_eval:
        eval_loaders = []
        for i, task in enumerate(task_names):
            eval_examples = processor_list[i].get_dev_examples(args.data_dir + data_dirs[i])
            eval_features = convert_examples_to_features(
                eval_examples, label_list[i], args.max_seq_length, tokenizer, task)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            if task != 'sts':
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            else:
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float32)
    
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

            if args.local_rank == -1:
                eval_sampler = SequentialSampler(eval_data)
            else:
                eval_sampler = DistributedSampler(eval_data)
            eval_loaders.append(DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size))

    global_step = 0
    if args.do_train:
        loaders = []
        logger.info("  Num Tasks = %d", len(train_examples))
        for i, task in enumerate(task_names):
            train_features = convert_examples_to_features(
                train_examples[i], label_list[i], args.max_seq_length, tokenizer, task)
            logger.info("***** training data for %s *****", task)
            logger.info("  Data size = %d", len(train_features))

            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            if task != 'sts':
                all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            else:
                all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float32)

            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            loaders.append(iter(DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)))
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("  Num param = {}".format(total_params))
        loaders = [cycle(it) for it in loaders]
        model.train()
        best_score = 0.       
        data_size ={'cola':6680, 'mrpc':2865, 'mnli':306798, 'rte':1945, 'sts':4491, 'sst':52616, 'qqp':284257, 'qnli':84715}
        probs = []
        for task in task_names:
            probs.append(data_size[task])
        if args.sample == 'sqrt' or args.sample == 'prop':
#             probs = [6680, 2865, 306798, 1945, 4491, 52616, 284257, 84715]
            if args.sample == 'prop':
                alpha = 1.
            if args.sample == 'sqrt':
                alpha = 0.5
            probs = [p**alpha for p in probs]
            tot = sum(probs)
            probs = [p/tot for p in probs]
        task_id = 0
        epoch = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            if args.sample == 'anneal':
#                 probs = [6680, 2865, 306798, 1945, 4491, 52616, 284257, 84715]
                alpha = 1. - 0.8 * epoch / (args.num_train_epochs - 1)
                probs = [p**alpha for p in probs]
                tot = sum(probs)
                probs = [p/tot for p in probs]

            tr_loss = [0. for i in range(num_tasks)]
            nb_tr_examples, nb_tr_steps = 0, 0
            for step in range(steps_per_epoch):
                if args.sample != 'rr':
                    if step % args.gradient_accumulation_steps == 0:
                        task_id = np.random.choice(len(probs), p=probs)
                else:
                    task_id = task_id % num_tasks
                batch = next(loaders[task_id])
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                print(input_ids, input_mask, segment_ids, label_ids, task_names[task_id], label_ids)
                loss, _ = model(input_ids, segment_ids, input_mask, task_id, task_names[task_id], label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss[task_id] += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if step % 1000 < num_tasks:
                    logger.info("Task: {}, Step: {}".format(task_id, step))
                    logger.info("Loss: {}".format(tr_loss[task_id]/nb_tr_steps))
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()    # We have accumulated enought gradients
                    if args.optim != 'normal':
                        optimizer_mult.step()
                    model.zero_grad()
                    global_step += 1
                    if not args.sample:
                        task_id += 1
            epoch += 1
            ev_acc = 0.
            for i, task in enumerate(task_names):
                ev_acc += do_eval(model, logger, args, device, tr_loss[i], nb_tr_steps, global_step, processor_list[i], 
                                  label_list[i], tokenizer, eval_loaders[i], task, i)
            logger.info("Total acc: {}".format(ev_acc))
            if ev_acc > best_score:
                best_score = ev_acc
                model_dir = os.path.join(args.output_dir, "best_model.pth")
                torch.save(model.state_dict(), model_dir)
            logger.info("Best Total acc: {}".format(best_score))

        ev_acc = 0.
        for i, task in enumerate(task_names):
            ev_acc += do_eval(model, logger, args, device, tr_loss[i], nb_tr_steps, global_step, processor_list[i], 
                              label_list[i], tokenizer, eval_loaders[i], task, i)
        logger.info("Total acc: {}".format(ev_acc))


if __name__ == "__main__":
    main()
