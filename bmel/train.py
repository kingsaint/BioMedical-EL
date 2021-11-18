# coding=utf-8

""" Finetuning BioBERT models on MedMentions.
    Adapted from HuggingFace `examples/run_glue.py`"""

import logging
import os
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

from .utils_e2e_span import get_base_model, get_comm_magic, load_and_cache_examples, save_checkpoint, set_seed

import horovod.torch as hvd
from sparkdl import HorovodRunner
import mlflow


logger = logging.getLogger(__name__)

TRAINING_ARGS = {   "lambda_1",
                    "lambda_2",
                    "weight_decay",
                    "learning_rate",
                    "adam_epsilon",
                    "max_grad_norm",
                    "num_train_epochs"
                    "n_gpu",
                    "max_mention_length",
                    "max_seq_length",
                    "gradient_accumulation_steps"
                    "per_gpu_train_batch_size",
                    "num_candidates",
                    "num_max_mentions",
                    "max_steps",
                    "use_tfidf_candidates",
                    "use_random_candidates",
                    "use_hard_negatives",
                    "use_hard_and_random_negatives",
                    "ner",
                    "ner_and_ned",
                    "seed"
                }   

def train_hvd(args):
    """ Train the model """
    mlflow.set_tracking_uri("databricks")
    os.environ['DATABRICKS_HOST'] = "https://trend-prod.cloud.databricks.com/"
    os.environ['DATABRICKS_TOKEN'] = args.db_token
    with mlflow.start_run(run_id = args.active_run_id): 
        hvd.init() 
        os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
        comm = get_comm_magic()
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if hvd.rank() in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            hvd.rank(),
            args.device,
            args.n_gpu,
            bool(hvd.rank() != -1),
            args.fp16,
        )
        logger.info("Training/evaluation parameters %s", args)
        # Load pretrained model and tokenizer
        

        tokenizer_class, tokenizer, model = get_base_model(args)

        
            
        if args.device.type == 'cuda':
         # Pin GPU to local rank
            torch.cuda.set_device(0)
            #torch.cuda.set_device(hvd.local_rank())
        
        model.to(args.device)  
        
        if args.use_random_candidates:
            train_dataset, _, _= load_and_cache_examples(args, tokenizer)
        elif args.use_hard_negatives or args.use_hard_and_random_negatives:
            train_dataset, _, _ = load_and_cache_examples(args,tokenizer, model)
        else:
            train_dataset, _, _ = load_and_cache_examples(args, tokenizer)

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size)
        
        logger.info("***** Running training *****")
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate * hvd.size(), eps=args.adam_epsilon)
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
        
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=hvd.rank()!=0)
        set_seed(args)  # Added here for reproductibility
        
        for epoch_num in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=hvd.rank()!=0)
            train_one_epoch(args, tokenizer_class, tokenizer, model, optimizer, scheduler, global_step, epoch_num, epoch_iterator)
            if args.use_random_candidates:
                # New data loader at every epoch for random sampler if we use random negative samples
                train_dataset, _, _= load_and_cache_examples(args, tokenizer)
                train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
                train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size)
            elif args.use_hard_negatives or args.use_hard_and_random_negatives:
                # New data loader at every epoch for hard negative sampler if we use hard negative mining
                train_dataset, _, _= load_and_cache_examples(args, tokenizer, model)
                train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
                train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size)
            # Anneal the lamba_1 and lambda_2 weights
            args.lambda_1 = args.lambda_1 - 1 / (epoch_num + 1)
            args.lambda_2 = args.lambda_2 + 1 / (epoch_num + 1)

def train_one_epoch(args, tokenizer_class, tokenizer, model, optimizer, scheduler, global_step, epoch_num, epoch_iterator):
    for step, batch in enumerate(epoch_iterator):
        tr_loss_averaged_across_all_instances = train_one_batch(args, model, optimizer, scheduler, global_step, step, batch)
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
    if hvd.rank() == 0:    
        mlflow.log_metrics({"averaged_training_loss_per_epoch":tr_loss_averaged_across_all_instances},epoch_num)
            #save checkpoint at end or after prescribed number of epochs
        if (epoch_num==args.num_train_epochs or epoch_num % args.save_epochs == 0):
            save_checkpoint(args,epoch_num,tokenizer,tokenizer_class,model,optimizer,scheduler)
            
            # New data loader for the next epoch


def train_one_batch(args,  model, optimizer, scheduler, global_step, step, batch):
    model.train()
    ner_inputs, ned_inputs = get_inputs(args, model, batch)
    if args.ner:
        loss, _ = model.forward(**ner_inputs)
    elif args.alternate_batch:
                    # Randomly choose whether to do tagging or NED for the current batch
        if random.random() <= 0.5:
            loss = model.forward(**ner_inputs)
        else:
            loss, _ = model.forward(**ned_inputs)
    elif args.ner_and_ned:
        ner_loss, last_hidden_states = model.forward(**ner_inputs)
        ned_inputs["last_hidden_states"] = last_hidden_states
        ned_loss, _ = model.forward(**ned_inputs)
        loss = ner_loss + ned_loss
    else:
        logger.info(" Specify a training protocol from (ner, alternate_batch, ner_and_ned)")


    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    else:
        loss.backward()
    if hvd.rank() == 0:
        tr_loss_averaged_across_all_instances = hvd.allreduce(loss).item()
        mlflow.log_metrics({"averaged_training_loss_per_step":tr_loss_averaged_across_all_instances},step)

    if (step + 1) % args.gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1
    return tr_loss_averaged_across_all_instances

def get_inputs(args, batch):
    batch = tuple(t.to(args.device) for t in batch)
    ner_inputs = {"args": args,
                                "mention_token_ids": batch[0],
                                "mention_token_masks": batch[1],
                                "mention_start_indices": batch[7],
                                "mention_end_indices": batch[8],
                                "mode": 'ner'
                                }

    if args.use_hard_and_random_negatives:
        ned_inputs = {"args": args,
                                    "last_hidden_states": None,
                                    "mention_start_indices": batch[7],
                                    "mention_end_indices": batch[8],
                                    "candidate_token_ids_1": batch[2],
                                    "candidate_token_masks_1": batch[3],
                                    "candidate_token_ids_2": batch[4],
                                    "candidate_token_masks_2": batch[5],
                                    "labels": batch[6],
                                    "mode": 'ned'
                                    }
    else:
        ned_inputs = {"args": args,
                                    "mention_token_ids": batch[0],
                                    "mention_token_masks": batch[1],
                                    "mention_start_indices": batch[7],
                                    "mention_end_indices": batch[8],
                                    "candidate_token_ids_1": batch[2],
                                    "candidate_token_masks_1": batch[3],
                                    "labels": batch[6],
                                    "mode": 'ned'
                                    }
                        
    return ner_inputs,ned_inputs


        

    
  

