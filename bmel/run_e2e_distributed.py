# coding=utf-8

""" Finetuning BioBERT models on MedMentions.
    Adapted from HuggingFace `examples/run_glue.py`"""

import argparse
import glob
import logging
import os
import random
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import pdb

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    # BertConfig,
    # BertForSequenceClassification,
    # BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    get_linear_schedule_with_warmup,
)

from .utils_e2e_span import get_examples, convert_examples_to_features

from .modeling_bert import BertModel
from .tokenization_bert import BertTokenizer
from .configuration_bert import BertConfig
from .modeling_e2e_span import DualEncoderBert, PreDualEncoder

import horovod.torch as hvd
from sparkdl import HorovodRunner
import mlflow

from mpi4py import MPI
comm = None

def get_comm_magic():
    global comm
    if comm is None:
      comm = MPI.COMM_WORLD
    return comm
  

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in [BertConfig]), ()
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
}



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train_hvd(args):
    """ Train the model """
    hvd.init() 
    os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
    comm == get_comm_magic()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if hvd.rank() in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        hvd.rank(),
        device,
        args.n_gpu,
        bool(hvd.rank() != -1),
        args.fp16,
    )
    logger.info("Training/evaluation parameters %s", args)
    # Load pretrained model and tokenizer
    
    if hvd.rank()!=0:
        comm.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    pretrained_bert = PreDualEncoder.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # Add new special tokens '[Ms]' and '[Me]' to tag mention
    new_tokens = ['[Ms]', '[Me]']
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    pretrained_bert.resize_token_embeddings(len(tokenizer))

    model = DualEncoderBert(config, pretrained_bert)

    if hvd.rank()==0:
        comm.barrier()  # Make sure only the first process in distributed training will download model & vocab
        
    if device.type == 'cuda':
      # Pin GPU to local rank
      torch.cuda.set_device(0)
      #torch.cuda.set_device(hvd.local_rank())
      
    model.to(device)  
    
    if args.use_random_candidates:
        train_dataset, _, _= load_and_cache_examples(args, tokenizer)
    elif args.use_hard_negatives or args.use_hard_and_random_negatives:
        train_dataset, _, _ = load_and_cache_examples(args, tokenizer, model)
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
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=hvd.rank()!=0
    )
    set_seed(args)  # Added here for reproductibility
    
    for epoch_num in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=hvd.rank()!=0)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)


            ner_inputs = {"args": args,
                            "mention_token_ids": batch[0],
                            "mention_token_masks": batch[1],
                            "mention_start_indices": batch[7],
                            "mention_end_indices": batch[8],
                            "mode": 'ner',
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
                                "mode": 'ned',
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
                                "mode": 'ned',
                                }
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

            tr_loss_averaged_across_all_instances = hvd.allreduce(loss).item()

            mlflow.log_metrics({"averaged_training_loss_per_step":tr_loss_averaged_across_all_instances},step)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                

            
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        #save checkpoint at end or after prescribed number of epochs
        if hvd.rank() == 0 and (epoch_num==args.num_train_epochs or epoch_num % args.save_epochs == 0):
            save_checkpoint(args,epoch_num,tokenizer,tokenizer_class,model,device,optimizer,scheduler)
            
        mlflow.log_metrics({"averaged_training_loss_per_epoch":tr_loss_averaged_across_all_instances},epoch_num)
      # New data loader for the next epoch
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
    
    
    return global_step, tr_loss / global_step
  
def load_and_cache_examples(args, tokenizer, model=None):
    if hvd.rank() not in [-1, 0]:
        comm.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    mode = 'train' if args.do_train else 'test'
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}".format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop()),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        all_entities = np.load(os.path.join(args.data_dir, 'all_entities.npy'))
        all_entity_token_ids = np.load(os.path.join(args.data_dir, 'all_entity_token_ids.npy'))
        all_entity_token_masks = np.load(os.path.join(args.data_dir, 'all_entity_token_masks.npy'))
        all_document_ids = np.load(os.path.join(args.data_dir, 'all_document_ids.npy'))
        all_label_candidate_ids = np.load(os.path.join(args.data_dir, 'all_label_candidate_ids.npy'))
        logger.info("Finished loading features from cached file %s", cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples, docs, entities = get_examples(args.data_dir, mode)
        features, (all_entities, all_entity_token_ids, all_entity_token_masks), (all_document_ids, all_label_candidate_ids) = convert_examples_to_features(
            examples,
            docs,
            entities,
            args.max_seq_length,
            tokenizer,
            args,
            model,
        )
        if hvd.rank() in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            np.save(os.path.join(args.data_dir, 'all_entities.npy'),
                        np.array(all_entities))
            np.save(os.path.join(args.data_dir, 'all_entity_token_ids.npy'),
                    np.array(all_entity_token_ids))
            np.save(os.path.join(args.data_dir, 'all_entity_token_masks.npy'),
                    np.array(all_entity_token_masks))
            np.save(os.path.join(args.data_dir, 'all_document_ids.npy'),
                    np.array(all_document_ids))
            np.save(os.path.join(args.data_dir, 'all_label_candidate_ids.npy'),
                    np.array(all_label_candidate_ids))

    if hvd.rank() == 0:
        comm.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_mention_token_ids = torch.tensor([f.mention_token_ids for f in features], dtype=torch.long)
    
    all_mention_token_masks = torch.tensor([f.mention_token_masks for f in features], dtype=torch.long)
    all_candidate_token_ids_1 = torch.tensor([f.candidate_token_ids_1 if f.candidate_token_ids_1 is not None else [0] for f in features], dtype=torch.long)
    all_candidate_token_masks_1 = torch.tensor([f.candidate_token_masks_1 if f.candidate_token_masks_1 is not None else [0] for f in features], dtype=torch.long)
    all_candidate_token_ids_2 = torch.tensor([f.candidate_token_ids_2 if f.candidate_token_ids_2 is not None else [0] for f in features], dtype=torch.long)
    all_candidate_token_masks_2 = torch.tensor([f.candidate_token_masks_2 if f.candidate_token_masks_2 is not None else [0] for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    #print([len(f.mention_end_indices) for f in features])
    all_mention_start_indices = torch.tensor([f.mention_start_indices for f in features], dtype=torch.long)
    all_mention_end_indices = torch.tensor([f.mention_end_indices for f in features], dtype=torch.long)
    all_num_mentions = torch.tensor([f.num_mentions for f in features], dtype=torch.long)
    all_seq_tag_ids = torch.tensor([f.seq_tag_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_mention_token_ids,
                            all_mention_token_masks,
                            all_candidate_token_ids_1,
                            all_candidate_token_masks_1,
                            all_candidate_token_ids_2,
                            all_candidate_token_masks_2,
                            all_labels,
                            all_mention_start_indices,
                            all_mention_end_indices,
                            all_num_mentions,
                            all_seq_tag_ids,
                            )
    return dataset, (all_entities, all_entity_token_ids, all_entity_token_masks), (all_document_ids, all_label_candidate_ids)

def save_checkpoint(args,epoch_num,tokenizer,tokenizer_class,model,device,optimizer,scheduler):
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # Create output directory if needed
    training_run_dir = os.path.join(args.output_dir,"training_run_{}GPUs_{}epochs".format(args.gpu,epoch_num))
    if args.num_train_epochs == epoch_num:
        final = True
    if final:
        output_dir = os.path.join(training_run_dir, "checkpoint-{}-FINAL".format(epoch_num))
    else:
        output_dir = os.path.join(training_run_dir, "checkpoint-{}".format(epoch_num))

    logger.info("Saving model checkpoint to %s", output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training

    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)
    
    # Load a trained model and vocabulary that you have fine-tuned to ensure proper
    if final:
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        tokenizer = tokenizer_class.from_pretrained(output_dir)
        model.to(device)
        
    logger.info("Saved model checkpoint to %s", output_dir)
            

    
def main(args=None):
    args = get_args(args)
    
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    mlflow.set_experiment(args.experiment_name)
    # Set seed
    set_seed(args)
    with mlflow.start_run():
        for arg_name,arg_value in args.__dict__.items():
            mlflow.log_param(arg_name,arg_value)


        # Training
        if args.do_train:
            hr = HorovodRunner(np=args.n_gpu,driver_log_verbosity='all') 
            hr.run(train_hvd, args=args)
        

  
def get_args(dict_args = None):

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--resume_path",
        default=None,
        type=str,
        required=False,
        help="Path to the checkpoint from where the training should resume"
    )
    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_mention_length",
        default=20,
        type=int,
        help="Maximum length of a mention span"
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", default=False, help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_epochs", type=int, default=1, help="Save checkpoint every X epochs.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--use_random_candidates", action="store_true", help="Use random negative candidates during training"
    )
    parser.add_argument(
        "--use_tfidf_candidates", action="store_true", help="Use random negative candidates during training"
    )
    parser.add_argument(
        "--use_hard_negatives",  action="store_true", help="Use hard negative candidates during training"
    )
    parser.add_argument(
        "--use_hard_and_random_negatives", action="store_true", help="Use hard negative candidates during training"
    )
    parser.add_argument(
        "--include_positive", action="store_true", help="Includes the positive candidate during inference"
    )
    parser.add_argument(
        "--use_all_candidates", action="store_true", help="Use all entities as candidates"
    )
    parser.add_argument(
        "--num_candidates", type=int, default=10, help="Number of candidates to consider per mention"
    )
    parser.add_argument(
        "--num_max_mentions", type=int, default=8, help="Maximum number of mentions in a document"
    )
    parser.add_argument(
        "--ner", type=bool, default=False, help="Model will perform only BIO tagging"
    )
    parser.add_argument(
        "--alternate_batch", type=bool, default=False, help="Model will perform either BIO tagging or entity linking per batch during training"
    )
    parser.add_argument(
        "--ner_and_ned", type=bool, default=True, help="Model will perform both BIO tagging and entity linking per batch during training"
    )
    parser.add_argument(
        "--gamma", type=float, default=0, help="Threshold for mention candidate prunning"
    )
    parser.add_argument(
        "--lambda_1", type=float, default=1, help="Weight of the random candidate loss"
    )
    parser.add_argument(
        "--lambda_2", type=float, default=0, help="Weight of the hard negative candidate loss"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("---mflow_experiment", type=str, default="", help="To log parameters and metrics.")
    list_args = []
    if dict_args != None:
      for key,value in dict_args.items():
        if value =="True":
          list_args.append("--"+key)
        else:
          list_args.append("--"+key)
          list_args.append(value)
      args = parser.parse_args(list_args)
    else:
      args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
    
  

