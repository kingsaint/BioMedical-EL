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

from .utils_e2e_span import MODEL_CLASSES, get_args, get_comm_magic, load_and_cache_examples, save_checkpoint, set_seed


from .modeling_e2e_span import DualEncoderBert, PreDualEncoder

import horovod.torch as hvd
from sparkdl import HorovodRunner
import mlflow


logger = logging.getLogger(__name__)



def train_hvd(args):
    """ Train the model """
    mlflow.set_tracking_uri("databricks")
    os.environ['DATABRICKS_HOST'] = "https://trend-prod.cloud.databricks.com/"
    os.environ['DATABRICKS_TOKEN'] = args.db_token
    with mlflow.start_run(run_id = args.active_run_id): 
        hvd.init() 
        os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
        comm = get_comm_magic()
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
                    

                
                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if hvd.rank() == 0:    
                mlflow.log_metrics({"averaged_training_loss_per_epoch":tr_loss_averaged_across_all_instances},epoch_num)
            #save checkpoint at end or after prescribed number of epochs
            if hvd.rank() == 0 and (epoch_num==args.num_train_epochs or epoch_num % args.save_epochs == 0):
                save_checkpoint(args,epoch_num,tokenizer,tokenizer_class,model,device,optimizer,scheduler)
            
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
            

    
def main(db_token,args=None):
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
    with mlflow.start_run() as run:
        for arg_name,arg_value in args.__dict__.items():
            mlflow.log_param(arg_name,arg_value)
        
        # Training
        if args.do_train:
            args.active_run_id = mlflow.active_run().info.run_id
            args.db_token = db_token
            hr = HorovodRunner(np=args.n_gpu,driver_log_verbosity='all') 
            hr.run(train_hvd, args=args)
        

if __name__ == "__main__":
    main()
    
  

