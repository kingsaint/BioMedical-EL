
import logging
import os
import glob
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import tqdm

from .utils_e2e_span import MODEL_CLASSES, get_args, get_comm_magic, get_model, load_and_cache_examples, save_checkpoint, set_seed


import horovod.torch as hvd
from sparkdl import HorovodRunner
import mlflow

from .utils_e2e_span import get_all_candidates, get_args, load_and_cache_examples, get_comm_magic, set_seed


logger = logging.getLogger(__name__)

def eval_hvd(args, model, tokenizer, prefix=""):
    mlflow.set_tracking_uri("databricks")
    os.environ['DATABRICKS_HOST'] = "https://trend-prod.cloud.databricks.com/"
    os.environ['DATABRICKS_TOKEN'] = args.db_token
    with mlflow.start_run(run_id = args.active_run_id):
        hvd.init()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
        comm = get_comm_magic()

        eval_dataset, (all_entities, all_entity_token_ids, all_entity_token_masks), \
        (all_document_ids, all_label_candidate_ids) = load_and_cache_examples(args, tokenizer)
        
        logger.info("Evaluation Dataset Created")
        if not os.path.exists(args.output_dir) and hvd.rank==0:
            os.makedirs(args.output_dir)
        comm.barrier()
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        
        eval_sampler = DistributedSampler(eval_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_train_batch_size)
        
        if args.use_all_candidates:
            all_candidate_embeddings = get_all_candidates(args, model, all_entity_token_ids, all_entity_token_masks)
        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        single_process_gold_path = os.path.join(args.output_dir,f'gold_{hvd.rank()}.csv')
        single_process_pred_path = os.path.join(args.output_dir,f'pred_{hvd.rank()}.csv')
        single_process_gold_file = open(single_process_gold_path, 'w+')
        single_process_pred_file = open(single_process_pred_path, 'w+')

        num_mention_processed = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            eval_one_batch(args, model, all_entities, all_document_ids, all_label_candidate_ids, all_candidate_embeddings, single_process_gold_file, single_process_pred_file, num_mention_processed, batch)
        comm.barrier()
        ##ONCE ALL BATCHES ARE FINISHED, COMBINE THEM INTO A SINGLE CSV USING THE ROOT NODE.
        if hvd.rank==0:
            for file_type in ["gold","pred"]:
                all_files = glob.glob(os.path.join(args.data_dir, "gold_*.csv"))
                df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
                df_merged = pd.concat(df_from_each_file, ignore_index=True)
                all_together_file_path= os.path.join(args.output_dir,f"{file_type}_ALL.csv")
                df_merged.to_csv(all_together_file_path)
        mlflow.log_artificats(args.output_dir)

def eval_one_batch(args, model, all_entities, all_document_ids, all_label_candidate_ids, all_candidate_embeddings, single_process_gold_file, single_process_pred_file, num_mention_processed, batch):
    batch = tuple(t.to(args.device) for t in batch)
    with torch.no_grad():
        doc_input = {"args": args,
                        "mention_token_ids": batch[0],
                        "mention_token_masks": batch[1],
                        "mode": 'ner',
                        }
        pred_mention_start_indices, pred_mention_end_indices, pred_mention_span_scores, last_hidden_states = model.forward(**doc_input)
        pred_mention_span_probs = torch.sigmoid(pred_mention_span_scores)
        spans_after_prunning = torch.where(pred_mention_span_probs >= args.gamma)
        if spans_after_prunning[0].size(0) <= 0:
            _, spans_after_prunning = torch.topk(pred_mention_span_probs, 8)

        mention_start_indices = pred_mention_start_indices[spans_after_prunning]
        mention_end_indices = pred_mention_end_indices[spans_after_prunning]
            
        if args.use_all_candidates:
            mention_inputs = {"args": args,
                                "last_hidden_states": last_hidden_states,
                                "mention_start_indices": mention_start_indices.unsqueeze(0),
                                # batch[7],  #overlapping_start_indices,
                                "mention_end_indices": mention_end_indices.unsqueeze(0),
                                # batch[8], # overlapping_end_indices,
                                "all_candidate_embeddings": all_candidate_embeddings,
                                "mode": 'ned',
                                }
        else:#does not work, --use_all_candidates must be set to true.
            mention_inputs = {"args": args,
                                "last_hidden_states": last_hidden_states,
                                "mention_start_indices": mention_start_indices.unsqueeze(0),
                                "mention_end_indices": mention_start_indices.unsqueeze(0),
                                "candidate_token_ids_1": batch[2],
                                "candidate_token_masks_1": batch[3],
                                "mode": 'ned',
                                }

        _, logits = model(**mention_inputs)
        logger.info(str(logits))
        preds = logits.detach().cpu().numpy()
            # out_label_ids = batch[6]
            # out_label_ids = out_label_ids.reshape(-1).detach().cpu().numpy()
            
        sorted_preds = np.flip(np.argsort(preds), axis=1)
        logger.info(str(sorted_preds))
        predicted_entities = []
        for i, sorted_pred in enumerate(sorted_preds):
            predicted_entity_idx = sorted_preds[i][0]
            predicted_entity = all_entities[predicted_entity_idx]
            predicted_entities.append(predicted_entity)

            # Write the gold entities
        num_mentions = batch[9].detach().cpu().numpy()[0]
        document_ids = all_document_ids[num_mention_processed:num_mention_processed + num_mentions]
        assert all(doc_id == document_ids[0] for doc_id in document_ids)
        gold_mention_start_indices = batch[7].detach().cpu().numpy()[0][:num_mentions]
        gold_mention_end_indices = batch[8].detach().cpu().numpy()[0][:num_mentions]
        gold_entities = all_label_candidate_ids[num_mention_processed:num_mention_processed + num_mentions]
        for j in range(num_mentions):
                # if gold_mention_start_indices[j] == gold_mention_end_indices[j]:
                #     gold_mention_end_indices[j] += 1
            if gold_mention_start_indices[j] > gold_mention_end_indices[j]:
                continue
            gold_write = document_ids[j] + '\t' + str(gold_mention_start_indices[j]) \
                            + '\t' + str(gold_mention_end_indices[j]) \
                            + '\t' + str(gold_entities[j]) \
                            + '\t' + str(1.0) \
                            + '\t' + 'NA' + '\n'
            single_process_gold_file.write(gold_write)

            # Write the predicted entities
        doc_id_processed = document_ids[0]
        num_pred_mentions = len(predicted_entities)
        mention_start_indices = mention_start_indices.detach().cpu().numpy()
        mention_end_indices = mention_end_indices.detach().cpu().numpy()
        mention_probs = pred_mention_span_probs[spans_after_prunning].detach().cpu().numpy()

        for j in range(num_pred_mentions):
                # if pred_mention_start_indices[j] == pred_mention_end_indices[j]:
                #     pred_mention_end_indices[j] += 1
            if pred_mention_start_indices[j] > pred_mention_end_indices[j]:
                continue
            pred_write = doc_id_processed + '\t' + str(mention_start_indices[j]) \
                            + '\t' + str(mention_end_indices[j]) \
                            + '\t' + str(predicted_entities[j]) \
                            + '\t' + str(mention_probs[j]) \
                            + '\t' + 'NA' + '\n'
            single_process_pred_file.write(pred_write)

        num_mention_processed += num_mentions

