
import logging
import os
import glob
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
import torch
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from neleval.__main__ import APPS as neleval_apps

from .utils_e2e_span import  get_comm_magic, get_trained_model, load_and_cache_examples


import horovod.torch as hvd
import mlflow

from .utils_e2e_span import load_and_cache_examples, get_comm_magic

EVALUATOR_CLASS = neleval_apps[0]
MEASURES={'overlap-maxmax::span+kbid':"overlapping",
           'strong_all_match':"strict"
}
logger = logging.getLogger(__name__)
##DISTRIBUTED EVAL WITH MORE THAN 1 GPU DOES NOT WORK
def eval_hvd(args, prefix=""):
    mlflow.set_tracking_uri("databricks")
    os.environ['DATABRICKS_HOST'] = "https://trend-prod.cloud.databricks.com/"
    os.environ['DATABRICKS_TOKEN'] = args.db_token
    with mlflow.start_run(run_id = args.active_run_id):
        hvd.init()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
        comm = get_comm_magic()
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _,tokenizer,model= get_trained_model(args)

        if args.device.type == 'cuda':
         # Pin GPU to local rank
            torch.cuda.set_device(0)
            #torch.cuda.set_device(hvd.local_rank())

        model.to(args.device)

        eval_dataset, (all_entities, _, _,all_candidate_embeddings), \
        (all_document_ids, all_label_candidate_ids) = load_and_cache_examples(args,tokenizer,model)
        
        logger.info("Evaluation Dataset Created")
        if not os.path.exists(args.output_dir) and hvd.rank==0:
            os.makedirs(args.output_dir)
        comm.barrier()
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        # Evaluation only supports args.per_gpu_eval_batch_size=1 n.gpu=1
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size)

        all_candidate_embeddings = all_candidate_embeddings.unsqueeze(0).expand(args.eval_batch_size, -1, -1)
        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        for gamma in np.linspace(args.gamma_lower,args.gamma_upper,args.gamma_step):
            args.gamma = gamma
            logger.info("Evaluating gamma %.2f", args.gamma)
            gamma_dir = os.path.join(args.output_dir,'evaluation/gamma_{:.2f}'.format(args.gamma))
            if not os.path.exists(gamma_dir) and hvd.rank()==0:
                os.makedirs(gamma_dir)
            comm.barrier()
            with mlflow.start_run(experiment_id=args.experiment_id,nested=True):
                mlflow.log_param("gamma",args.gamma)
                single_process_gold_path = os.path.join(gamma_dir,f'gold_{hvd.rank()}.csv')
                single_process_pred_path = os.path.join(gamma_dir,f'pred_{hvd.rank()}.csv')
                single_process_gold_file = open(single_process_gold_path, 'w+')
                single_process_pred_file = open(single_process_pred_path, 'w+')
                num_mention_processed = 0
                for batch_id,batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                    new_doc_id = str(batch_id)
                    model.eval()
                    num_mentions_processed_in_batch = eval_one_batch(args, model, all_entities, all_document_ids, all_label_candidate_ids, all_candidate_embeddings, single_process_gold_file, single_process_pred_file, num_mention_processed, batch,new_doc_id)
                    num_mention_processed += num_mentions_processed_in_batch
                single_process_gold_file.close()
                single_process_pred_file.close()
                comm.barrier()
            ##ONCE ALL BATCHES ARE FINISHED, COMBINE THEM INTO A SINGLE CSV USING THE ROOT NODE.
                logger.info(num_mention_processed)
                all_together_file_paths = {}
                if hvd.rank()==0:
                    for file_type in ["gold","pred"]:
                        all_files = glob.glob(os.path.join(gamma_dir, f"{file_type}_[0-9]*.csv"))
                        all_together_file_paths[file_type]= os.path.join(gamma_dir,f"{file_type}_ALL.csv")
                        df_from_each_file = [pd.read_csv(f, sep='\t',header=None,index_col=False) for f in all_files if os.stat(f).st_size != 0]
                        if df_from_each_file:
                            df_merged = pd.concat(df_from_each_file)
                            df_merged.to_csv(all_together_file_paths[file_type],sep="\t",header=False,index=False,na_rep='NA')
                        else:
                            #create empty file
                            with open(all_together_file_paths[file_type], 'w') as fp:
                                pass
                            

                    evaluator = EVALUATOR_CLASS(all_together_file_paths["pred"],
                                                all_together_file_paths["gold"],
                                                measures=MEASURES)
                    results = evaluator()
                    for measure_name,metrics in results.items():
                        easy_measure_name = MEASURES[measure_name]
                        for metric_name,metric in metrics.items():
                            full_metric_name = easy_measure_name + "_" + metric_name
                            mlflow.log_metric(full_metric_name,metric)
                mlflow.log_artifacts(gamma_dir)
        mlflow.log_artifacts(args.output_dir)

def eval_one_batch(args, model, all_entities, all_document_ids, all_label_candidate_ids, all_candidate_embeddings, single_process_gold_file, single_process_pred_file, num_mention_processed, batch,new_doc_id):
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
        if spans_after_prunning[0].size(0) > 0:
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
            #logger.info(str(logits))
            preds = logits.detach().cpu().numpy()
                # out_label_ids = batch[6]
                # out_label_ids = out_label_ids.reshape(-1).detach().cpu().numpy()
                
            sorted_preds = np.flip(np.argsort(preds), axis=1)
            #logger.info(str(sorted_preds))
            predicted_entities = []
            for i, _ in enumerate(sorted_preds):
                predicted_entity_idx = sorted_preds[i][0]
                predicted_entity = all_entities[predicted_entity_idx]
                predicted_entities.append(predicted_entity)
            # Write the predicted entities
            num_pred_mentions = len(predicted_entities)
            mention_start_indices = mention_start_indices.detach().cpu().numpy()
            mention_end_indices = mention_end_indices.detach().cpu().numpy()
            mention_probs = pred_mention_span_probs[spans_after_prunning].detach().cpu().numpy()

            for j in range(num_pred_mentions):
                    # if pred_mention_start_indices[j] == pred_mention_end_indices[j]:
                    #     pred_mention_end_indices[j] += 1
                if pred_mention_start_indices[j] > pred_mention_end_indices[j]:
                    continue
                pred_write = new_doc_id + '\t' + str(mention_start_indices[j]) \
                                + '\t' + str(mention_end_indices[j]) \
                                + '\t' + str(predicted_entities[j]) \
                                + '\t' + str(mention_probs[j]) \
                                + '\t' + 'NA' + '\n'
                single_process_pred_file.write(pred_write)
        num_mentions = batch[9].detach().cpu().numpy()[0]
        gold_mention_start_indices = batch[7].detach().cpu().numpy()[0][:num_mentions]
        gold_mention_end_indices = batch[8].detach().cpu().numpy()[0][:num_mentions]
        gold_entities = all_label_candidate_ids[num_mention_processed:num_mention_processed + num_mentions]
        for j in range(num_mentions):
            if gold_mention_start_indices[j] > gold_mention_end_indices[j]:
                continue
            gold_write = new_doc_id + '\t' + str(gold_mention_start_indices[j]) \
                            + '\t' + str(gold_mention_end_indices[j]) \
                            + '\t' + str(gold_entities[j]) \
                            + '\t' + str(1.0) \
                            + '\t' + 'NA' + '\n'
            single_process_gold_file.write(gold_write)
            
        return num_mentions

        

