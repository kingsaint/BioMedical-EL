# Databricks notebook source
pip install transformers torch

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForMaskedLM
  
model = AutoModelForMaskedLM.from_pretrained("monologg/biobert_v1.1_pubmed")

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")

# COMMAND ----------

model.save_pretrained("/dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/model")

# COMMAND ----------

tokenizer.save_pretrained("/dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/model")

# COMMAND ----------

pip install

# COMMAND ----------

# MAGIC %sh
# MAGIC python ../run_e2e_span.py --model_type bert --data_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data --model_name_or_path /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/model --output_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/output_dir --overwrite_output_dir --do_train --overwrite_cache --num_candidates 8  --num_train_epochs 3 --use_random_candidates  --per_gpu_train_batch_size 1 --n_gpu 4

# COMMAND ----------

# MAGIC %sh nvidia-smi

# COMMAND ----------

# MAGIC %sh
# MAGIC python ../run_e2e_span.py --model_type bert --data_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data --model_name_or_path /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/model --output_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/output_dir --do_eval --use_all_candidates --overwrite_cache --gamma .2

# COMMAND ----------

pip install git+https://github.com/wikilinks/neleval

# COMMAND ----------

import torch
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %sh gpustat -p

# COMMAND ----------

# MAGIC %sh python -m neleval --help

# COMMAND ----------

# MAGIC %sh python -m neleval list-measures

# COMMAND ----------

# MAGIC %sh
# MAGIC neleval evaluate -g /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data/gold.csv /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data/pred.csv -m overlap-maxmax::span+kbid -m strong_all_match

# COMMAND ----------

# MAGIC %sh
# MAGIC neleval evaluate -g /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data/gold.csv /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data/pred.csv 

# COMMAND ----------



# COMMAND ----------

import pyspark.sql.functions as f
import os
gold = spark.read.csv("/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data/gold.csv",sep="\t")
predictions = spark.read.csv("/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data/pred.csv",sep="\t")
display(gold)
display(predictions)

# COMMAND ----------

predictions.count()

# COMMAND ----------

gold.count()

# COMMAND ----------

import pyspark.sql.functions as F
import os
evaluation_df = (predictions.alias("predictions")
                   .join(gold.alias("gold"),((predictions._c0==gold._c0)&(predictions._c1==gold._c1)&(predictions._c2==gold._c2)))
                   .select(F.col("gold._c3").alias("gold_entity"),F.col("predictions._c3").alias("prediction_entity"),F.col("predictions._c4").alias("mention_score"))
                   .where(F.col("gold_entity")==F.col("prediction_entity"))
#                    .groupBy("gold_entity")
#                    .agg(f.collect_set(f.col("prediction_entity")))
                )
display(evaluation_df)

# COMMAND ----------

display(predictions.select("_c3").dropDuplicates())

# COMMAND ----------

experiment_log_dir = "/dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/tensorboard_log"

# COMMAND ----------

import tensorflow as tf; 
print(tf.__version__)

# COMMAND ----------

from tensorboard import notebook
notebook.start("--logdir {}".format(experiment_log_dir))

# COMMAND ----------

# MAGIC %sh kill 2674

# COMMAND ----------

from torch.utils.tensorboard import SummaryWriter
import time
writer = SummaryWriter(experiment_log_dir,flush_secs=5)
x = range(100)
for i in x:
    time.sleep(1)
    writer.add_scalar('y=3x', i * 3, i)
    print(i)
writer.close()

# COMMAND ----------

import torch
features = torch.load("/dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data/cached_test_model")

# COMMAND ----------

len(features)

# COMMAND ----------


