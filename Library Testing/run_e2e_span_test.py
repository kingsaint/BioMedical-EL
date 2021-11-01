# Databricks notebook source
pip install transformers torch faiss-gpu

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
# MAGIC python ../run_e2e_span.py --model_type bert --data_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data --model_name_or_path /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/model --output_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/output_dir --overwrite_output_dir --do_train --do_eval --overwrite_cache --num_candidates 8  --num_train_epochs 1 

# COMMAND ----------

# MAGIC %sh
# MAGIC python ../run_e2e_span.py --model_type bert --data_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data --model_name_or_path /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/model --output_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/output_dir --do_eval --use_all_candidates

# COMMAND ----------

pip install git+https://github.com/wikilinks/neleval

# COMMAND ----------

# MAGIC %sh python -m neleval --help

# COMMAND ----------

# MAGIC %sh python -m neleval list-measures

# COMMAND ----------

# MAGIC %sh
# MAGIC neleval evaluate -g /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data/gold.csv /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data/pred.csv 

# COMMAND ----------

# MAGIC %sh nvidia-smi

# COMMAND ----------

import os
os.path.join("/dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data","gold_file.txt")

# COMMAND ----------


