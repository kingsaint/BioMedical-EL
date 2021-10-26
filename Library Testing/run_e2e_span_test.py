# Databricks notebook source
pip install transformers torch==1.5 faiss-cpu tensorboard==1.14.0

# COMMAND ----------

# MAGIC %sh
# MAGIC python ../run_e2e_span.py --model_type bert --data_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data --model_name_or_path bert-base-uncased --output_dir dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR --do_train --overwrite_cache

# COMMAND ----------


