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

# MAGIC %sh
# MAGIC python ../run_e2e_span.py --model_type bert --data_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data --model_name_or_path /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/model --output_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/output_dir --do_train

# COMMAND ----------


