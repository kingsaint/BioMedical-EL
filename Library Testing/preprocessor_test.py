# Databricks notebook source
pip install transformers

# COMMAND ----------

from data_preprocessing import preprocess_data
preprocess_data("/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR",
                dbutils)

# COMMAND ----------


