# Databricks notebook source
pip install transformers torch mpi4py

# COMMAND ----------

# MAGIC %sh
# MAGIC python ../run_e2e_distributed.py --model_type bert --data_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data --model_name_or_path /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/model --output_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/output_dir_distributed --do_train --overwrite_cache --overwrite_output_dir --num_candidates 5  --num_train_epochs 1 --use_random_candidates --per_gpu_train_batch_size 1 --n_gpu 2

# COMMAND ----------

from run_e2e_distributed import main
dict_args = {"model_type":"bert",
             "data_dir":"/dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data",
             "model_name_or_path":"/dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/model",
             "output_dir":"/dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/output_dir_distributed",
             "max_steps":"15",
             "per_gpu_train_batch_size": "1",
             "n_gpu":"2",
             "num_train_epochs":"1",
             "num_candidates":"5",
             "do_train":"True",
             "use_random_candidates":"True",
             "overwrite_output_dir":"True"
                 }


main(dict_args)

# COMMAND ----------

import horovod.torch as hvd
from sparkdl import HorovodRunner
import logging 
import torch
logger = logging.getLogger(__name__)

def test_hvd():
  hvd.init()




hr = HorovodRunner(np=2,driver_log_verbosity='all') 
hr.run(test_hvd)


# COMMAND ----------

# MAGIC %sh
# MAGIC python ../run_e2e_span.py --model_type bert --data_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data --model_name_or_path /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/model --output_dir /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/output_dir --do_eval --use_all_candidates --overwrite_cache --gamma .8

# COMMAND ----------

pip install git+https://github.com/wikilinks/neleval

# COMMAND ----------

# MAGIC %sh
# MAGIC neleval evaluate -g /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data/gold.csv /dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/processed_data/pred.csv -m overlap-maxmax::span+kbid -m strong_all_match

# COMMAND ----------


