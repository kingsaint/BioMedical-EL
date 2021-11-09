# Databricks notebook source
import horovod.torch as hvd
from sparkdl import HorovodRunner
def test_hvd():
  hvd.init()
hr = HorovodRunner(np=-1,driver_log_verbosity='all') 
hr.run(test_hvd)

# COMMAND ----------


