# BioMedical-EL

## Data Preprocessing

Use the `data_preprocessing.py` script to preprocess the data. In this pre-processing phase, each document is segmented into multiple 
chunks where each chunk contains a maximum of 8 mentions.

Example:

```
python data_preprocessing.py --data_dir data/BC5CDR
```

## Download BioBERT model

Download the BioBERT v1.1 model and place is under the root directory.

## Requirements

* Python 3.6+
* pytorch 1.4 
* [faiss](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md)
* [neleval](https://github.com/wikilinks/neleval)

## Training and Inference

### End-to-end span-based model 

```
optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   The input data dir. Should contain the .tsv files (or other data files) for the task.
  --model_type MODEL_TYPE
                        Model type selected in the list: bert
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pre-trained model or shortcut name selected in the list: bert-base-uncased, bert-
                        large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-
                        multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-
                        masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-
                        squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc,
                        bert-base-german-dbmdz-cased, bert-base-german-dbmdz-uncased, bert-base-japanese, bert-base-
                        japanese-whole-word-masking, bert-base-japanese-char, bert-base-japanese-char-whole-word-
                        masking, bert-base-finnish-cased-v1, bert-base-finnish-uncased-v1, bert-base-dutch-cased
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be written.
  --resume_path RESUME_PATH
                        Path to the checkpoint from where the training should resume
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the same as model_name
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as model_name
  --cache_dir CACHE_DIR
                        Where do you want to store the pre-trained models downloaded from s3
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after tokenization. Sequences longer than this will be
                        truncated, sequences shorter will be padded.
  --max_mention_length MAX_MENTION_LENGTH
                        Maximum length of a mention span
  --do_train            Whether to run training.
  --do_eval             Whether to run eval on the test set.
  --evaluate_during_training
                        Rul evaluation during training at each logging step.
  --do_lower_case       Set this flag if you are using an uncased model.
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Batch size per GPU/CPU for evaluation.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --weight_decay WEIGHT_DECAY
                        Weight decay if we apply some.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_steps MAX_STEPS
                        If > 0: set total number of training steps to perform. Override num_train_epochs.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps.
  --eval_all_checkpoints
                        Evaluate all checkpoints starting with the same prefix as model_name ending and ending with
                        step number
  --no_cuda             Avoid using CUDA when available
  --n_gpu N_GPU         Number of GPUs to use when available
  --overwrite_output_dir
                        Overwrite the content of the output directory
  --overwrite_cache     Overwrite the cached training and evaluation sets
  --use_random_candidates
                        Use random negative candidates during training
  --use_tfidf_candidates
                        Use random negative candidates during training
  --use_hard_negatives  Use hard negative candidates during training
  --use_hard_and_random_negatives
                        Use hard negative candidates during training
  --include_positive    Includes the positive candidate during inference
  --use_all_candidates  Use all entities as candidates
  --num_candidates NUM_CANDIDATES
                        Number of candidates to consider per mention
  --num_max_mentions NUM_MAX_MENTIONS
                        Maximum number of mentions in a document
  --ner NER             Model will perform only NER
  --alternate_batch ALTERNATE_BATCH
                        Model will perform either NER or entity linking per batch during training
  --ner_and_ned NER_AND_NED
                        Model will perform both NER and entity linking per batch during training
  --gamma GAMMA         Threshold for mention candidate prunning
  --seed SEED           random seed for initialization
  --fp16                Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit
  --fp16_opt_level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].See details at
                        https://nvidia.github.io/apex/amp.html
  --local_rank LOCAL_RANK
                        For distributed training: local_rank
  --server_ip SERVER_IP
                        For distant debugging.
  --server_port SERVER_PORT
                        For distant debugging.
```

## Evaluation

```
neleval evaluate -g ./neleval/BC5CDR-AllSpan/gold.csv ./neleval/BC5CDR-AllSpan/pred.csv -m overlap-maxmax::span+kbid -m strong_all_match
```