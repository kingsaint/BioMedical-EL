import argparse
import os
from .utils_e2e_span import MODEL_CLASSES, set_seed
from .train import train_hvd
from .evaluate import eval_hvd
from .configuration_bert import BertConfig
from sparkdl import HorovodRunner
import mlflow
import glob

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in [BertConfig]), ()
)

EVAL_ARGS = {
                    "n_gpu",
                    "max_mention_length",
                    "max_seq_length",
                    "use_all_candidates",
                    "ner_and_ned",
                    "seed",
                    "gamma_lower",
                    "gamma_upper",
                    "gamma_step",
                    "kb_size"
                }   
TRAINING_ARGS = {   
                    "lambda_1",
                    "lambda_2",
                    "weight_decay",
                    "learning_rate",
                    "adam_epsilon",
                    "max_grad_norm",
                    "num_train_epochs",
                    "n_gpu",
                    "max_mention_length",
                    "max_seq_length",
                    "gradient_accumulation_steps",
                    "per_gpu_train_batch_size",
                    "num_candidates",
                    "num_max_mentions",
                    "max_steps",
                    "use_tfidf_candidates",
                    "use_random_candidates",
                    "use_hard_negatives",
                    "use_hard_and_random_negatives",
                    "ner_and_ned",
                    "overwrite_hard_negatives",
                    "seed",
                    "resume_path",
                    "overwrite_cache",
                    "kb_size"
                }  
def get_args(dict_args = None):


    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--base_model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--base_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--resume_path",
        default=None,
        type=str,
        required=False,
        help="Path to the checkpoint from where the training should resume"
    )
    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_mention_length",
        default=20,
        type=int,
        help="Maximum length of a mention span"
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", default=False, help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_epochs", type=int, default=1, help="Save checkpoint every X epochs.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--use_random_candidates", action="store_true", help="Use random negative candidates during training"
    )
    parser.add_argument(
        "--use_tfidf_candidates", action="store_true", help="Use random negative candidates during training"
    )
    parser.add_argument(
        "--use_hard_negatives",  action="store_true", help="Use hard negative candidates during training"
    )
    parser.add_argument(
        "--use_hard_and_random_negatives", action="store_true", help="Use hard negative candidates during training"
    )
    parser.add_argument(
        "--include_positive", action="store_true", help="Includes the positive candidate during inference"
    )
    parser.add_argument(
        "--use_all_candidates", action="store_true", help="Use all entities as candidates"
    )
    parser.add_argument(
        "--num_candidates", type=int, default=10, help="Number of candidates to consider per mention"
    )
    parser.add_argument(
        "--num_max_mentions", type=int, default=8, help="Maximum number of mentions in a document"
    )
    parser.add_argument(
        "--ner", type=bool, default=False, help="Model will perform only BIO tagging"
    )
    parser.add_argument(
        "--alternate_batch", type=bool, default=False, help="Model will perform either BIO tagging or entity linking per batch during training"
    )
    parser.add_argument(
        "--ner_and_ned", type=bool, default=True, help="Model will perform both BIO tagging and entity linking per batch during training"
    )
    parser.add_argument(
        "--gamma_lower", type=float, default=.1, help="Lower boundary for gamma for mention candidate prunning"
    )
    parser.add_argument(
        "--gamma_upper", type=float, default=.9, help="Upper boundary for gamma for mention candidate prunning"
    )
    parser.add_argument(
        "--gamma_step", type=int, default=10, help="Number of steps between upper and lower boundaries for gamma range to evaluate"
    )
    parser.add_argument(
        "--lambda_1", type=float, default=1, help="Weight of the random candidate loss"
    )
    parser.add_argument(
        "--lambda_2", type=float, default=0, help="Weight of the hard negative candidate loss"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--overwrite_hard_negatives", type=bool, default=True, help="Model will start without prepopulated hard negatives."
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--experiment_dir", type=str, default="", help="To log parameters and metrics.")
    parser.add_argument("--kb_size", default="full", type=str, help="Select from 'easy','very_easy' or 'full'."
    )
    list_args = []
    if dict_args != None:
      for key,value in dict_args.items():
        if value =="True":
          list_args.append("--"+key)
        else:
          list_args.append("--"+key)
          list_args.append(value)
      args = parser.parse_args(list_args)
    else:
      args = parser.parse_args()
    return args

def main(db_token,args=None):
    args = get_args(args)
    
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    
    # Set seed
    set_seed(args)
    if not args.do_train and not args.do_eval:
        raise Exception("Pick at least one of do_train or do_eval.")

    if args.do_train:
        args.experiment_name = os.path.join(args.experiment_dir,"training")
        mlflow.set_experiment(args.experiment_name)
        experiment = mlflow.get_experiment_by_name(args.experiment_name)
        args.experiment_id = experiment.experiment_id
        with mlflow.start_run() as run:
            for arg_name,arg_value in args.__dict__.items():
                if arg_name in TRAINING_ARGS:
                    mlflow.log_param(arg_name,arg_value)
            args.active_run_id = run.info.run_id
            args.db_token = db_token
            hr = HorovodRunner(np=args.n_gpu,driver_log_verbosity='all') 
            hr.run(train_hvd, args=args)
    if args.do_eval:
        
        args.experiment_name = os.path.join(args.experiment_dir,"evaluation")
        mlflow.set_experiment(args.experiment_name)
        experiment = mlflow.get_experiment_by_name(args.experiment_name)
        args.experiment_id = experiment.experiment_id
        if not args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "checkpoint-[0-9].-FINAL/", recursive=True))
            )
        else:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "checkpoint**/", recursive=True))
            )
        print(checkpoints)
        
        for checkpoint in checkpoints:
            with mlflow.start_run() as run:
                mlflow.log_param("checkpoint",checkpoint.split("-")[-1])
                args.output_dir = checkpoint
                with mlflow.start_run(experiment_id=args.experiment_id,nested=True) as run:
                    for arg_name,arg_value in args.__dict__.items():
                        if arg_name in EVAL_ARGS:
                            mlflow.log_param(arg_name,arg_value)
                    args.active_run_id = run.info.run_id
                    args.db_token = db_token
                    hr = HorovodRunner(np=args.n_gpu,driver_log_verbosity='all') 
                    hr.run(eval_hvd, args=args)

    


if __name__ == "__main__":
    main()