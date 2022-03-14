from el_toolkit.entity_linkers.dual_embedder.featurizer import BertDualEmbedderTrainFeaturizer, DualEmbedderEvalFeaturizer, DualEmbedderTrainFeaturizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

def truncate(token_ids,pad_token_id,max_seq_len):
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]
        tokens_mask = [1] * max_seq_len
    else:
        token_number = len(token_ids)
        pad_len = max_seq_len - token_number
        token_ids += [pad_token_id] * pad_len
        tokens_mask = [1] * token_number + [0] * pad_len
    return token_ids,tokens_mask

class EntityLinker:
    def __init__(self):
        raise NotImplementedError
    def featurize(self,docs,lkb):
        raise NotImplementedError
    def train(self,docs):
        raise NotImplementedError

class DualEmbedderEntityLinker(EntityLinker):
    def __init__(self,concept_embedder,document_embedder,dual_embedder_model):#Might make sense to turn this into a factory.
        self._concept_embedder = concept_embedder
        self._document_embedder = document_embedder
        self._train_featurizer = BertDualEmbedderTrainFeaturizer(concept_embedder.lkb,self)
        self._eval_featurizer =  DualEmbedderEvalFeaturizer(self)
        self._dual_embedder_model = dual_embedder_model
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dual_embedder_model.to(self._device)
    def train_featurize(self,docs,num_hard_negatives=0,num_random_negatives=0,num_max_mentions=8,hvd=None):
        return self._train_featurizer.featurize(docs,num_hard_negatives=num_hard_negatives,num_random_negatives=num_random_negatives,num_max_mentions=num_max_mentions,hvd=hvd)
    def eval_featurize(self,docs):
        return self._eval_featurizer.featurize(docs)
    def train(self,docs,num_hard_negatives=0,num_random_negatives=0,num_max_mentions=8,single_node_batch_size=1,num_epochs=100,learning_rate=1e-5,adam_epsilon=1e-8,weight_decay=0.0,warmup_steps=0,gradient_accumulation_steps=1,max_grad_norm=1.0,hvd=None):
        writer = SummaryWriter()
        t_total = len(docs) // gradient_accumulation_steps * num_epochs
        self._dual_embedder_model.zero_grad()
        epochs_trained = 0
        train_iterator = trange(epochs_trained, num_epochs, desc="Epoch" #, disable=args.local_rank not in [-1, 0]
                               )
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self._dual_embedder_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in self._dual_embedder_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        ]
        if hvd:#distributed
            batch_size = single_node_batch_size * hvd.size()
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate * hvd.size(), eps=adam_epsilon)
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=self._dual_embedder_model.named_parameters())
            hvd.broadcast_parameters(self._dual_embedder_model.state_dict(), root_rank=0)
        else:
            batch_size = single_node_batch_size
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
        self._dual_embedder_model.train()
        for epoch_number in train_iterator:
            train_dataset = self.train_featurize(docs,num_hard_negatives=num_hard_negatives,num_random_negatives=num_random_negatives,num_max_mentions=num_max_mentions,hvd=hvd)
            train_sampler = RandomSampler(train_dataset) if not hvd else DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
            num_examples = len(train_dataloader)
            epoch_iterator = tqdm(train_dataloader, desc="Iteration",disable=True)#, disable=args.local_rank not in [-1, 0])
            epoch_loss = 0 
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self._device) for t in batch)
                inputs = {field:batch[i] for i,field in enumerate(self._train_featurizer.TrainingInputFeatures._fields)}
                _,loss = self._dual_embedder_model.forward(**inputs)
                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self._dual_embedder_model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self._dual_embedder_model.zero_grad()
                epoch_loss += loss.item()
            writer.add_scalar('Loss/train', epoch_loss/num_examples, epoch_number)
    @property
    def concept_embedder(self):
        return self._concept_embedder
    @property
    def document_embedder(self):
        return self._document_embedder



