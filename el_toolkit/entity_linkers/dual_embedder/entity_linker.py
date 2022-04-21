from copy import copy
from el_toolkit.entity_linkers.base_entity_linker import EntityLinker
import el_toolkit.entity_linkers.config as config
from el_toolkit.entity_linkers.dual_embedder.featurizer import BertDualEmbedderTrainFeaturizer, DualEmbedderEvalFeaturizer
from el_toolkit.entity_linkers.dual_embedder.model import BertDualEmbedderModel
from lkb.basic_lkb import Lexical_Knowledge_Base
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)




class DualEmbedderEntityLinker(EntityLinker,config.SavableCompositeComponent):
    def __init__(self,bert_concept_embedder,document_embedder,knowledge_data_path=None,distributed=False):#Might make sense to turn this into a factory.
        self._concept_embedder = bert_concept_embedder
        self._document_embedder = document_embedder
        self._dual_embedder_model = BertDualEmbedderModel(self._concept_embedder.bert_model,document_embedder.span_detector)
        self._distributed = distributed
        if self._distributed:
            import horovod.torch as hvd
            self._hvd = hvd
        self._knowledge_data_path = knowledge_data_path
        lkb = Lexical_Knowledge_Base.read_json(knowledge_data_path)
        self._train_featurizer = BertDualEmbedderTrainFeaturizer(lkb,self,distributed=distributed)
        self._eval_featurizer =  DualEmbedderEvalFeaturizer(self)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dual_embedder_model.to(self._device)
    @property
    def concept_embedder(self):
        return self._concept_embedder
    @property
    def document_embedder(self):
        return self._document_embedder
    @property
    def knowledge_data_path(self):
        return self._knowledge_data_path
    def train_featurize(self,docs,num_hard_negatives=0,num_random_negatives=0,num_max_mentions=8):
        return self._train_featurizer.featurize(docs,num_hard_negatives=num_hard_negatives,num_random_negatives=num_random_negatives,num_max_mentions=num_max_mentions)
    def eval_featurize(self,docs):
        return self._eval_featurizer.featurize(docs)
    def train(self,docs,num_hard_negatives=0,num_random_negatives=0,num_max_mentions=8,single_node_batch_size=1,num_epochs=100,learning_rate=1e-5,adam_epsilon=1e-8,weight_decay=0.0,warmup_steps=0,gradient_accumulation_steps=1,max_grad_norm=1.0):
        writer = SummaryWriter()
        t_total = len(docs) // gradient_accumulation_steps * num_epochs
        self._dual_embedder_model.zero_grad()
        epochs_trained = 0
        if self._distributed:
            disable = self._hvd.rank()!=0
        else:
            disable = False
        train_iterator = trange(epochs_trained, num_epochs, desc="Epoch",disable=disable
                               )
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self._dual_embedder_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in self._dual_embedder_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        ]
        if self._distributed:#distributed
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate * self._hvd.size(), eps=adam_epsilon)
            optimizer = self._hvd.DistributedOptimizer(optimizer, named_parameters=self._dual_embedder_model.named_parameters())
            self._hvd.broadcast_parameters(self._dual_embedder_model.state_dict(), root_rank=0)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
        self._dual_embedder_model.train()
        print("TRAINING STARTING")
        for epoch_number in train_iterator:
            train_dataset = self.train_featurize(docs,num_hard_negatives=num_hard_negatives,num_random_negatives=num_random_negatives,num_max_mentions=num_max_mentions)
            train_sampler = RandomSampler(train_dataset) if not self._distributed else DistributedSampler(train_dataset, num_replicas=self._hvd.size(), rank=self._hvd.rank())
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=single_node_batch_size)
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
    def accept(self,visitor):
        return visitor.visit_dual_embedder(self)
    @classmethod
    def class_accept(cls,visitor):
        return visitor.visit_dual_embedder(cls)
    
        

        
    

