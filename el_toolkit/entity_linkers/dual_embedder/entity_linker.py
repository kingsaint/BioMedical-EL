
from collections import namedtuple
from el_toolkit.entity_linkers.dual_embedder.featurizer import DualEmbedderEvalFeaturizer, DualEmbedderTrainFeaturizer
from el_toolkit.mpi_utils import partition

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
        self._dual_embedder_model = dual_embedder_model
    def train_featurize(self,docs,lkb,num_hard_negatives=0,num_random_negatives=0,num_max_mentions=8):
        featurizer = DualEmbedderTrainFeaturizer(self,lkb,num_hard_negatives,num_random_negatives,num_max_mentions)
        return featurizer.featurize(docs)
    def eval_featurize(self,docs):
        featurizer = DualEmbedderEvalFeaturizer(self)
        return featurizer.featurize(docs)
    @property
    def concept_embedder(self):
        return self._concept_embedder
    @property
    def document_embedder(self):
        return self._document_embedder



