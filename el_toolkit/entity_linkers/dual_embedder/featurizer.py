from collections import namedtuple
from el_toolkit.utils import partition
from mpi4py import MPI
import torch
import random
from torch.utils.data.dataset import TensorDataset

COMM = MPI.COMM_WORLD
InferenceFeatures = namedtuple("InferenceInputFeatures",["doc_token_ids","doc_token_masks"])

class DualEmbedderFeaturizer:
    def __init__(self,dual_embedder,distributed=False):
        self._dual_embedder = dual_embedder
        self._distributed = distributed
        if self._distributed:
            import horovod.torch as hvd
            self._hvd = hvd
    def featurize(self,docs,**intialization_kwargs):
        self.initialize(**intialization_kwargs)
        if self._distributed:#distributed
            docs = partition(docs,self._hvd.size(),self._hvd.rank())
        features = [self.featurize_doc(doc) for doc in docs]
        if self._distributed:
            features = [features for node_features in COMM.allgather(features) for features in node_features]
        return self.get_tensor_dataset(features)
    def pad_mention_indices(self,mention_start_indices,mention_end_indices):
        # Pad the mention start and end indices
        padded_mention_start_indices = [-1] * self._num_max_mentions
        padded_mention_end_indices = [-1] * self._num_max_mentions
        num_mentions = len(mention_start_indices)
        if num_mentions > 0: 
            if num_mentions <= self._num_max_mentions:
                padded_mention_start_indices[:num_mentions] = mention_start_indices
                padded_mention_end_indices[:num_mentions] = mention_end_indices
            else:
                padded_mention_start_indices = mention_start_indices[:self._num_max_mentions]
                padded_mention_end_indices = mention_end_indices[:self._num_max_mentions]
        return padded_mention_start_indices,padded_mention_end_indices
    def get_tensor_dataset(self,feature_list):
        grouped_features = list(zip(*feature_list))
        tensors = [torch.tensor(feature_group, dtype=torch.long) for feature_group in grouped_features]  
        return TensorDataset(*tensors)
    def initialize(self):
        pass
    

class DualEmbedderTrainFeaturizer(DualEmbedderFeaturizer):
    def __init__(self,lkb,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._lkb = lkb
        self._concept_ids = self._lkb.get_concept_ids()
        self._max_entity_len =  self._dual_embedder.concept_embedder.max_ent_len
    def initialize(self,num_hard_negatives=0,num_random_negatives=0,num_max_mentions=8):
        self._total_number_of_candidates = num_random_negatives + num_hard_negatives + 1
        self._num_hard_negatives = num_hard_negatives
        self._num_random_negatives = num_random_negatives
        self._num_max_mentions = num_max_mentions
        if num_hard_negatives:
            self._encoded_concepts = self._dual_embedder.concept_embedder.get_encodings(self._lkb)
            self._embeddings = self._dual_embedder.concept_embedder.embed_from_concept_encodings(self._encoded_concepts)
        else:
            self._encoded_concepts = None
    def get_candidate_id_lists(self,doc,doc_token_ids,doc_token_mask,mention_start_indices,mention_end_indices):
        candidate_id_lists = [[mention.concept_id] for mention in doc.mentions]
        if self._num_random_negatives:
            random_negative_id_lists = [self.get_random_negative_concept_ids(mention.concept_id,self._num_random_negatives) for mention in doc.mentions]
            for i,random_negative_id_list in enumerate(random_negative_id_lists):
                candidate_id_lists[i].extend(random_negative_id_list)
        if self._num_hard_negatives:
            mention_embeddings = self.get_mention_embeddings(self,doc_token_ids,doc_token_mask,mention_start_indices,mention_end_indices)
            hard_negative_id_lists = self._embeddings.get_most_similar(mention_embeddings,self._num_hard_negatives)
            for i,hard_negative_id_list in enumerate(hard_negative_id_lists):
                candidate_id_lists[i].extend(hard_negative_id_list)
        return candidate_id_lists
    def get_random_negative_concept_ids(self,concept_id,k):
        candidate_pool = self._concept_ids - set(concept_id)
        m_random_negative_ids = random.sample(candidate_pool,k)
        return m_random_negative_ids
    def get_label_ids(self,doc,num_max_mentions):
        label_ids = [-1] * num_max_mentions
        for i,mention in enumerate(doc.mentions):
            if i < num_max_mentions:
                if mention.concept_id in self._concept_ids:
                    label_ids[i] = 0
                # else:
                #     label_ids[i] = -100 
        return label_ids


class BertDualEmbedderTrainFeaturizer(DualEmbedderTrainFeaturizer):
    TrainingInputFeatures = namedtuple("TrainingInputFeatures",["doc_token_ids","doc_token_masks","mention_start_indices", "mention_end_indices","label_ids","num_mentions","candidate_token_ids","candidate_token_masks","candidate_masks"])
    def featurize_doc(self,doc):
        num_mentions = len(doc.mentions)
        doc_token_ids,doc_token_mask,mention_start_indices,mention_end_indices,_,_  = self._dual_embedder.document_embedder.encode_document(doc)
        mention_start_indices,mention_end_indices = self.pad_mention_indices(mention_start_indices,mention_end_indices)
        candidate_id_lists = self.get_candidate_id_lists(doc,doc_token_ids,doc_token_mask,mention_start_indices,mention_end_indices)
        candidate_token_ids,candidate_token_masks,candidate_masks = self.get_candidate_tokens(candidate_id_lists)
        label_ids = self.get_label_ids(doc,self._num_max_mentions)
        return self.TrainingInputFeatures(doc_token_ids,doc_token_mask,mention_start_indices,mention_end_indices,label_ids,num_mentions,candidate_token_ids,candidate_token_masks,candidate_masks)
    def get_candidate_tokens(self,candidate_id_lists):
        candidate_token_ids = [[self._dual_embedder.concept_embedder.tokenizer.pad_token_id] * self._max_entity_len] * (self._num_max_mentions * self._total_number_of_candidates)
        candidate_token_masks= [[0] * self._max_entity_len] * (self._num_max_mentions * self._total_number_of_candidates)
        candidate_masks = [0]*(self._num_max_mentions * self._total_number_of_candidates)
        for mention_number,candidate_id_list in enumerate(candidate_id_lists):
            c_idx = mention_number * self._total_number_of_candidates
            for candidate_id in candidate_id_list:
                if self._encoded_concepts:
                    encoded_candidate = self._encoded_concepts.get_concept_encoding(candidate_id)
                else:
                    encoded_candidate = self._dual_embedder.concept_embedder.encode_concept(candidate_id,self._lkb)
                candidate_token_ids[c_idx] = encoded_candidate.token_ids
                candidate_token_masks[c_idx] = encoded_candidate.token_masks
                candidate_masks[c_idx] = 1
                c_idx += 1
        return candidate_token_ids,candidate_token_masks,candidate_masks
    
class DualEmbedderEvalFeaturizer(DualEmbedderFeaturizer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.EvalInputFeatures = namedtuple("EvalInputFeatures",["doc_token_ids","doc_token_masks","mention_start_indices","mention_end_indices","label_ids","seq_tag_ids"])
    def featurize_doc(self,doc):
        if self._lower():
                doc = doc.lower()
        doc_token_ids,doc_token_mask,mention_start_indices,mention_end_indices,seq_tag_ids,too_long  = self._dual_embedder.document_embedder.encode_document(doc)
        mention_start_indices,mention_end_indices = self.pad_mention_indices(mention_start_indices,mention_end_indices)
        label_ids = self._get_label_ids(doc,100)
        return self.EvalInputFeatures(doc_token_ids,doc_token_mask,mention_start_indices,mention_end_indices,label_ids,seq_tag_ids)
