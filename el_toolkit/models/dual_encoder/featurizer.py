from argparse import ArgumentError
from collections import namedtuple
import functools
import math
from statistics import mode
from el_toolkit.models.dual_encoder.document_embedder import Document_Embedder
from el_toolkit.models.dual_encoder.encoder import Document_Encoder,Entity_Encoder
from el_toolkit.models.dual_encoder.entity_embedder import Entity_Embedder
from el_toolkit.mpi_utils import partition
from pydoc import doc
import random
comm = None

#define different input features for different needs.
class Featurizer:#Only works for non-overlapping spans.
    def __init__(self,tokenizer,hvd,entities,encoded_entities = None,max_seq_length=256,lower_case=False):
        self._lower_case = lower_case
        self._num_candidates
        if self._comm is None:
            self._comm = MPI.COMM_WORLD
        self._hvd = hvd
        if lower_case:
            self._entities = {key:value.lower() for key,value in entities.items()}
        self._encoded_entities = encoded_entities
        self._entity_encoder = Entity_Encoder(tokenizer,max_seq_length)
        self._document_encoder = Document_Encoder(tokenizer,max_seq_length)
    def featurize(self,docs):#distributed
        docs = partition(docs,self._hvd.size(),self._hvd.rank())
        if self._lower_case:
            docs = [doc.lower() for doc in docs]
        features = [self.featurize_doc(doc) for doc in docs]
        return self._hvd.all_gather(features)#list of input_features objects

    def create_mention_index_matrices(self,encoded_doc):
        # Pad the mention start and end indices
        mention_start_indices = [0] * self._num_max_mentions
        mention_end_indices = [0] * self._num_max_mentions
        if encoded_doc.num_mentions > 0: 
            if encoded_doc.num_mentions <= self._num_max_mentions:
                mention_start_indices[:encoded_doc.num_mentions] = encoded_doc.mention_start_markers
                mention_end_indices[:encoded_doc.num_mentions] = encoded_doc.mention_end_markers
            else:
                mention_start_indices = encoded_doc.mention_start_markers[:self._num_max_mentions]
                mention_end_indices = encoded_doc.mention_end_markers[:self._num_max_mentions]

        return mention_start_indices,mention_end_indices
    def featurize_doc(self,doc):
        raise NotImplementedError
    

class TrainFeaturizer(Featurizer):
    def __init__(self,*args,use_random_negatives=True,use_hard_negatives=True,num_hard_negatives=8,num_random_negatives=8,model=None,**kwargs):
        self.super().__init__(*args,**kwargs)
        self._entity_embedder =  Entity_Embedder(model,self._tokenizer)
        self._document_embedder = Document_Embedder(model,self._tokenizer)
        self._use_random_negatives = use_random_negatives
        self._use_hard_negatives = use_hard_negatives
        if self._use_hard_negatives:
            if model==None:
                raise ArgumentError
        else:
            self._num_hard_negatives = 0
        if not self._use_random_negatives:
            self._num_random_negatives = 0 
        self._total_number_of_candidates = self._num_random_negatives + self._num_hard_negatives + 1
        self._concept_id_set = self._entities.keys()
        if self._use_hard_negatives:
            if not self._encoded_entities:
                self._encoded_entities = self._entity_encoder.encode_all_entities(self._entities)
            self._all_embedded_entities = self._entity_embedder.embed_all_entities(self._encoded_entities.values())
        else:
            self._all_embedded_entities = None
    #entities is a dictionary from concept_id to term
    def featurize_doc(self,doc):
        if self._lower_case:
            doc = doc.lower()
        encoded_doc = self._document_encoder.encode_doc(doc)
        label_ids = [0 if mention.concept_id in self._concept_id else -100 for mention in doc.mentions]
        if self._encoded_entities:
            encoded_positive_entity = [self._encoded_entities[mention.concept_id] for mention in doc.mentions()]
        else:
            encoded_positive_entity = [self._entity_encoder(mention.concept_id) for mention in doc.mentions()]
        encoded_candidates = [encoded_positive_entity]
        if self._use_random_negatives:
            encoded_random_negatives = self.get_random_negatives(doc)
            encoded_candidates.extend(encoded_random_negatives)
        if self._encoded_entities:
            encoded_hard_negatives = self.get_hard_negatives(encoded_doc)
            encoded_candidates.extend(encoded_hard_negatives)
        candidate_token_ids,candidate_token_masks = self.create_mention_index_matrices(encoded_candidates)
        mention_start_indices,mention_end_indices = self._create_mention_index_matrices(encoded_doc)
        return TrainingInputFeatures(doc.token_ids,doc.doc_tokens_mask,candidate_token_ids,candidate_token_masks,label_ids,mention_start_indices,mention_end_indices)

    def create_candidate_feature_matrices(self,encoded_candidates):
        ##encode_candidates
        candidate_token_ids = [[self._tokenizer.pad_token_id] * self._max_entity_len] * (self._num_max_mentions * self._total_number_of_candidates)
        candidate_token_masks= [[0] * self._max_entity_len] * (self._num_max_mentions * self._total_number_of_candidates)
        c_idx = 0 
        for mention_encoded_candidates in encoded_candidates:
            for encoded_candidate in mention_encoded_candidates:
                candidate_token_ids[c_idx] = encoded_candidate.entity_token_ids
                candidate_token_masks[c_idx] = encoded_candidate.entity_token_masks
                c_idx += 1
        return candidate_token_ids,candidate_token_masks

    def get_random_negatives(self,doc):
        encoded_random_negatives = []
        for mention in doc.mentions:
            candidate_pool = self._concept_id_set - set(mention.concept_id)
            m_random_negative_ids = random.sample(candidate_pool, self._num_candidates - 1)
            mention_encoded_random_negatives = []
            for id in m_random_negative_ids:
                if not self._use_hard_negatives:
                    mention_encoded_random_negatives.append(self._entity_encoder(self._entities[id]))
                else:
                    mention_encoded_random_negatives.append(self._all_encoded_entities[id])
                encoded_random_negatives.append(mention_encoded_random_negatives)
        return encoded_random_negatives
    def get_hard_negatives(self,encoded_doc):
        hard_negative_ids = self._all_embedded_entities.get_most_similar(encoded_doc,self._num_candidates)
        encoded_hard_negatives = []
        for mention_hard_negative_ids in hard_negative_ids:
            mention_encoded_hard_negatives = [self._all_encoded_entities[id] for id in mention_hard_negative_ids]
            encoded_hard_negatives.append(mention_encoded_hard_negatives)
        return encoded_hard_negatives

class InputFeatures:
    def __input__(self,doc_token_ids,doc_token_masks):
        self._doc_token_ids = doc_token_ids
        self._doc_token_masks = doc_token_masks

class TrainingInputFeatures(InputFeatures):
    #Input features for a single doc.
    def __init__(self, 
                *args,
                 candidate_token_ids, 
                 candidate_token_masks,
                 label_ids,
                 mention_start_indices, 
                 mention_end_indices,
                 num_mentions, 
                 seq_tag_ids):
        self.super().__init__(*args)
        self.candidate_token_ids = candidate_token_ids
        self.candidate_token_masks = candidate_token_masks
        self.label_ids = label_ids
        self.mention_start_indices = mention_start_indices
        self.mention_end_indices = mention_end_indices
        self.num_mentions = num_mentions
        self.seq_tag_ids = seq_tag_ids

class EvalFeaturizer(Featurizer):
    def featurize_doc(self,doc):
       TrainingInputFeatures
        


 


    
    

    
