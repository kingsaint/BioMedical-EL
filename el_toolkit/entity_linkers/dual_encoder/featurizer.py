from argparse import ArgumentError
from collections import namedtuple
from el_toolkit.entity_linkers.dual_encoder.document_embedder import Document_Embedder
from el_toolkit.entity_linkers.dual_encoder.encoder import Document_Encoder, Entity,Entity_Encoder
from el_toolkit.entity_linkers.dual_encoder.entity_embedder import Entity_Embedder
from el_toolkit.mpi_utils import partition
import random
comm = None
#define different input features for different needs.
class Featurizer:#Only works for non-overlapping spans.
    def __init__(self,tokenizer,entities,hvd = None,max_seq_length=256,lower_case=False):
        self._lower_case = lower_case
        self._hvd = hvd
        self._distributed = self._hvd != None
        self._tokenizer = tokenizer
        self._entities = entities
        self._entity_index = {entity.concept_id:idx for idx,entity in enumerate(entities)}
        self._entity_encoder = Entity_Encoder(tokenizer,max_seq_length,self._hvd)
        self._document_encoder = Document_Encoder(tokenizer,max_seq_length)
    def featurize(self,docs):
        if self._distributed:#distributed
            docs = partition(docs,self._hvd.size(),self._hvd.rank())
        if self._lower_case:
            docs = [doc.lower() for doc in docs]
        features = [self.featurize_doc(doc) for doc in docs]
        if self._distributed:
            features = self._hvd.all_gather(features)#list of input_features objects
        return self.get_tensor_dataset(features)
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
        
    def get_tensor_dataset(feature_list):
        grouped_features = list(zip(*feature_list))
        tensors = [torch.tensor(feature_group, dtype=torch.long) for feature_group in grouped_features]  
        return TensorDataset(*tensors)
    
    def get_label_ids(self,doc):
        label_ids = []
        for mention in doc.mentions:
            if mention.concept_id in self.all_concept_ids:
                label_ids.append(self._entities(self._entity_index(mention.concept_id)))
            else:
                label_ids.append(-100)
        return label_ids
    def featurize_doc(self,doc):
        raise NotImplementedError


class TrainFeaturizer(Featurizer):
    def __init__(self,*args,num_hard_negatives=0,num_random_negatives=0,model=None,**kwargs):
        super().__init__(*args,**kwargs)
        self._document_embedder = Document_Embedder(model,self._hvd)
        self._all_concept_ids = {entity.concept_id for entity in self._entities}
        self._num_hard_negatives = num_hard_negatives
        self._num_random_negatives = num_random_negatives
        self._use_hard_negatives = bool(num_hard_negatives)
        self._use_random_negatives = bool(num_random_negatives)
        self._total_number_of_candidates = self._num_random_negatives + self._num_hard_negatives + 1
        if self._use_hard_negatives:
            self._entity_embedder =  Entity_Embedder(model,self._hvd)
            self._encoded_entities = self._entity_encoder.encode_all_entities(self._entities,self._hvd)
            self._all_embedded_entities = self._entity_embedder.embed_all_entities(self._encoded_entities,self._hvd)
        else:
            self._encoded_entities = None
            self._all_embedded_entities = None
    #entities is a dictionary from concept_id to term
    def featurize_doc(self,doc):
        if self._lower_case:
            doc = doc.lower()
        encoded_doc = self._document_encoder.encode_doc(doc)
        if self._encoded_entities:
            encoded_positive_entities = [self._encoded_entities[self._entity_index[mention.concept_id]] for mention in doc.mentions]
        else:
            encoded_positive_entities = [self._entity_encoder.encode_entity(self._entities[self._entity_index[mention.concept_id]]) for mention in doc.mentions]
        encoded_candidates = [encoded_positive_entities]
        if self._use_random_negatives:
            encoded_random_negatives = self.get_random_negatives(doc)
            encoded_candidates.extend(encoded_random_negatives)
        if self._encoded_entities:
            encoded_hard_negatives = self.get_hard_negatives(encoded_doc)
            encoded_candidates.extend(encoded_hard_negatives)
        candidate_token_ids,candidate_token_masks = self.create_mention_index_matrices(encoded_candidates)
        mention_start_indices,mention_end_indices = self._create_mention_index_matrices(encoded_doc)
        label_ids = self.get_label_ids(doc)
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
            candidate_pool = self._all_concept_ids - set(mention.concept_id)
            m_random_negative_ids = random.sample(candidate_pool, self._num_random_negatives)
            mention_encoded_random_negatives = []
            for id in m_random_negative_ids:
                if not self._use_hard_negatives:
                    mention_encoded_random_negatives.append(self._entity_encoder(self._entities[self._entity_index[id]]))
                else:
                    mention_encoded_random_negatives.append(self._encoded_entities[id])
                encoded_random_negatives.append(mention_encoded_random_negatives)
        return encoded_random_negatives
    def get_hard_negatives(self,encoded_doc):
        hard_negative_ids = self._all_embedded_entities.get_most_similar(encoded_doc,self._num_hard_negatives)
        encoded_hard_negatives = []
        for mention_hard_negative_ids in hard_negative_ids:
            mention_encoded_hard_negatives = [self._encoded_entities[id] for id in mention_hard_negative_ids]
            encoded_hard_negatives.append(mention_encoded_hard_negatives)
        return encoded_hard_negatives

