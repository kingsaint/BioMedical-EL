from collections import namedtuple
from el_toolkit.mpi_utils import partition
import torch
from torch import TensorDataset

EvalInputFeatures = namedtuple("EvalInputFeatures",["doc_token_ids","doc_token_masks","mention_start_indices","mention_end_indices","label_ids","num_mentions","seq_tag_ids"])
TrainingInputFeatures = namedtuple("TrainingInputFeatures",["doc_token_ids","doc_token_masks","mention_start_indices", "mention_end_indices","label_ids","num_mentions","seq_tag_ids""candidate_token_ids","candidate_token_masks"])

class DualEmbedderFeaturizer:
    def __init__(self,dual_embedder,hvd=None):
        self._dual_embedder = dual_embedder
        self._hvd = hvd
    def featurize(self,docs):
        if self._hvd:#distributed
            docs = partition(docs,self._hvd.size(),self._hvd.rank())
        features = [self.train_featurize_doc(doc) for doc in docs]
        if self._hvd:
            features = self._hvd.all_gather(features)#list of input_features objects
        return self.get_tensor_dataset(features)
    def pad_mention_indices(self,mention_start_indices,mention_end_indices):
        # Pad the mention start and end indices
        padded_mention_start_indices = [0] * self._num_max_mentions
        padded_mention_end_indices = [0] * self._num_max_mentions
        num_mentions = len(mention_start_indices)
        if num_mentions > 0: 
            if num_mentions <= self._num_max_mentions:
                padded_mention_start_indices[:num_mentions] = mention_start_indices
                padded_mention_end_indices[:num_mentions] = mention_end_indices
            else:
                padded_mention_start_indices = mention_start_indices[:self._num_max_mentions]
                padded_mention_end_indices = mention_end_indices[:self._num_max_mentions]
        return mention_start_indices,mention_end_indices
    def get_tensor_dataset(feature_list):
        grouped_features = list(zip(*feature_list))
        tensors = [torch.tensor(feature_group, dtype=torch.long) for feature_group in grouped_features]  
        return TensorDataset(*tensors)
class DualEmbedderTrainFeaturizer(DualEmbedderFeaturizer):
    def __init__(self,*args,lkb,num_hard_negatives=0,num_random_negatives=0,num_max_mentions=8,**kwargs):
        super().__init__(*args,**kwargs)
        self._lkb = lkb
        self._total_number_of_candidates = num_random_negatives + num_hard_negatives + 1
        self._num_hard_negatives = num_hard_negatives
        self._num_random_negatives = num_random_negatives
        self._num_max_mentions = num_max_mentions
        self._max_entity_len =  self._dual_embedder.concept_embedder.max_sequence_len//4
        if num_hard_negatives:
            self._encoded_concepts = self._dual_embedder.concept_embedder._concept_encoder.from_lkb(lkb,self._hvd)
            self._embeddings = self._dual_embedder.concept_embedder.embed_from_concept_encodings(self._encoded_concepts,self._hvd)
    def featurize_doc(self,doc):
        num_mentions = len(doc.mentions)
        if self._lower():
                doc = doc.lower()
        doc_token_ids,doc_token_mask,mention_start_indices,mention_end_indices,seq_tag_ids,too_long  = self._dual_embedder.document_embedder.encode_document(self,doc)
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
        mention_start_indices,mention_end_indices = self.pad_mention_indices(mention_start_indices,mention_end_indices)
        candidate_token_ids,candidate_token_masks = self.get_candidate_tokens(candidate_id_lists)
        label_ids = self._dual_embedder.concept_embedder.get_label_ids(doc)
        return TrainingInputFeatures(doc_token_ids,doc_token_mask,mention_start_indices,mention_end_indices,label_ids,num_mentions,seq_tag_ids,candidate_token_ids,candidate_token_masks)
    def get_candidate_tokens(self,candidate_id_lists):
        candidate_token_ids = [[self._dual_embedder.concept_embedder.tokenizer.pad_token_id] * self._max_entity_len] * (self._num_max_mentions * self._total_number_of_candidates)
        candidate_token_masks= [[0] * self._max_entity_len] * (self._num_max_mentions * self._total_number_of_candidates)
        c_idx = 0 
        for candidate_id_list in candidate_id_lists:
            for candidate_id in candidate_id_list:
                if self._encoded_concepts:
                    encoded_candidate = self._encoded_concepts.get_concept_encoding(candidate_id)
                else:
                    encoded_candidate = self._dual_embedder.concept_embedder.encode_concept(candidate_id)
                candidate_token_ids[c_idx] = encoded_candidate.token_ids
                candidate_token_masks[c_idx] = encoded_candidate.token_masks
                c_idx += 1
        return candidate_token_ids,candidate_token_masks
class DualEmbedderEvalFeaturizer(DualEmbedderFeaturizer):
    def featurize_doc(self,doc):
        num_mentions = len(doc.mentions)
        if self._lower():
                doc = doc.lower()
        doc_token_ids,doc_token_mask,mention_start_indices,mention_end_indices,seq_tag_ids,too_long  = self._dual_embedder.document_embedder.encode_document(self,doc)
        mention_start_indices,mention_end_indices = self.pad_mention_indices(mention_start_indices,mention_end_indices)
        label_ids = self._dual_embedder.concept_embedder.get_label_ids(doc)
        return EvalInputFeatures(doc_token_ids,doc_token_mask,mention_start_indices,mention_end_indices,label_ids,num_mentions,seq_tag_ids)
