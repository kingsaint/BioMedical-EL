from collections import namedtuple
from el_toolkit.document import Document
from el_toolkit.entity_linkers.dual_encoder.entity_embedder import Dual_Encoder_Entity_Embedder
from el_toolkit.mpi_utils import partition

EvalInputFeatures = namedtuple("EvalInputFeatures",["doc_token_ids","doc_token_masks","label_ids","mention_start_indices", "mention_end_indices","num_mentions","seq_tag_ids"])
TrainingInputFeatures = namedtuple("TrainingInputFeatures",["doc_token_ids","doc_token_masks","label_ids","mention_start_indices", "mention_end_indices","num_mentions","seq_tag_ids""candidate_token_ids","candidate_token_masks"])
EvalInputFeatures = namedtuple("EvalInputFeatures",["doc_token_ids","doc_token_masks"])


class DualEncoderFeaturizer:
    def __init__(self,dual_encoder,lkb):
        self._dual_encoder = dual_encoder
        self._lkb = lkb
    def featurize(self,docs,hvd=None):
        if hvd:#distributed
            docs = partition(docs,hvd.size(),hvd.rank())
        features = [self.featurize_doc(doc) for doc in docs]
        if hvd:
            features = hvd.all_gather(features)#list of input_features objects
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
    def pad_candidate_tokens(self,encoded_candidates):
        ##encode_candidates
        candidate_token_ids = [[self._dual_embedder.concept_embedder.tokenizer.pad_token_id] * self._max_entity_len] * (self._num_max_mentions * self._total_number_of_candidates)
        candidate_token_masks= [[0] * self._max_entity_len] * (self._num_max_mentions * self._total_number_of_candidates)
        c_idx = 0 
        for mention_encoded_candidates in encoded_candidates:
            for encoded_candidate in mention_encoded_candidates:
                candidate_token_ids[c_idx] = encoded_candidate[0]
                candidate_token_masks[c_idx] = encoded_candidate[1]
                c_idx += 1
        return candidate_token_ids,candidate_token_masks
    def get_tensor_dataset(feature_list):
        grouped_features = list(zip(*feature_list))
        tensors = [torch.tensor(feature_group, dtype=torch.long) for feature_group in grouped_features]  
        return TensorDataset(*tensors)
   

class DualEncoderTrainFeaturizer(DualEncoderFeaturizer):
    def __init__(self,*args,num_hard_negatives=0,num_random_negatives=0,num_max_mentions=8):
        super().__init__(*args)
        self._total_number_of_candidates = num_random_negatives + num_hard_negatives + 1
        self._num_hard_negatives = num_hard_negatives
        self._num_random_negatives = num_random_negatives
        self._num_max_mentions = num_max_mentions
        self._max_entity_len =  self._dual_encoder.concept_embedder.max_sequence_len//4
        if num_hard_negatives:
            self._encoded_concepts = self._dual_encoder.concept_embedder._concept_encoder.from_lkb(lkb,hvd=hvd)
            self._embeddings = self._dual_encoder.concept_embedder.embed_from_concept_encodings(self._encoded_concepts,hvd=hvd)
    def train_featurize_doc(self,doc):
        num_mentions = len(doc.mentions)
        if self._lower():
                doc = doc.lower()
        doc_token_ids,doc_token_mask,mention_start_markers,mention_end_markers,seq_tag_ids,too_long  = self._dual_encoder.document_embedder.encode_document(self,doc)
        
        candidate_token_ids,candidate_token_masks = self.pad_candidate_tokens(encoded_candidates)
        mention_start_indices,mention_end_indices = self.pad_mention_indices(mention_start_indices,mention_end_indices)
        label_ids = self._dual_encoder.concept_embedder.get_label_ids(doc)
        return TrainingInputFeatures(doc.token_ids,doc.doc_tokens_mask,candidate_token_ids,candidate_token_masks,label_ids,mention_start_indices,mention_end_indices,num_mentions,seq_tag_ids)
    def get_candidate_ids(self,doc_token_ids,doc_token_mask,mention_start_markers,mention_end_markers):
        if self._num_hard_negatives:
            mention_embeddings = self.get_mention_embeddings(self,doc_token_ids,doc_token_mask,mention_start_markers,mention_end_markers)
            hard_negative_ids = self._embeddings.get_most_similar(mention_embeddings,self._num_hard_negatives)
        if self._num_random_negatives:
            random_negative_ids = [self.get_random_negative_concept_ids(mention.concept_i,self._num_random_negatives) for mention in doc.mentions]
        candidate_ids = []
        for i in range(len(doc.mentions)):
            candidate_ids.append([doc.mentions[i]]+[hard_negative_ids[i]]+[random_negative_ids[i]])
        if self._encoded_concepts:
            encoded_candidates = [self._encoded_concepts.get_concept_encoding[candidate_id] for candidate_id in candidate_ids]
        else:
            encoded_candidates = [self._dual_encoder.concept_embedder.encode_concept(candidate_id) for candidate_id in candidate_ids]
        
    