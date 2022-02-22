from collections import namedtuple
import math
from el_toolkit.models.dual_encoder.document_embedder import Document_Embedder
from el_toolkit.models.dual_encoder.encoder import Encoder
from el_toolkit.models.dual_encoder.entity_embedder import Entity_Embedder
from el_toolkit.mpi_utils import partition
import torch
from pydoc import doc
import random
comm = None

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, mention_token_ids, mention_token_masks,
                 candidate_token_ids_1, candidate_token_masks_1,
                 candidate_token_ids_2, candidate_token_masks_2,
                 label_ids, mention_start_indices, mention_end_indices,
                 num_mentions, seq_tag_ids):
        self.mention_token_ids = mention_token_ids
        self.mention_token_masks = mention_token_masks
        self.candidate_token_ids_1 = candidate_token_ids_1
        self.candidate_token_masks_1 = candidate_token_masks_1
        self.candidate_token_ids_2 = candidate_token_ids_2
        self.candidate_token_masks_2 = candidate_token_masks_2
        self.label_ids = label_ids
        self.mention_start_indices = mention_start_indices
        self.mention_end_indices = mention_end_indices
        self.num_mentions = num_mentions
        self.seq_tag_ids = seq_tag_ids

class Featurizer:#Only works for non-overlapping spans.
    def __init__(self,tokenizer,model,hr,hvd,num_candidates=8,max_seq_length=256,lower_case=False):
        self._lower_case = lower_case
        self._num_candidates
        self._model = model
        if self._comm is None:
            self._comm = MPI.COMM_WORLD
        self._hvd = hvd
        self._encoder = Encoder(tokenizer,max_seq_length)
        self._entity_embedder =  Entity_Embedder(model,tokenizer)
        self._document_embedder = Document_Embedder(model,tokenizer)

    def featurize(self,docs,entities):
        num_longer_docs += 1
        features = []
        num_longer_docs = 0
        all_concept_ids = {entity.concept_id for entity in entities}
        entity_partition = partition(entities,self._hvd.size(),self._hvd.rank())
        encoded_entities = self.encode_entities(entity_partition)
        entity_embeddings = self.embed_entities(encoded_entities)
        all_entity_embeddings = self._hvd.allgather(entity_embeddings)
        for doc in docs:
            if encodings.too_long:
                continue
            else:
                self.featurize_doc(doc,encodings)
                if encoded_doc.num_mentions > 0:

    def featurize_doc(self,doc,encoded_doc):
        # Build list of candidates
        label_candidate_ids = []
        # Number of mentions in the documents
        num_mentions = len(doc.mentions)
        
    def encode_entities(self,entities):
        return [self._encoder.encode_entity(entity) for entity in entities] 

    def embed_entities(self,encoded_entities):
        return [self._entity_embedder.get_candidate_embeddings(encoded_entities.entity_token_ids,encoded_entities.entity_token_masks)]

    def get_random_negatives(self,mention,all_concept_ids):
        m_candidates = []
        candidate_pool = all_concept_ids - set(mention.concept_id)
        negative_candidates = random.sample(candidate_pool, self.num_candidates - 1)
        m_candidates += negative_candidates
        return m_candidates

    def get_hard_negatives(self,encoded_doc,all_entity_embeddings):
        

        candidates_2 = []
        # candidates_2.append(label_candidate_id)  # positive candidate
        # Append hard negative candidates
        for m_idx, m in enumerate(mentions[document_id]):
            if len(mention_hard_negatives[mention_id]) < args.num_candidates:  # args.num_candidates - 1
                m_hard_candidates = mention_hard_negatives[mention_id]
            else:
                candidate_pool = mention_hard_negatives[mention_id]
                m_hard_candidates = random.sample(candidate_pool, args.num_candidates)  # args.num_candidates - 1
            candidates_2.append(m_hard_candidates)


        

        


 


    
    

    
