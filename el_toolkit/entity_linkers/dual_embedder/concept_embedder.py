from collections import namedtuple
from el_toolkit.entity_linkers.dual_embedder.entity_linker import truncate
import torch
import random


class ConceptIndex:
    def __init__(self,concept_ids):
        self._concept_id_to_concept_idx = {concept_id:idx for idx,concept_id in enumerate(concept_ids)}
        self._concept_idx_to_concept_id = {idx:concept_id for idx,concept_id in enumerate(concept_ids)}
    def get_concept_idx(self,concept_id):
        return self._concept_id_to_concept_idx[concept_id]
    def get_concept_id(self,concept_idx):
        return self._concept_idx_to_concept_id[concept_idx]
    @property
    def concept_ids(self):
        return set(self._concept_id_to_concept_idx.keys())
    @property
    def concept_idxs(self):
        return set(self._concept_idx_to_concept_id.keys())
    def __len__(self):
        return len(self._concept_id_to_concept_idx)
    def __iter__(self):
        return (self._concept_idx_to_concept_id for i in range(len(self)))
    def __getitem__(self,index):
        return (self._concept_idx_to_concept_id[index])
    def __contains__(self,concept_id):
        return concept_id in self.concept_ids
    

class EncodedConcepts:
    def __init__(self,encoded_concepts,concept_index):
        self._encoded_concepts = {encoded_concept[0]:encoded_concept[1] for encoded_concept in encoded_concepts}
        self._concept_index = concept_index
    def get_concept_encoding(self,concept_id):
        return self._encoded_concepts[concept_id]
    def __iter__(self):
        return (self._concept_index.get_concept_encoding(concept_id) for concept_id in self._concept_index.__iter__())
    def __getitem__(self,index):
        concept_id = self._concept_index[index]
        return self._encoded_concepts[concept_id]


class EmbeddedConcepts:
    def __init__(self,embedding_tensor,concept_index,hidden_size):
        # concept_ids = list[concept_id_to_embedding_index.keys()]
        self._embedding_tensor = embedding_tensor
        self._concept_idx = concept_index
        self._hidden_size = hidden_size
    def get_most_similar_concept_ids(self,mention_embeddings,k):
        num_m = mention_embeddings.size(0)  #
        all_candidate_embeddings_ = self._entity_embeddings_tensor.unsqueeze(0).expand(num_m, -1, self._hidden_size) # M X C_all X H

        # candidate_indices = candidate_indices[0]  # original size 1 X 10 -> 10
        similarity_scores = torch.bmm(mention_embeddings,
                                    all_candidate_embeddings_.transpose(1, 2))  # M X 1 X C_all
        similarity_scores = similarity_scores.squeeze(1)  # M X C_all
        # print(similarity_scores)
        _, candidate_indices = torch.topk(similarity_scores, k)
        most_similar_concept_ids = []
        for mention_candidate_indices in candidate_indices:
            most_similar_concept_ids.append([self._concept_idx_to_concept_id[cidx] for cidx in mention_candidate_indices])
        return most_similar_concept_ids
    def get_embedding(self,concept_id):
        concept_idx = self._concept_index.get_concept_idx(concept_id)
        return self._embedding_tensor[concept_idx]
    def __getitem__(self,index):
        return self._embedding_tensor[index]

class ConceptEmbedder:
    def __init__(self,lkb): 
        self._lkb = lkb
        self._concept_ids = lkb.get_concept_ids()
        self._concept_index = ConceptIndex(list(self._concept_ids))
    def get_encodings(self,hvd=None):
        concept_index = self._concept_index
        if hvd:
            concept_index = partition(concept_index,hvd.size(),hvd.rank())
        encoded_concepts =  [(concept_id,self.encode_concept(concept_id)) for concept_id in concept_index]
        if hvd:
            encoded_concepts = hvd.all_gather(encoded_concepts)
        return EncodedConcepts(encoded_concepts,self._concept_index)
    def embed_from_concept_encodings(self,encoded_concepts,hvd=None):
        if hvd:
            encoded_concepts = partition(encoded_concepts,hvd.size(),hvd.rank())
        candidate_embeddings = []
        for encoded_concept in encoded_concepts:
            candidate_embeddings.append(self.embed_concept_encoding(encoded_concept))
                #logger.info(str(candidate_embedding))
        if hvd:
            candidate_embeddings = hvd.all_gather(candidate_embeddings)
        embedding_tensor = torch.cat(candidate_embeddings, dim=0)
        #logger.info("INFO: Collected candidate embeddings.")
        return EmbeddedConcepts(embedding_tensor,self._concept_idx,self.hidden_size)
    def get_embedding(self,concept_id):
        return self.get_embedding(self.encode_concept(concept_id))
    def get_embeddings(self,hvd=None):
        encoded_concepts = self.get_encodings(hvd)
        return self.embed_from_concept_encodings(encoded_concepts,hvd=hvd)
    def get_label_ids(self,doc,num_max_mentions):
        label_ids = [-1] * num_max_mentions
        for i,mention in enumerate(doc.mentions):
            if i < num_max_mentions:
                if mention.concept_id in self._concept_index:
                    label_ids[i] = 1
        return label_ids
    def get_random_negative_concept_ids(self,concept_id,k):
        candidate_pool = self._concept_ids - set(concept_id)
        m_random_negative_ids = random.sample(candidate_pool,k)
        return m_random_negative_ids

class BertConceptEmbedder(ConceptEmbedder):
    def __init__(self,*args,bert_model,tokenizer,max_seq_len,lower_case=False):
        super().__init__(*args)
        self._bert_model = bert_model 
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._lower_case = lower_case
        if hasattr(self._bert_model, "module"):
            self._hidden_size = self._bert_model.module.config.hidden_size
        else:
            self._hidden_size = self._bert_model.config.hidden_size
    @property
    def max_seq_len(self):
        return self._max_seq_len
    @property
    def tokenizer(self):
        return self._tokenizer
    def encode_concept(self,concept_id):
        Encoded_Concept = namedtuple("Encoded_Concept",["token_ids","token_masks"])
        term = self._lkb.get_terms_from_concept_id(concept_id)[0].string
        if self._lower_case:
            term = term.lower()
        max_entity_len = self._max_seq_len // 4  # Number of tokens
        term_tokens = self._tokenizer.tokenize(term)
        if len(term_tokens) > max_entity_len:
            term_tokens = term_tokens[:max_entity_len]
        term_tokens = [self._tokenizer.cls_token] + term_tokens + [self._tokenizer.sep_token]
        term_token_ids =self._tokenizer.convert_tokens_to_ids(term_tokens)
        term_token_ids,term_token_masks = truncate(term_token_ids,self._tokenizer.pad_token_id,max_entity_len)
        return Encoded_Concept(term_token_ids,term_token_masks)
    def embed_concept_encoding(self,encoded_concept):
        with torch.no_grad():
            candidate_token_ids = torch.LongTensor([encoded_concept.entity_token_ids]).to(self._bert_model.device)
            candidate_token_masks = torch.LongTensor([encoded_concept.entity_tokens_masks]).to(self._bert_model.device)
            candidate_outputs = self._bert_model(input_ids=candidate_token_ids,attention_mask=candidate_token_masks)
            candidate_embedding = candidate_outputs[1]
            return candidate_embedding

    
