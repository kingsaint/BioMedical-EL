
from collections import namedtuple
import random
from el_toolkit.entity_linkers.dual_embedder.featurizer import DualEmbedderEvalFeaturizer, DualEmbedderTrainFeaturizer
from el_toolkit.mpi_utils import partition
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
    def __init__(self,concept_embedder,document_embedder):
        self._concept_embedder = concept_embedder
        self._document_embedder = document_embedder
    def train_featurize(self,docs,lkb,num_hard_negatives=0,num_random_negatives=0,num_max_mentions=8):
        featurizer = DualEncoderTrainFeaturizer(self,lkb,num_hard_negatives,num_random_negatives,num_max_mentions)
        return featurizer.featurize(docs)
    def eval_featurize(self,docs):
        featurizer = DualEncoderEvalFeaturizer(self)
        return featurizer.featurize(docs)
    @property
    def concept_embedder(self):
        return self._concept_embedder
    @property
    def document_embedder(self):
        return self._document_embedder

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
        return list[self._concept_id_to_concept_idx.keys()]
    @property
    def concept_idxs(self):
        return list[self._concept_idx_to_concept_id.keys()]
    def __len__(self):
        return len(self._concept_id_to_concept_idx)
    def __iter__(self):
        return self.concept_ids()
    def __getitem__(self,index):
        return (self._concept_idx_to_concept_id[index])
    def __contains__(self,concept_id):
        return concept_id in self.concept_ids()
    

class EncodedConcepts:
    def __init__(self,encoded_concepts,concept_index):
        self._encoded_concepts = {encoded_concept[0]:encoded_concept[1] for encoded_concept in encoded_concepts}
        self._concept_index = concept_index
    def get_concept_encoding(self,concept_id):
        return self._encoded_concepts[concept_id]
    def __iter__(self):
        return (self._concept_index.get_concept_encoding(concept_id) for concept_id in self._concept_index.__iter__())
    def __getitem__(self,index):
        concept_id = self._concept_index.__getitem__(self,index)
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




class BertConceptEmbedder:
    def __init__(self,lkb,model,tokenizer,max_seq_len,lower_case=False):
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._lower_case = lower_case
        self._model = model 
        self._lkb = lkb
        self._concept_index = list(lkb.get_concept_ids())
    @property
    def max_seq_len(self):
        return self._max_seq_len
    @property
    def tokenizer(self):
        return self._tokenizer
    def get_encodings(self,hvd=None):
        concept_index = self._concept_index
        if hvd:
            concept_index = partition(concept_index,hvd.size(),hvd.rank())
        encoded_concepts =  [(concept_id,self.encode_concept(concept_id)) for concept_id in concept_index]
        if hvd:
            encoded_concepts = hvd.all_gather(encoded_concepts)
        return EncodedConcepts(encoded_concepts,self._concept_index),
    def encode_concept(self,concept_id):
        Encoded_Concept = namedtuple("Encoded_Concept",["token_ids","token_masks"])
        term = self._lkb.get_terms_from_concept_id(concept_id)[0].string
        if self._lower_case:
            term = term.lower()
        max_entity_len = self._max_seq_len // 4  # Number of tokens
        term_tokens = self._tokenizer.tokenize(term)
        if len(term_tokens) > max_entity_len:
            term_tokens = term_token_ids[:max_entity_len]
        term_tokens = [self._tokenizer.cls_token] + term_tokens + [self._tokenizer.sep_token]
        term_token_ids =self._tokenizer.convert_tokens_to_ids(term_tokens)
        token_ids,token_masks = truncate(term_tokens,self._tokenizer.pad_token_id)
        return Encoded_Concept(token_ids,token_masks)
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
        if hasattr(self._model, "module"):
            hidden_size = self._model.module.hidden_size
        else:
            hidden_size = self._model.hidden_size
        return EmbeddedConcepts(embedding_tensor,self._concept_idx,hidden_size)
    def embed_concept_encoding(self,encoded_concept):
        with torch.no_grad():
            candidate_token_ids = torch.LongTensor([encoded_concept.entity_token_ids]).to(self._model.device)
            candidate_token_masks = torch.LongTensor([encoded_concept.entity_tokens_masks]).to(self._model.device)
            candidate_outputs = self._model.bert_candidate.bert(input_ids=candidate_token_ids,attention_mask=candidate_token_masks)
            candidate_embedding = candidate_outputs[1]
            return candidate_embedding
    def get_embedding(self,concept_id):
        return self.get_embedding(self.encode_concept(concept_id))
    def get_embeddings(self,hvd=None):
        encoded_concepts = self.get_encodings(hvd)
        return self.embed_from_concept_encodings(encoded_concepts,hvd=hvd)
    def get_label_ids(self,doc):
        label_ids = []
        for mention in doc.mentions:
            if mention.concept_id in self._concept_index:
                label_ids.append(self._concept_index.get_concept_index(mention.concept_id))
            else:
                label_ids.append(-100)
        return label_ids
    def get_random_negative_concept_ids(self,concept_id,k):
        candidate_pool = self._concept_ids - set(concept_id)
        m_random_negative_ids = random.sample(candidate_pool,k)
        return m_random_negative_ids

class DocumentEmbedder:
    def __init__(self,model,tokenizer,max_seq_len,lower_case=False):
        self._model = model
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._lower_case = lower_case
    @property
    def tokenizer(self):
        return self._tokenizer
    @property
    def max_seq_len(self):
        return self._max_seq_len
    def encode_document(self,doc):
        if self._lower_case:
            doc = doc.lower()
        mention_start_markers = []
        mention_end_markers = []
        tokenized_text = [self._tokenizer.cls_token]
        sequence_tags = []
        prev_end_index = 0
        for m in doc.mentions:
            # Text between the end of last mention and the beginning of current mention
            prefix = m.text
            # Tokenize prefix and add it to the tokenized text
            prefix_tokens = self._tokenizer.tokenize(prefix)
            tokenized_text += prefix_tokens
            # The sequence tag for prefix tokens is 'O' , 'DNT' --> 'Do Not Tag'
            for j, token in enumerate(prefix_tokens):
                sequence_tags.append('O' if not token.startswith('##') else 'DNT')
            # Add mention start marker to the tokenized text
            mention_start_markers.append(len(tokenized_text))
            # tokenized_text += ['[Ms]']
            # Tokenize the mention and add it to the tokenized text
            mention_tokens = self._tokenizer.tokenize(m.text)
            tokenized_text += mention_tokens
            # Sequence tags for mention tokens -- first token B, other tokens I
            for j, token in enumerate(mention_tokens):
                if j == 0:
                    sequence_tags.append('B')
                else:
                    sequence_tags.append('I' if not token.startswith('##') else 'DNT')

            # Add mention end marker to the tokenized text
            mention_end_markers.append(len(tokenized_text) - 1)
            # tokenized_text += ['[Me]']
            # Update prev_end_index
            prev_end_index = m.end_index
        suffix = doc.message[prev_end_index:]
        if len(suffix) > 0:
            suffix_tokens = self._tokenizer.tokenize(suffix)
            tokenized_text += suffix_tokens
            # The sequence tag for suffix tokens is 'O'
            for j, token in enumerate(suffix_tokens):
                sequence_tags.append('O' if not token.startswith('##') else 'DNT')
        tokenized_text += [self._tokenizer.sep_token]
        too_long = len(tokenized_text) > self._max_seq_len
        doc_tokens = self._tokenizer.convert_tokens_to_ids(tokenized_text)
        doc_tokens,doc_tokens_mask = truncate(doc_tokens,self._tokenizer.pad_token_id,self._max_seq_len)
        sequence_tags,_ = truncate(sequence_tags,-100,self._max_seq_len)
        return doc_tokens,doc_tokens_mask,mention_start_markers,mention_end_markers,sequence_tags,too_long
    def get_last_hidden_states(self,doc_token_ids,doc_token_mask):
        # Hard negative candidate mining
        # print("Performing hard negative candidate mining ...")
        # Get mention embeddings
        input_token_ids = torch.LongTensor([doc_token_ids]).to(self._model.device)
        input_token_masks = torch.LongTensor([doc_token_mask]).to(self._model.device)
        # Forward pass through the mention encoder of the dual encoder
        with torch.no_grad():
            if hasattr(self._model, "module"):
                mention_outputs = self._model.module.bert_mention.bert(
                    input_ids=input_token_ids,
                    attention_mask=input_token_masks,
                )
            else:
                mention_outputs = self._model.bert_mention.bert(
                    input_ids=input_token_ids,
                    attention_mask=input_token_masks,
                )
        last_hidden_states = mention_outputs[0]  # B X L X H
        return last_hidden_states
    def get_mention_embeddings(self,doc_token_ids,doc_token_mask,mention_start_markers,mention_end_markers):
        last_hidden_states = self.get_last_hidden_states(doc_token_ids,doc_token_mask)
        mention_embeddings = []
        for i, (s_idx, e_idx) in enumerate(zip(mention_start_markers, mention_end_markers)):
            m_embd = torch.mean(last_hidden_states[:, s_idx:e_idx+1, :], dim=1)
            mention_embeddings.append(m_embd)
        return torch.cat(mention_embeddings, dim=0).unsqueeze(1)
    def get_mention_embeddings_from_doc(self,doc):
        doc_token_ids,doc_token_mask,mention_start_markers,mention_end_markers,_,_ = self.encode_document(doc)
        return self.get_mention_embeddings(doc_token_ids,doc_token_mask,mention_start_markers,mention_end_markers)

