from collections import namedtuple
import dataclasses
from dataclasses import dataclass
import functools

from el_toolkit.mpi_utils import partition


def truncate(token_ids,pad_token_id,max_seq_length):
    if len(token_ids) > max_seq_length:
        token_ids = token_ids[:max_seq_length]
        tokens_mask = [1] * max_seq_length
    else:
        token_number = len(token_ids)
        pad_len = max_seq_length - token_number
        token_ids += [pad_token_id] * pad_len
        tokens_mask = [1] * token_number + [0] * pad_len
    return token_ids,tokens_mask

class TermEncoder(Encoder):
    def encode_term(self,term):
        Encoded_Term = namedtuple("Encoded_Term",["term_token_ids","term_token_masks"])
        max_entity_len = self._max_seq_length // 4  # Number of tokens
        entity_tokens = self._tokenizer.tokenize(term)
        if len(entity_tokens) > max_entity_len:
            entity_tokens = entity_token_ids[:max_entity_len]
        entity_tokens = [self._tokenizer.cls_token] + entity_tokens + [self._tokenizer.sep_token]
        entity_token_ids = self._tokenizer.convert_tokens_to_ids(entity_tokens)
        truncated_entity_token_ids,entity_token_masks = self.truncate(entity_tokens,self._tokenizer.pad_token_id)
        return Encoded_Term(truncated_entity_token_ids,entity_token_masks)
    def encode_all_entities(self,entities):
        if self._distributed:
            entities =  partition(entities,self._hvd.size(),self._hvd.rank())
        entity_embeddings = [self.embed_entity(entity) for entity in entities]
        if self._distributed:
            entity_embeddings = self._hvd.allgather(entity_embeddings)
        return entity_embeddings