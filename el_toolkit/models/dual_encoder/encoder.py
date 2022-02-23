from collections import namedtuple
import functools

from el_toolkit.mpi_utils import partition



class Encoder:
    def __init__(self,tokenizer,max_seq_length):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
    def truncate(self,token_ids):
        if len(token_ids) > self._max_seq_length:
            token_ids = token_ids[:self._max_seq_length]
            tokens_mask = [1] * self._max_seq_length
        else:
            token_number = len(token_ids)
            pad_len = self._max_seq_length - token_number
            token_ids += [self._tokenizer.pad_token_id] * pad_len
            tokens_mask = [1] * token_number + [0] * pad_len
        return token_ids,tokens_mask


class Document_Encoder(Encoder):
    def encode_doc(self,doc):
        Encoded_Document = namedtuple("Encoded_Document",["token_ids","doc_tokens_mask","mention_start_markers","mention_end_markers","bio_tag_ids","too_long","num_mentions"])
        tokenized_text,mention_start_markers,mention_end_markers,bio_tags = self.get_bio_encoding(doc)
        doc_token_ids = self._tokenizer.convert_tokens_to_ids(tokenized_text)
        if len(doc_token_ids) > self._max_seq_length:
            too_long = True
        truncated_token_ids,doc_tokens_mask = self.truncate(doc_token_ids,self._tokenizer.pad_token_id)
        truncated_bio_tag_ids,_ = self.truncate(bio_tags,-100)
        num_mentions = len(doc.mentions)
        return Encoded_Document(truncated_token_ids,doc_tokens_mask,mention_start_markers,mention_end_markers,truncated_bio_tag_ids,too_long,num_mentions)
    def get_bio_encoding(self,doc):
        mention_start_markers = []
        mention_end_markers = []
        tokenized_text = [self._tokenizer.cls_token]
        sequence_tags = []
        prev_end_index = 0
        for m in doc.mentions:
            # Text between the end of last mention and the beginning of current mention
            prefix = doc.message[prev_end_index: m.start_index]
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
        return tokenized_text,mention_start_markers,mention_end_markers,sequence_tags
    @staticmethod
    def convert_tags_to_ids(seq_tags):
        tag_to_id_map = {'O': 0, 'B': 1, 'I': 2, 'DNT': -100}
        seq_tag_ids = [-100]  # corresponds to the [CLS] token
        for t in seq_tags:
            seq_tag_ids.append(tag_to_id_map[t])
        seq_tag_ids.append(-100)  # corresponds to the [SEP] token
        return seq_tag_ids

class Entity_Encoder(Encoder):
    def encode_entity(self,entity):
        Encoded_Entity = namedtuple("Encoded_Entity",["entity","enitity_token_ids","entity_token_masks"])
        max_entity_len = self._max_seq_length // 4  # Number of tokens
        entity_tokens = self._tokenizer.tokenize(entity.term_text)
        if len(entity_tokens) > max_entity_len:
            entity_tokens = entity_token_ids[:max_entity_len]
        entity_tokens = [self._tokenizer.cls_token] + entity_tokens + [self._tokenizer.sep_token]
        entity_token_ids = self._tokenizer.convert_tokens_to_ids(entity_tokens)
        truncated_entity_token_ids,entity_token_masks = self.truncate(entity_tokens)
        return Encoded_Entity(entity,truncated_entity_token_ids,entity_token_masks)

    def encode_all_entities(self,entities):#distributed
        entity_keys = partition(entities.keys(),self._hvd.size(),self._hvd.rank())
        encoded_entities = {entity.concept_id:self._entity_encoder.encode_entity(entity) for entity in entities}
        _all_entity_encodings = self._hvd.allgather(encoded_entities)
        all_entity_encodings = functools.reduce(lambda dict_1,dict_2: {**dict_1,**dict_2},_all_entity_encodings)
        return all_entity_encodings