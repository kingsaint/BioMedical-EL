import math
from pydoc import doc

from el_toolkit.document import BIO_Encoded_Doc

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

class BIO_Encoded_Doc:
    def __init__(self,doc,tokenizer):
        self._tokenized_text = [tokenizer.cls_token]
        self._mention_start_markers = []
        self._mention_end_markers = []
        sequence_tags = []
        prev_end_index = 0
        for m in doc.mentions:
            # Text between the end of last mention and the beginning of current mention
            prefix = doc.message[prev_end_index: m.start_index]
            # Tokenize prefix and add it to the tokenized text
            prefix_tokens = tokenizer.tokenize(prefix)
            tokenized_text += prefix_tokens
            # The sequence tag for prefix tokens is 'O' , 'DNT' --> 'Do Not Tag'
            for j, token in enumerate(prefix_tokens):
                sequence_tags.append('O' if not token.startswith('##') else 'DNT')
            # Add mention start marker to the tokenized text
            self._mention_start_markers.append(len(tokenized_text))
            # tokenized_text += ['[Ms]']
            # Tokenize the mention and add it to the tokenized text
            mention_tokens = tokenizer.tokenize(m.text)
            tokenized_text += mention_tokens
            # Sequence tags for mention tokens -- first token B, other tokens I
            for j, token in enumerate(mention_tokens):
                if j == 0:
                    self._sequence_tags.append('B')
                else:
                    self._sequence_tags.append('I' if not token.startswith('##') else 'DNT')

            # Add mention end marker to the tokenized text
            self.mention_end_markers.append(len(tokenized_text) - 1)
            # tokenized_text += ['[Me]']
            # Update prev_end_index
            prev_end_index = m.end_index
        suffix = doc.message[prev_end_index:]
        if len(suffix) > 0:
            suffix_tokens = tokenizer.tokenize(suffix)
            self.tokenized_text += suffix_tokens
            # The sequence tag for suffix tokens is 'O'
            for j, token in enumerate(suffix_tokens):
                self.sequence_tags.append('O' if not token.startswith('##') else 'DNT')
        self._tokenized_text += [tokenizer.sep_token]
        self._token_ids = tokenizer.convert_tokens_to_ids(self.tokenized_text)
        self._seq_tag_ids = self.convert_tags_to_ids(self.seq_tags)
    @staticmethod
    def convert_tags_to_ids(seq_tags):
        tag_to_id_map = {'O': 0, 'B': 1, 'I': 2, 'DNT': -100}
        seq_tag_ids = [-100]  # corresponds to the [CLS] token
        for t in seq_tags:
            seq_tag_ids.append(tag_to_id_map[t])
        seq_tag_ids.append(-100)  # corresponds to the [SEP] token
        return seq_tag_ids
    def handle_longer_docs(doc):
    @property
    def token_ids(self):
        return self.token_ids
    @property
    def mention_start_markers(self):
        return self.mention_start_markers
    @property
    def mention_end_markers(self):
        return self.mention_end_markers
    @property
    def seq_tag_ids(self):
        return self.tag_ids
    
    
        




        


class Featurizer:#Only works for non-overlapping spans.
    def __init__(self,tokenizer,max_seq_length=256,lower_case=False):
        self._lower_case = lower_case
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
    def featurize(self,docs,lkb):
        for doc in docs:
            doc_bio_encoding = BIO_Encoded_Doc(doc,self.tokenizer)
        if len(doc_bio_encoding) > self._max_seq_length:
            doc_tokens = doc_bio_encoding.token_ids[:self._max_seq_length]
            seq_tag_ids = doc_bio_encoding.seq_tag_ids[:self._max_seq_length]
            doc_tokens_mask = [1] * self._max_seq_length
            num_longer_docs += 1
            continue
            else:
                mention_len = len(doc_tokens)
                pad_len = max_seq_length - mention_len
                doc_tokens += [tokenizer.pad_token_id] * pad_len
                doc_tokens_mask = [1] * mention_len + [0] * pad_len
                seq_tag_ids += [-100] * pad_len
        


 


    
    

    
