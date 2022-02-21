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
    def __init__(self,doc,tokenizer)
        tokenized_text = [tokenizer.cls_token]
        mention_start_markers = []
        mention_end_markers = []
        sequence_tags = []
        prev_end_index = 0
        for m in self.mentions:
            extracted_mention = m.text
            # Text between the end of last mention and the beginning of current mention
            prefix = context_text[prev_end_index: m.start_index]
            # Tokenize prefix and add it to the tokenized text
            prefix_tokens = tokenizer.tokenize(prefix)
            tokenized_text += prefix_tokens
            # The sequence tag for prefix tokens is 'O' , 'DNT' --> 'Do Not Tag'
            for j, token in enumerate(prefix_tokens):
                sequence_tags.append('O' if not token.startswith('##') else 'DNT')
            # Add mention start marker to the tokenized text
            mention_start_markers.append(len(tokenized_text))
            # tokenized_text += ['[Ms]']
            # Tokenize the mention and add it to the tokenized text
            mention_tokens = tokenizer.tokenize(extracted_mention)
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

        suffix = context_text[prev_end_index:]
        if len(suffix) > 0:
            suffix_tokens = tokenizer.tokenize(suffix)
            tokenized_text += suffix_tokens
            # The sequence tag for suffix tokens is 'O'
            for j, token in enumerate(suffix_tokens):
                sequence_tags.append('O' if not token.startswith('##') else 'DNT')
        tokenized_text += [tokenizer.sep_token]
        return BIO_Encoded_Doc(tokenized_text, mention_start_markers, mention_end_markers, sequence_tags)


class Featurizer:#Only works for non-overlapping spans.
    def __init__(self,tokenizer,max_seq_length=256,lower_case=False):
        self._lower_case = lower_case
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
    def featurize(self,doc,lkb):
        bio_encoding = doc.bio_encoded

 


    
    

    
