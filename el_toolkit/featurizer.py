import math
from pydoc import doc

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
    def __init__(self,tokenizer,max_seq_length=256,lower_case=False):
        self._lower_case = lower_case
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
    def featurize(self,doc,lkb):
        if self._lower_case:
            doc = doc.lower()
        for mention in doc.mentions:
            mention_text = mention.text.lower() if self._lower_case else mention.text.lower()
            max_entity_len = self._max_seq_length // 4  # Number of tokens
            mention_text.lower
            entity_window = get_entity_window(entity_text, max_entity_len, tokenizer)
            # [CLS] candidate text [SEP]
            candidate_tokens = [tokenizer.cls_token] + entity_window + [tokenizer.sep_token]
            candidate_tokens = tokenizer.convert_tokens_to_ids(candidate_tokens)
            if len(candidate_tokens) > max_seq_length:
                candidate_tokens = candidate_tokens[:max_seq_length]
                candidate_masks = [1] * max_seq_length
            else:
                candidate_len = len(candidate_tokens)
                pad_len = max_seq_length - candidate_len
                candidate_tokens += [tokenizer.pad_token_id] * pad_len
                candidate_masks = [1] * candidate_len + [0] * pad_len

            assert len(candidate_tokens) == max_seq_length
            assert len(candidate_masks) == max_seq_length

            all_entity_token_ids.append(candidate_tokens)
            all_entity_token_masks.append(candidate_masks)   


    
    

    
