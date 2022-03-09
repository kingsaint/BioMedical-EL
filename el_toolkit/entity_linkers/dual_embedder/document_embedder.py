from el_toolkit.entity_linkers.dual_embedder.entity_linker import truncate
import torch
class DocumentEmbedder:
    def __init__(self,span_detector,tokenizer,max_seq_len,lower_case=False):
        self._span_detector = span_detector
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
            prefix = doc.message[prev_end_index:m.start_index]
            # Tokenize prefix and add it to the tokenized text
            prefix_tokens = self._tokenizer.tokenize(prefix)
            tokenized_text += prefix_tokens
            # The sequence tag for prefix tokens is 'O' , 'DNT' --> 'Do Not Tag'
            for j, token in enumerate(prefix_tokens):
                sequence_tags.append('O' if not token.startswith('##') else 'DNT')
            # Add mention start marker to the tokenized text
            mention_start = len(tokenized_text)
            mention_start_markers.append(mention_start)
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
            mention_end = len(tokenized_text) - 1
            # Add mention end marker to the tokenized text
            mention_end_markers.append(mention_end)
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
        sequence_tags = self.convert_tags_to_ids(sequence_tags)
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
        input_token_ids = torch.LongTensor([doc_token_ids]).to(self._span_detector.device)
        input_token_masks = torch.LongTensor([doc_token_mask]).to(self._span_detector.device)
        # Forward pass through the mention encoder of the dual encoder
        with torch.no_grad():
            mention_outputs = self._span_detector(
                    input_ids=input_token_ids,
                    attention_mask=input_token_masks,
                )
        last_hidden_states = mention_outputs[0]  # B X L X H
        return last_hidden_states
    def get_mention_embeddings(self,doc_token_ids,doc_token_mask,mention_start_markers,mention_end_markers):
        last_hidden_states = self.span_detector.get_last_hidden_states(doc_token_ids,doc_token_mask)
        return self.span_detector.pool_mentions(mention_start_markers,mention_end_markers,doc_token_ids,doc_token_mask)
    def get_mention_embeddings_from_doc(self,doc):
        doc_token_ids,doc_token_mask,mention_start_markers,mention_end_markers,_,_ = self.encode_document(doc)
        return self.get_mention_embeddings(doc_token_ids,doc_token_mask,mention_start_markers,mention_end_markers)
    @staticmethod
    def convert_tags_to_ids(seq_tags):
        tag_to_id_map = {'O': 0, 'B': 1, 'I': 2, 'DNT': -100}
        seq_tag_ids = [-100]  # corresponds to the [CLS] token
        for t in seq_tags:
            seq_tag_ids.append(tag_to_id_map[t])
        seq_tag_ids.append(-100)  # corresponds to the [SEP] token
        return seq_tag_ids