import math
from pydoc import doc

class Featurizer:
    def __init__(self,tokenizer,lower_case=False):
        self.lower_case = lower_case
        self._tokenizer = tokenizer
    def get_marked_mentions(self,doc,tokenizer):
        context_text = doc.message.lower() if self.lower_case else doc.message
        tokenized_text = [tokenizer.cls_token]
        mention_start_markers = []
        mention_end_markers = []
        sequence_tags = []

        # print(len(context_text))
        # print(len(mentions[document_id]))
        prev_end_index = 0
        for m in doc:
            if m.start_index >= len(context_text):
                continue
            extracted_mention = context_text[m.start_index: m.end_index]
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
            prev_end_index = end_index

        suffix = context_text[prev_end_index:]
        if len(suffix) > 0:
            suffix_tokens = tokenizer.tokenize(suffix)
            tokenized_text += suffix_tokens
            # The sequence tag for suffix tokens is 'O'
            for j, token in enumerate(suffix_tokens):
                sequence_tags.append('O' if not token.startswith('##') else 'DNT')
        tokenized_text += [tokenizer.sep_token]

        return tokenized_text, mention_start_markers, mention_end_markers, sequence_tags

    
