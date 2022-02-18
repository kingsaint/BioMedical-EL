import math

class Featurize:
    def __init__(self,tokenizer):
        self._tokenizer = tokenizer
    def get_mention_window(mention_id, mentions, docs,  max_seq_length, tokenizer):
        max_len_context = max_seq_length - 2 # number of characters
        # Get "enough" context from space-tokenized text.
        content_document_id = mentions[mention_id]['content_document_id']
        context_text = docs[content_document_id]['text']
        start_index = mentions[mention_id]['start_index']
        end_index = mentions[mention_id]['end_index']
        prefix = context_text[max(0, start_index - max_len_context): start_index]
        suffix = context_text[end_index: end_index + max_len_context]
        extracted_mention = context_text[start_index: end_index]

        assert extracted_mention == mentions[mention_id]['text']

        # Get window under new tokenization.
        return get_window(tokenizer.tokenize(prefix),
                        tokenizer.tokenize(extracted_mention),
                        tokenizer.tokenize(suffix),
                        max_len_context)
def get_window(prefix, mention, suffix, max_size):
    if len(mention) >= max_size:
        window = mention[:max_size]
        return window, 0, len(window) - 1

    leftover = max_size - len(mention)
    leftover_half = int(math.ceil(leftover / 2))

    if len(prefix) >= leftover_half:
        prefix_len = leftover_half if len(suffix) >= leftover_half else \
                     leftover - len(suffix)
    else:
        prefix_len = len(prefix)

    prefix = prefix[-prefix_len:]  # Truncate head of prefix
    window = prefix + ['[Ms]'] + mention + ['[Me]'] + suffix
    window = window[:max_size]  # Truncate tail of suffix

    mention_start_index = len(prefix)
    mention_end_index = len(prefix) + len(mention) - 1

    return window, mention_start_index, mention_end_index

    
