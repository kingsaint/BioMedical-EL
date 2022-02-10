from __future__ import annotations
from data import Document,Mention
from dataclasses import dataclass
from el_toolkit.lexical_knowledge_base import Knowledge_Data

def segment_document(tokenizer,max_mention_per_new_doc,doc:Document) -> Document:
    def segment_recursive(remaining_text,remaining_mentions,doc_id=0,prev_seq_len=0):
        omitted_mentions = 0
        new_document_mentions = []
        mentions_added = 0 
        segment_text = ""
        while mentions_added < len(remaining_mentions):
            mention_to_add =  remaining_mentions[mentions_added]
            tentative_segment_text = remaining_text[:mention_to_add.end_index]
            tokens = tokenizer.tokenize(tentative_segment_text)
            if (len(new_document_mentions) != max_mention_per_new_doc and len(['[CLS]'] + tokens + ['[SEP]']) < 256):
                segment_text = tentative_segment_text
                new_mention_id = doc_id + '_' + str(len(new_document_mentions) + omitted_mentions % max_mention_per_new_doc)
                mention_start_index_in_new_doc = mention_to_add.start_index - prev_seq_len
                mention_end_index_in_new_doc = mention_to_add.start_index - prev_seq_len
                if mention_start_index_in_new_doc < mention_end_index_in_new_doc  and mention_start_index_in_new_doc>= 0 and mention_end_index_in_new_doc > 0:
                    new_document_mentions.append(Mention(new_mention_id,mention_start_index_in_new_doc,mention_end_index_in_new_doc,mention_to_add.concept_id))
                else:
                    omitted_mentions += 1
                mentions_added += 1
            else:
                segmented_doc = Document(doc_id,segment_text,new_document_mentions)
                remaining_segmented_docs,remaining_omitted_mentions = segment_recursive(remaining_text[len(segment_text):],remaining_mentions[mentions_added:],doc_id+1,prev_seq_len+len(segment_text))
                return [segmented_doc].extend(remaining_segmented_docs), omitted_mentions + remaining_omitted_mentions
        segmented_doc = Document(doc_id,remaining_text,new_document_mentions)
        return [segmented_doc],omitted_mentions#base case
    return segment_recursive(doc.message,doc.mentions)


def derive_domain_dataset(docs:list[Document],kd:Knowledge_Data) -> tuple(list[Document],Knowledge_Data):
    pass

def remove_overlaps(doc:Document,broad_strategy:bool=True) -> Document:
    if broad_strategy:
        old_mentions = doc.mentions
        new_docs = []
        for span_1 in old_mentions:
            contained=False
            for span_2 in old_mentions:
                if span_1.start_index >= span_2.start_index and span_1.end_index <= span_2.end_index and span_1 != span_2:
                    contained = True
            if not contained:
                new_docs.append(Document(doc.doc_id,doc.message,span_1))
        return new_docs
    else:
        raise NotImplementedError























