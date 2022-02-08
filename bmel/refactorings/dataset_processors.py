from __future__ import annotations
from collections import defaultdict
from documents import *
from dataclasses import dataclass

class Dataset_Processor:
    def transform(self,el_dataset:EL_Dataset)->EL_Dataset:
        raise NotImplementedError

class Overlap_Remover(Dataset_Processor):
    def transform(self,el_dataset:EL_Dataset)->EL_Dataset:
        new_docs = [self.remove_overlaps(doc) for doc in el_dataset.docs]
        return EL_Dataset(new_docs,el_dataset.kb)
    def remove_overlaps(self,doc:Document)->Document:
        raise NotImplementedError

class Overlap_Remover_Narrow(Overlap_Remover):
    def remove_overlaps(self,doc):
        raise NotImplementedError

class Overlap_Remover_Broad(Overlap_Remover):
    def remove_overlaps(self,doc):
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
        
class Segmenter(Dataset_Processor):
    def __init__(self,tokenizer,max_mention_per_new_doc = None):
        self.tokenizer = tokenizer
        self.max_mention_per_new_doc = max_mention_per_new_doc
    def transform(self,el_dataset:EL_Dataset)->EL_Dataset:
        new_docs = [self.segment(doc) for doc in el_dataset.docs]
        return EL_Dataset(new_docs,el_dataset.kb)
    def segment(self,doc:Document) -> Document:
        raise NotImplementedError

class Recursive_Segmenter(Segmenter):
    def segment(self,doc:Document)-> Document:
        segmented_docs,omitted_mentions = self.segment_recursive(doc.message,doc.mentions)
        print(f"Omitted_mentions: {omitted_mentions}")
        return segmented_docs
    def segment_recursive(self,remaining_text,remaining_mentions,doc_id=0,prev_seq_len=0):
        omitted_mentions = 0
        new_document_mentions = []
        mentions_added = 0 
        segment_text = ""
        while mentions_added < len(remaining_mentions):
            mention_to_add =  remaining_mentions[mentions_added]
            tentative_segment_text = remaining_text[:mention_to_add.end_index]
            tokens = self.tokenizer.tokenize(tentative_segment_text)
            if (len(new_document_mentions) != self.max_mention_per_new_doc and len(['[CLS]'] + tokens + ['[SEP]']) < 256):
                segment_text = tentative_segment_text
                new_mention_id = doc_id + '_' + str(len(new_document_mentions) + omitted_mentions % self.max_mention_per_new_doc)
                mention_start_index_in_new_doc = mention_to_add.start_index - prev_seq_len
                mention_end_index_in_new_doc = mention_to_add.start_index - prev_seq_len
                if mention_start_index_in_new_doc < mention_end_index_in_new_doc  and mention_start_index_in_new_doc>= 0 and mention_end_index_in_new_doc > 0:
                    new_document_mentions.append(Mention(new_mention_id,mention_start_index_in_new_doc,mention_end_index_in_new_doc,mention_to_add.concept_id))
                else:
                    omitted_mentions += 1
                mentions_added += 1
            else:
                segmented_doc = Document(doc_id,segment_text,new_document_mentions)
                remaining_segmented_docs,remaining_omitted_mentions = self.segment(self,remaining_text[len(segment_text):],remaining_mentions[mentions_added:],doc_id+1,prev_seq_len+len(segment_text))
                return [segmented_doc].extend(remaining_segmented_docs), omitted_mentions + remaining_omitted_mentions
        segmented_doc = Document(doc_id,remaining_text,new_document_mentions)
        return [segmented_doc],omitted_mentions#base case
        















