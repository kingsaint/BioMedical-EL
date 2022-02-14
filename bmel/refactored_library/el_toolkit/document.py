from __future__ import annotations
from dataclasses import dataclass
import dataclasses
import json
from el_toolkit.lexical_knowledge_base import Knowledge_Data

@dataclass(frozen=True)
class Mention():
    """Class for keeping track of mentions"""
    mention_id: str
    start_index:int
    end_index:int
    concept_id: str

@dataclass(frozen=True)
class Document:
    message: str
    doc_id: str
    mentions: list[Mention]
    def __init__(self,doc_id,message,span_dicts):
        self.doc_id = doc_id
        self.message = message
        self.mentions = f"{doc_id}_{i}" **span_dicts
        self.mentions.sort(key=lambda mention:(mention.start_index,mention.end_index))
    def write_json(self,file_path):
        with open(file_path, 'w') as outfile:
            dictionary = dataclasses.asdict(self)
            json.dump(dictionary,outfile,indent = 2)
    @classmethod
    def read_json(cls,file_path):
        with open(file_path, 'r') as infile:
            dictionary = json.load(infile)
        return Document(**dictionary)
    def check_for_span_overlaps(self):
        for mention_1 in self.mentions:
            for mention_2 in self.mentions:
                if mention_1.start_index >= mention_2.start_index and mention_1.end_index <= mention_2.end_index and mention_1 != mention_2:
                    return True
        return False


class Displayer:
    def display(self):
        raise NotImplementedError

class No_Overlap_Displayer(Displayer):
    def display(self,documents,lkb=None):
        assert True not in [document.check_for_span_overlaps() for document in documents]
        raise NotImplementedError


def segment_document(doc:Document,tokenizer,max_mention_per_new_doc) -> list[Document]:
    def segment_recursive(remaining_text,remaining_mentions,doc_number=0,prev_seq_len=0):
        new_doc_id = doc.doc_id + "_" + doc_number
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
                new_mention_id = new_doc_id + str((len(new_document_mentions) + omitted_mentions) % max_mention_per_new_doc)
                mention_start_index_in_new_doc = mention_to_add.start_index - prev_seq_len
                mention_end_index_in_new_doc = mention_to_add.start_index - prev_seq_len
                if mention_start_index_in_new_doc < mention_end_index_in_new_doc  and mention_start_index_in_new_doc>= 0 and mention_end_index_in_new_doc > 0:
                    new_document_mentions.append(Mention(new_mention_id,mention_start_index_in_new_doc,mention_end_index_in_new_doc,mention_to_add.concept_id))
                else:
                    omitted_mentions += 1
                mentions_added += 1
            else:
                segmented_doc = Document(new_doc_id,segment_text,new_document_mentions)
                remaining_segmented_docs,remaining_omitted_mentions = segment_recursive(remaining_text[len(segment_text):],remaining_mentions[mentions_added:],doc_number+1,prev_seq_len+len(segment_text))
                return [segmented_doc].extend(remaining_segmented_docs), omitted_mentions + remaining_omitted_mentions
        segmented_doc = Document(new_doc_id,remaining_text,new_document_mentions)
        return [segmented_doc],omitted_mentions#base case
    return segment_recursive(doc.message,doc.mentions)

def remove_overlaps(doc,broad_strategy:bool=True) -> Document:
    if broad_strategy:
        no_overlap_mentions = []
        for mention_1 in doc.mentions:
            contained=False
            for mention_2 in doc.mentions:
                if mention_1.start_index >= mention_2.start_index and mention_1.end_index <= mention_2.end_index and mention_1 != mention_2:
                    contained = True
            if not contained:
                no_overlap_mentions.append(mention_1)
        return Document(doc.doc_id,doc.message,no_overlap_mentions)
    else:
        raise NotImplementedError































