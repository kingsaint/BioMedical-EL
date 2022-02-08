from __future__ import annotations
from dataclasses import dataclass
import json
from collections import defaultdict
from lexical_knowledge_base import Lexical_Knowledge_Base


@dataclass(frozen=True)
class Mention():
    """Class for keeping track of spans"""
    mention_id:str
    start_index:int
    end_index:int
    concept_id: str

@dataclass(frozen=True)
class Document:
    message: str
    doc_id: str
    mentions: list[Mention]
    def __init__(self,doc_id,message,mentions):
        self.doc_id = doc_id
        self.message = message
        self.mentions = mentions
        self.mentions.sort(key=lambda mention:(mention.start_index,mention.end_index))
    def check_for_span_overlaps(self):
        for doc_spans in self.mentions:
            doc_spans
            for span_1 in doc_spans:
                for span_2 in doc_spans:
                    if span_1.start_index >= span_2.start_index and span_1.end_index <= span_2.end_index and span_1 != span_2:
                        return True
    

class EL_Dataset:
    def __init__(self,docs,lkb):
        self.docs = docs
        self.lkb = lkb
    def check_for_span_overlaps(self):
        return True in [doc.check_for_overlap() for doc in self.docs]
    def get_verbose_doc_dictionaries(self):
        #gets all docs with mentions including their respective concepts, useful for debugging
        doc_dicts = []
        for doc in self.docs:
            mention_dicts = []
            for mention in doc.mentions:
                sparse_mention_dict = dataclass.as_dict(mention)
                concept_dict = dataclass.as_dict(self.lkb.get_concept(mention.concept_id))
                mention_dicts.append(sparse_mention_dict|concept_dict)
            doc_dicts.append({"doc_id":doc.doc_id,"doc":doc.message,"mentions":doc.mentions_dic})
        return doc_dicts
    
class Displayer:
    def display(self,el_dataset):
        raise NotImplementedError

class No_Overlap_Displayer(Displayer):

    def display(self,el_dataset):
        assert not el_dataset.check_for_span_overlaps()
        raise NotImplementedError
        