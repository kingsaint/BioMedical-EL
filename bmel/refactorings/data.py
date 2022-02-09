from __future__ import annotations
from dataclasses import dataclass
import json
from collections import defaultdict
from lexical_knowledge_base import Lexical_Knowledge_Base
from misc.hashdict import hashdict

@dataclass(frozen=True)
class Mention():
    """Class for keeping track of mentions"""
    mention_id:str
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
        self.mentions =[Mention(mention_id = f"{doc_id}_{i}",**span_dict) for i,span_dict in enumerate(span_dicts)]
        self.mentions.sort(key=lambda mention:(mention.start_index,mention.end_index))
    def check_for_span_overlaps(self):
        for doc_spans in self.mentions:
            doc_spans
            for span_1 in doc_spans:
                for span_2 in doc_spans:
                    if span_1.start_index >= span_2.start_index and span_1.end_index <= span_2.end_index and span_1 != span_2:
                        return True
    def get_verbose_mentions(self,lkb):
        #gets mentions along with their concept info
        mention_dicts = []
        for mention in self.mentions:
            sparse_mention_dict = dataclass.as_dict(mention)
            concept_dict = dataclass.as_dict(lkb.get_concept(mention.concept_id))
            mention_dicts.append(sparse_mention_dict|concept_dict)
        return mention_dicts

class Displayer:
    def display(self,el_dataset):
        raise NotImplementedError

class No_Overlap_Displayer(Displayer):
    def display(self,documents,lkb=None):
        assert True not in [document.check_for_span_overlaps() for document in documents]
        raise NotImplementedError

@dataclass(frozen=True)
class Concept:
    id: str
    info: hashdict

@dataclass(frozen=True)
class Term:
    id: str
    string: str

@dataclass(frozen=True)
class Conceptual_Edge:
    concept_id_1: str
    concept_id_2: str
    rel_id: str

@dataclass(frozen=True)
class Lexical_Edge:
    concept_id: str
    term_id: str

@dataclass(frozen=True)
class Conceptual_Relation:
    id: str
    string: str #relation name

@dataclass(frozen=True)
class Knowledge_Data:#Intended to be a single data format for sparse storage. Used to initialize an LKB
    concepts: set[Concept]
    terms: set[Term]
    conceptual_edges: set[Conceptual_Edge]
    lexical_edges: set[Lexical_Edge]
    conceptual_relations: set[Conceptual_Relation]
        