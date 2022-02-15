from __future__ import annotations
from ast import Return
from dataclasses import asdict, dataclass
import dataclasses
import json
from el_toolkit.lexical_knowledge_base import Lexical_Knowledge_Base
from dataclasses import dataclass

from ipymarkup import show_span_line_markup



@dataclass(frozen=True)
class Mention():
    """Class for keeping track of mentions"""
    start_index:int
    end_index:int
    concept_id: str
    
@dataclass(frozen=True)
class Document:
    doc_id: str
    message: str
    mentions: list[Mention]
    def write_json(self,file_path):
        with open(file_path, 'w') as outfile:
            dictionary = dataclasses.asdict(self)
            json.dump(dictionary,outfile,indent = 2)
    @classmethod
    def read_json(cls,file_path):
        with open(file_path, 'r') as infile:
            data = json.load(infile)
        return cls.from_dict(data)
    @classmethod
    def from_dict(cls,data):
        mentions = [Mention(**mention) for mention in data["mentions"]]
        return cls(data["doc_id"],data["message"],mentions)
    def check_for_span_overlaps(self):
        for mention_1 in self.mentions:
            for mention_2 in self.mentions:
                if mention_1.start_index >= mention_2.start_index and mention_1.end_index <= mention_2.end_index and mention_1 != mention_2:
                    return True
        return False
    def get_verbose_mentions(self):
        return [{**{"mention_text":self.message[mention.start_index:mention.end_index]},**asdict(mention)} for mention in self.mentions]