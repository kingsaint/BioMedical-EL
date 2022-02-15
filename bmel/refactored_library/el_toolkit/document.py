from __future__ import annotations
from ast import Return
from dataclasses import asdict, dataclass
import dataclasses
import json
from el_toolkit.lexical_knowledge_base import Knowledge_Data
from dataclasses import dataclass



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
        return Document(data["doc_id"],data["message"],mentions)
    def check_for_span_overlaps(self):
        for mention_1 in self.mentions:
            for mention_2 in self.mentions:
                if mention_1.start_index >= mention_2.start_index and mention_1.end_index <= mention_2.end_index and mention_1 != mention_2:
                    return True
        return False
    def get_verbose_mentions(self):
        return [{**{"mention_text":self.message[mention.start_index:mention.end_index]},**asdict(mention)} for mention in self.mentions]


class Displayer:
    def display(self):
        raise NotImplementedError

class No_Overlap_Displayer(Displayer):
    def display(self,documents,lkb=None):
        assert True not in [document.check_for_span_overlaps() for document in documents]
        raise NotImplementedError


def segment_document(doc:Document,tokenizer,max_mention_per_new_doc) -> list[Document]:
    def segment_recursive(remaining_text,remaining_mentions,doc_number=0,prev_seq_len=0):
        new_doc_id = doc.doc_id + "_" + str(doc_number)
        omitted_mentions = 0
        new_document_mentions = []
        mentions_added = 0 
        segment_text = ""
        while mentions_added < len(remaining_mentions):#we've processed all remaining mentions
            mention_to_add =  remaining_mentions[mentions_added]
            mention_start_index_in_new_doc = mention_to_add.start_index - prev_seq_len
            mention_end_index_in_new_doc = mention_to_add.end_index - prev_seq_len
            tentative_segment_text = remaining_text[:mention_end_index_in_new_doc]
            tokens = tokenizer.tokenize(tentative_segment_text)
            if (len(new_document_mentions) < max_mention_per_new_doc and len(['[CLS]'] + tokens + ['[SEP]']) < 256):
                segment_text = tentative_segment_text
                #new_mention_id = new_doc_id + str((len(new_document_mentions) + omitted_mentions) % max_mention_per_new_doc)
                if mention_start_index_in_new_doc < mention_end_index_in_new_doc  and mention_start_index_in_new_doc>= 0 and mention_end_index_in_new_doc > 0:
                    new_document_mentions.append(Mention(mention_start_index_in_new_doc,mention_end_index_in_new_doc,mention_to_add.concept_id))
                else:
                    omitted_mentions += 1
                mentions_added += 1
            else:
                segmented_doc = Document(new_doc_id,segment_text,new_document_mentions)
                remaining_segmented_docs,remaining_omitted_mentions = segment_recursive(remaining_text[len(segment_text):],remaining_mentions[mentions_added:],doc_number+1,prev_seq_len+len(segment_text))
                return [segmented_doc] + remaining_segmented_docs, omitted_mentions + remaining_omitted_mentions
        segmented_doc = Document(new_doc_id,segment_text,new_document_mentions)
        return [segmented_doc],omitted_mentions#base case
    segmented_docs,omitted_mentions = segment_recursive(doc.message,doc.mentions)
    print(f"Mentions Omitted: {omitted_mentions}")
    return segmented_docs

def remove_overlaps(doc:Document,broad_strategy:bool=True) -> Document:
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































