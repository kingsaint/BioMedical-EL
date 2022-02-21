from __future__ import annotations
from collections import namedtuple
from dataclasses import dataclass
import json

from ipymarkup import show_span_line_markup

class Mention:#Only let document initialize this class
    def __init__(self,start_index,end_index,concept_id):
        self._start_index = start_index
        self._end_index = end_index
        self._concept_id = concept_id
    def set_doc(self,doc):
        self._doc = doc
    @property
    def mention_data(self):
        return {"start_index":self._start_index,"end_index":self._end_index,"concept_id":self._concept_id}
    @property
    def text(self):
        return self._doc.message[self._start_index:self._end_index]
    @property
    def start_index(self):
        return self._start_index
    @property
    def start_index(self):
        return self._start_index
    @property
    def end_index(self):
        return self._end_index
    @property
    def concept_id(self):
        return self._concept_id
    @property
    def doc(self):
        return self._doc
    def __repr__(self):
        return {"start_index":self.start_index,"end_index":self.end_index,"concept_id":self.concept_id,"text":self.text}

class Document:
    def __init__(self,doc_id,message,mentions):
        self.doc_id = doc_id
        self.message = message
        for mention in mentions:
            mention.set_doc(self)
        self._mentions = mentions
    @property
    def mentions(self):
        return self._mentions
    @classmethod
    def read_json(cls,file_path):
        with open(file_path, 'r') as infile:
            data = json.load(infile)
        mentions = [Mention(**mention) for mention in data["mentions"]]
        return cls(data["doc_id"],data["message"],mentions)
    def write_json(self,file_path):
        data = self.get_data()
        with open(file_path, 'w+') as infile:
            json.dump(data,infile)
    def get_data(self):
        return {"doc_ids":self.doc_id,"message":self.message,"mentions":[mention.mention_data for mention in self._mentions]}
    def check_for_span_overlaps(self):
        for mention_1 in self.mentions:
            for mention_2 in self.mentions:
                if mention_1.start_index >= mention_2.start_index and mention_1.end_index <= mention_2.end_index and mention_1 != mention_2:
                    return True
        return False
    def __repr__(self):
        return f"doc_id:{self.doc_id},message:{self.message},mentions:{self._mentions}"
    def segment(self:Document,tokenizer,max_mention_per_new_doc) -> list(Document):
        assert not self.check_for_span_overlaps()
        def segment_recursive(remaining_text,remaining_mentions,doc_number=0,prev_seq_len=0):
            new_doc_id = self.doc_id + "_" + str(doc_number)
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
            segmented_doc = Document(new_doc_id,remaining_text,new_document_mentions)
            return [segmented_doc],omitted_mentions#base case
        segmented_docs,omitted_mentions = segment_recursive(self.message,self.mentions)
        print(f"Mentions Omitted: {omitted_mentions}")
        return segmented_docs
    def remove_overlaps(self:Document,broad_strategy:bool=True) -> Document:
        if broad_strategy:
            no_overlap_mention_data = []
            for mention_1 in self.mentions:
                contained=False
                for mention_2 in self.mentions:
                    if mention_1.start_index >= mention_2.start_index and mention_1.end_index <= mention_2.end_index and mention_1 != mention_2:
                        contained = True
                if not contained:
                    no_overlap_mention_data.append(mention_1)
            return Document(self.doc_id,self.message,self.mentions)
        else:
            raise NotImplementedError
    



