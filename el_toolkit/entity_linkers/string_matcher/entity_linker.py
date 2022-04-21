from argparse import ArgumentError
from multiprocessing.sharedctypes import Value
from el_toolkit.documents.document import Document, Mention
from el_toolkit.entity_linkers.base_entity_linker import EntityLinker
from collections import defaultdict
from ahocorapy.keywordtree import KeywordTree
import re

class AhoCorasickEntityLinker(EntityLinker):
    def __init__(self,lkb=None,trie=None):
        # create the aho-corasick trie if trie is not defined.
        self._lkb = lkb
        if trie == None:
            if lkb == None:
                raise ArgumentError
            else:
                self._string_to_term_id,pattern_list = self.get_term_data_structures()
                self._trie = KeywordTree(case_insensitive=True)
                for pattern in pattern_list:
                    self._trie.add(pattern)
                self._trie.finalize()
        else:
            self._trie = trie
    def get_term_data_structures(self):
        pattern_list = []
        string_to_term_id = {}
        for term in self._lkb.get_data()[1]:
            processed_string = term.string.lower()
            if len(processed_string) > 1:
                string_to_term_id[processed_string] = term.id
                pattern_list.append(f"{processed_string}")
        return string_to_term_id,pattern_list
    def link(self,docs):
        def string_found(string1, string2):
            if re.search(r"\b" + re.escape(string1) + r"\b", string2):
                return True
            return False
        # add linked mentions to docs. 
        linked_docs = []
        for doc in docs: 
            results = self._trie.search_all(doc.message)
            mentions = []
            for result in results:
                word,i = result
                word = word.lower()
                start_index = i
                end_index = i+len(word)
                if string_found(word,doc.message.lower()):#string occurs at word boundaries
                    term_id = self._string_to_term_id[word]
                    concepts = self._lkb.get_concepts_from_term_id(term_id)
                    concept_id = concepts[0].id
                    mentions.append(Mention(start_index,end_index,concept_id))
            linked_docs.append(Document(doc.message,mentions))
        return linked_docs
    @classmethod
    def save(self,file_path):
        raise NotImplementedError
    @classmethod
    def save(self,file_path):
        raise NotImplementedError


