from dataclasses import asdict
from os import remove
from el_toolkit.document import *
from el_toolkit.lkb.lexical_knowledge_base import Lexical_Knowledge_Base
from el_toolkit.lkb.rdf_lkb import RDF_Lexical_Knowledge_Base
from el_toolkit.lkb.wordnet_lkb import WordNet_Lexical_Knowledge_Base
from el_toolkit.entity_linkers.dual_embedder.document_embedder import DocumentEmbedder
from transformers import BertTokenizer
import pytest
from collections import namedtuple
import random
import os

    

class DataDirectedTest:
    filepaths = []
    test_name = None
    def write_expected_output(self,input_filepath):
        print(input_filepath)
        test_data = {}
        test_data["input_filepath"] = input_filepath
        random_input = self.generate_random_input(input_filepath)
        test_data["random_inputs"] = random_input
        test_data["expected_output"] = self.generate_expected_output(input_filepath,random_input)
        with open(self.get_output_path(input_filepath),"w+") as filename:
            json.dump(test_data,filename,indent = 2)
    def get_output_path(self,input_filepath):
        test_directory = os.path.join("tests/test_data/test_outputs/",self.test_name)
        if not os.path.isdir(test_directory):
            os.mkdir(test_directory)
        return os.path.join(test_directory,input_filepath.split('/')[-1])
    def get_expected_output(self,input_filepath):
        with open(self.get_output_path(input_filepath)) as file_name:
            return dict(json.load(file_name))["expected_output"]
    @staticmethod
    def get_all_filepaths(parent_directory):
        filepaths = []
        for root, _, files in os.walk(parent_directory):
            for file in files:
                filepaths.append(os.path.join(root,file))
        return filepaths
    def get_random_input(self,input_filepath):
        with open(self.get_output_path(input_filepath)) as file_name:
            return dict(json.load(file_name))["random_inputs"]
    def generate_random_input(self,input_file):
        return None
    def get_inputs(self):
        raise NotImplementedError
    def generate_expected_output(self):
        raise NotImplementedError
    def write_all_expected_outputs(self):
        for filepath in self.filepaths:
            self.write_expected_output(filepath)



class LKBTest(DataDirectedTest):
    lkb_classes = [RDF_Lexical_Knowledge_Base,Lexical_Knowledge_Base,WordNet_Lexical_Knowledge_Base]
    filepaths = DataDirectedTest.get_all_filepaths("tests/test_data/test_knowledge_data/")
    def get_knowledge_data(self,input_filepath):
        with open(input_filepath, 'r') as infile:
            dictionary = json.load(infile)
        return dictionary
    def get_lkb(self,lkb_type,input_filepath):
        return lkb_type.read_json(input_filepath)
    def generate_expected_output(self):
        raise NotImplementedError
    def generate_random_concept_id(self,input_filepath):
        return random.choice([concept["id"] for concept in self.get_knowledge_data(input_filepath)["concepts"]])
    def generate_random_term_id(self,input_filepath):
        return random.choice([term["id"] for term in self.get_knowledge_data(input_filepath)["terms"]])
    def generate_random_cr_id(self,input_filepath):
        return random.choice([cr['id'] for cr in self.get_knowledge_data(input_filepath)["conceptual_relations"]])
    @pytest.mark.parametrize('lkb_type',lkb_classes,ids=[cls.__name__ for cls in lkb_classes])
    @pytest.mark.parametrize('input_filepath',filepaths,ids=[filepath.split("/")[-1] for filepath in filepaths])
    def test_execute(self,lkb_type,input_filepath):
        random_input = self.get_random_input(input_filepath)
        assert self.generate_expected_output(input_filepath,random_input,lkb_type=lkb_type) == self.get_expected_output(input_filepath)
    
        
class TestConceptGetting(LKBTest):
    test_name = "get_concept"
    generate_random_input = LKBTest.generate_random_concept_id
    def generate_expected_output(self,input_filepath,random_input,lkb_type=Lexical_Knowledge_Base):
        lkb = self.get_lkb(lkb_type,input_filepath)
        return asdict(lkb.get_concept(random_input))
    
class TestTermGetting(LKBTest):
    test_name = "get_term"
    generate_random_input = LKBTest.generate_random_term_id
    def generate_expected_output(self,input_filepath,random_input,lkb_type=Lexical_Knowledge_Base):
        lkb = self.get_lkb(lkb_type,input_filepath)
        return asdict(lkb.get_term(random_input))
    
class TestCRGetting(LKBTest):
    test_name = "get_conceptual_relation"
    generate_random_input = LKBTest.generate_random_cr_id
    def generate_expected_output(self,input_filepath,random_input,lkb_type=Lexical_Knowledge_Base):
        lkb = self.get_lkb(lkb_type,input_filepath)
        return asdict(lkb.get_relation(random_input))

class TestOutwardEdgeGetting(LKBTest):
    test_name = "get_outward_edges"
    generate_random_input = LKBTest.generate_random_concept_id
    def generate_expected_output(self,input_filepath,random_input,lkb_type=Lexical_Knowledge_Base):
        lkb = self.get_lkb(lkb_type,input_filepath)
        
        return sorted([[asdict(relation),asdict(concept)] for relation,concept in lkb.get_outward_edges(random_input)],key=(lambda lis: (lis[0]["id"],lis[1]["id"])))

class TestInwardEdgeGetting(LKBTest):
    test_name = "get_inward_edges"
    generate_random_input = LKBTest.generate_random_concept_id
    def generate_expected_output(self,input_filepath,random_input,lkb_type=Lexical_Knowledge_Base):
        lkb = self.get_lkb(lkb_type,input_filepath)
        return sorted([[asdict(relation),asdict(concept)] for relation,concept in lkb.get_inward_edges(random_input)],key=(lambda lis: (lis[0]["id"],lis[1]["id"])))

class TestSynonymGetting(LKBTest):
    test_name = "get_synonyms"
    generate_random_input = LKBTest.generate_random_concept_id
    def generate_expected_output(self,input_filepath,random_input,lkb_type=Lexical_Knowledge_Base):
        lkb = self.get_lkb(lkb_type,input_filepath)
        return sorted([asdict(term) for term in lkb.get_terms_from_concept_id(random_input)],key=(lambda term: term["id"]))

class TestPolysemeGetting(LKBTest):
    test_name = "get_polysemes"
    generate_random_input = LKBTest.generate_random_term_id
    def generate_expected_output(self,input_filepath,random_input,lkb_type=Lexical_Knowledge_Base):
        lkb = self.get_lkb(lkb_type,input_filepath)
        return sorted([asdict(concept) for concept in lkb.get_concepts_from_term_id(random_input)],key=(lambda term: term["id"]))

class DocumentDirectedTest(DataDirectedTest):
    def get_doc(self,input_filepath):
        return Document.read_json(input_filepath)
    
class TestSegment(DocumentDirectedTest):
    test_name = "segment"
    filepaths = DataDirectedTest.get_all_filepaths("tests/test_data/test_docs/without_overlaps/")
    def get_biomedical_tokenizer(self):
        return BertTokenizer.from_pretrained(pretrained_model_name_or_path='monologg/biobert_v1.1_pubmed',do_lower_case=False, cache_dir=None)
    def generate_expected_output(self,input_filepath,random_input=None):
        doc = self.get_doc(input_filepath)
        return [segmented_doc.get_data() for segmented_doc in doc.segment(self.get_biomedical_tokenizer(),8)]
    @pytest.mark.parametrize('input_filepath',filepaths,ids=[filepath.split("/")[-1] for filepath in filepaths])
    def test_execute(self,input_filepath):
        assert self.generate_expected_output(input_filepath) == self.get_expected_output(input_filepath)

class TestOverlapRemove(DocumentDirectedTest):
    test_name = "remove_overlaps"
    filepaths = DataDirectedTest.get_all_filepaths("tests/test_data/test_docs/")
    def generate_expected_output(self,input_filepath,random_input=None):
        doc = self.get_doc(input_filepath)
        return doc.remove_overlaps().get_data()
    @pytest.mark.parametrize('input_filepath',filepaths,ids=[filepath.split("/")[-1] for filepath in filepaths])
    def test_execute(self,input_filepath):
        assert self.generate_expected_output(input_filepath) == self.get_expected_output(input_filepath)

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='monologg/biobert_v1.1_pubmed',do_lower_case=False, cache_dir=None)
document_embedder = DocumentEmbedder(bert_model=None,tokenizer=tokenizer,max_seq_len=256,lower_case=False)
class TestDualEncoderDocEncode(DocumentDirectedTest):
    test_name = "dual_encoder_doc_encode"
    filepaths = DataDirectedTest.get_all_filepaths("tests/test_data/test_docs/")
    def generate_expected_output(self,input_filepath,random_input=None):
        doc = self.get_doc(input_filepath)
        doc_tokens = tokenizer.tokenize(doc.message)
        doc_token_ids,doc_tokens_mask,mention_start_markers,mention_end_markers,_,_ =document_embedder.encode_document(doc)
        return {"tokens":doc_tokens,"doc_token_ids":doc_token_ids,"doc_tokens_mask":doc_tokens_mask,"mention_start_markers":mention_start_markers,"mention_end_markers":mention_end_markers}
    @pytest.mark.parametrize('input_filepath',filepaths,ids=[filepath.split("/")[-1] for filepath in filepaths])
    def test_execute(self,input_filepath):
        assert self.generate_expected_output(input_filepath) == self.get_expected_output(input_filepath)

if __name__ == "__main__":
    #create tests
    for cls in [TestConceptGetting,
                TestTermGetting,
                TestCRGetting,
                TestOutwardEdgeGetting,
                TestInwardEdgeGetting,
                TestSynonymGetting,
                TestPolysemeGetting,
                TestSegment,
                TestOverlapRemove,
                TestDualEncoderDocEncode

                ]:
        cls().write_all_expected_outputs()



