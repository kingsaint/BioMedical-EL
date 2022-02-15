from dataclasses import asdict
import random
from el_toolkit.lexical_knowledge_base import Basic_Lexical_Knowledge_Base, Knowledge_Data, RDF_Lexical_Knowledge_Base
from itertools import product
from tests.fixtures import *
import pytest 
from collections import namedtuple

LKB_DATASET_NAMES = ["small_example"]
KNOWLEDGE_BASE_CLASSES = [Basic_Lexical_Knowledge_Base,RDF_Lexical_Knowledge_Base]

LKB_test = namedtuple("LKB_test",["test_name","random_input_generator","expected_output_generator"])


def get_random_id(type):       
    def _get_random_id(knowledge_data):
        if type == "concept":
            set_of_items = [concept.id for concept in knowledge_data.concepts]
        elif type == "term":
            set_of_items = [term.id for term in knowledge_data.terms]
        elif type == "conceptual_relation":
            set_of_items = [cr.id for cr in knowledge_data.conceptual_relations]
        return random.choice(set_of_items)
    return _get_random_id

def handle_nulls(function):
    def _handle_nulls(id,lkb):
        if id == None:
            return None
        else:

            return function(id,lkb)
    return _handle_nulls

LKB_TESTS = [LKB_test("get_concept",get_random_id("concept"),handle_nulls((lambda id,lkb: asdict(lkb.get_concept(id))))),
             LKB_test("get_term",get_random_id("term"),handle_nulls((lambda id,lkb: asdict(lkb.get_term(id))))),
             LKB_test("get_relation",get_random_id("conceptual_relation"),handle_nulls((lambda id,lkb: asdict(lkb.get_relation(id))))),
             LKB_test("get_forward_edges",get_random_id("concept"),handle_nulls((lambda id,lkb: [[asdict(relation),asdict(concept)] for relation,concept in lkb.get_forward_edges(id)]))),
             LKB_test("get_backward_edges",get_random_id("concept"),handle_nulls((lambda id,lkb: [[asdict(relation),asdict(concept)] for relation,concept in lkb.get_backward_edges(id)]))),
             LKB_test("get_terms_from_concept_id",get_random_id("concept"),handle_nulls((lambda id,lkb: [asdict(term) for term in lkb.get_terms_from_concept_id(id)]))),
             LKB_test("get_concepts_from_term_id",get_random_id("term"),handle_nulls((lambda id,lkb: [asdict(concept) for concept in lkb.get_concepts_from_term_id(id)])))
]

def generate_lkb_test_data():
    for dataset_name in LKB_DATASET_NAMES:
        all_test_data = {}
        knowledge_data = Knowledge_Data.read_json(f"tests/test_data/lkb_test_data/{dataset_name}/knowledge_data.json")
        lkb = Basic_Lexical_Knowledge_Base(knowledge_data)
        for test in LKB_TESTS:
            test_data = {}
            test_data["input_id"] = test.random_input_generator(knowledge_data)
            test_data["expected_output"] = test.expected_output_generator(test_data["input_id"],lkb)
            all_test_data[test.test_name] = test_data
        with open(f"tests/test_data/lkb_test_data/{dataset_name}/test_data.json","w+") as filename:
            json.dump(all_test_data,filename,indent = 2)


def get_lkb_test_parameters():
    test_parameters = []
    ids = []
    for test,dataset_name,lkb_type in product(LKB_TESTS,LKB_DATASET_NAMES,KNOWLEDGE_BASE_CLASSES):
        with open(f"tests/test_data/lkb_test_data/{dataset_name}/test_data.json") as file_name:
            test_info = json.load(file_name)[test.test_name]
        input_id = test_info["input_id"]
        expected_output = test_info["expected_output"]
        test_parameters.append((test,input_id,expected_output,(lkb_type,dataset_name)))
        ids.append(f"{test.test_name}:{str(dataset_name)}:{str(lkb_type.__name__)}")
    return test_parameters,ids



