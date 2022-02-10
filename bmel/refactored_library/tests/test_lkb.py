import random
from dataclasses import asdict
from el_toolkit.lexical_knowledge_base import Basic_Lexical_Knowledge_Base
from collections import defaultdict
from tests.fixtures import *
import functools
import pytest

DATASET_NAMES = ["small_example"]
LKB_TYPES = [Basic_Lexical_Knowledge_Base,RDF_Lexical_Knowledge_Base]

def generate_concept(lkb,concept_id):
    return asdict(lkb.get_concept(concept_id))
def generate_term(lkb,term_id):
    return asdict(lkb.get_term(term_id))
def generate_relation(lkb,conceptual_relation_id):
    return asdict(lkb.get_relation(conceptual_relation_id))
def generate_forward_edges(lkb,concept_id):
    return [(asdict(relation),asdict(concept)) for relation,concept in lkb.get_forward_edges(concept_id)]
def generate_backward_edges(lkb,concept_id):
    return [(asdict(relation),asdict(concept)) for relation,concept in lkb.get_backward_edges(concept_id)]
def generate_concepts_from_term_id(lkb,concept_id):
    return [asdict(term) for term in lkb.get_terms_from_concept_id(concept_id)]
def generate_terms_from_concept_id(lkb,term_id):
    return [asdict(concept) for concept in lkb.get_terms_from_concept_id(term_id)]

tests = {"concept":[(generate_concept,'concept'),
                    (generate_forward_edges,'forward_edges'),
                    (generate_backward_edges,'backward_edges'),
                    (generate_terms_from_concept_id,'terms_connected_to_concept')
                   ],
         "term":[(generate_term,'term'),
                 (generate_concepts_from_term_id,'concepts_connected_to_terms')],
                
         "conceptual_relation":[(generate_relation,'concept_relation')]}

def generate_test_data(knowledge_data):
    generated_output = defaultdict(None)
    lkb = Basic_Lexical_Knowledge_Base(knowledge_data)
    inputs = {"concept":random.choice(knowledge_data.concepts),
              "term":random.choice(knowledge_data.terms),
              "conceptual_relation":random.choice(knowledge_data.conceptual_relations)}
    def generate_output(test_input_type,knowledge_data_entity):
        if knowledge_data_entity:
            generated_output[f"tests_with_{test_input_type}_id_input"] = {f"input_{test_input_type}_id":knowledge_data_entity.id,"results":{}}
            for test in tests[test_input_type]:
                example_generator,output_field = test
                generated_output[f"tests_with_{test_input_type}_id_input"]["results"][output_field] = example_generator(lkb,knowledge_data_entity.id)
            return generated_output
        else:
            return {f"tests_with_{test_input_type}_id_input":{}}
    return functools.reduce(lambda a, b: dict(a,**b), (generate_output(input_type,knowledge_data_entity) for input_type,knowledge_data_entity in inputs.items()))

test_parameters = []
ids = []
for test_input_type,params in tests.items():
    for output_generator,output_field in params:
        for dataset_name in DATASET_NAMES:
            for lkb_type in LKB_TYPES:
                test_parameters.append((test_input_type,output_generator,output_field,dataset_name,(lkb_type,dataset_name)))
                ids.append(f"{output_generator.__name__}:{str(dataset_name)}:{str(lkb_type.__name__)}")

@pytest.mark.parametrize("test_input_type,output_generator,output_field,dataset_name,lkb", test_parameters,ids =ids,indirect=["lkb"])
def test_lkb(test_input_type,output_generator,output_field,dataset_name,lkb,knowledge_datasets,kb_expected_outputs):
    knowledge_data = knowledge_datasets(dataset_name)
    expected_output = kb_expected_outputs(dataset_name)
    assert output_generator(lkb,knowledge_data) == expected_output[f"tests_with_{test_input_type}_id_input"]["results"][output_field]
