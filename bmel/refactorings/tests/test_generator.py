import random
from refactorings.lexical_knowledge_base import Basic_Lexical_Knowledge_Base
from dataclasses import asdict
from collections import defaultdict
import pytest

def generate_concept(lkb,concept_id):
    return asdict(lkb.get_concept(concept_id))
def generate_term(lkb,term_id):
    return asdict(lkb.get_term(term_id))
def generate_relation(lkb,term_id):
    return asdict(lkb.get_term(term_id))
def generate_forward_edges(lkb,concept_id):
    return {asdict(forward_edge) for forward_edge in lkb.get_forward_edges(concept_id)}
def generate_backward_edges(lkb,concept_id):
    return {asdict(backward_edge) for backward_edge in lkb.get_forward_edges(concept_id)}
def generate_concepts_from_term_id(lkb,concept_id):
    return {asdict(term) for term in lkb.get_terms_from_concept_id(concept_id)}
def generate_terms_from_concept_id(lkb,term_id):
    return {asdict(concept) for concept in lkb.get_terms_from_concept_id(term_id)}

def generate_test_data(knowledge_data):
    generated_output = defaultdict(None)
    random_concept = random.choice(knowledge_data.concepts)
    generated_output["input_concept_id"] = random_concept.id
    random_term = random.choice(knowledge_data.terms)
    if random_term:
        generated_output["input_term_id"] = random_concept.id
    random_conceptual_relation = random.choice(knowledge_data.terms)
    if random_conceptual_relation:
        generated_output["input_conceptual_relation_id"] = random_concept.id
    lkb = Basic_Lexical_Knowledge_Base(knowledge_data)
    for test in tests:
        function = test[0]
        input_key = test[1]
        output_key = test[2]
        if input_key == "input_term_id" and random_term:
                generated_output[input_key] = function(lkb,random_term.id)
        elif input_key == "input_conceptual_relation_id" and random_conceptual_relation:
                generated_output[input_key] = function(lkb,random_conceptual_relation.id)
        elif input_key == "input_concept_id":
            generated_output[input_key] = function(lkb,random_concept.id)

tests = [(generate_concept,'input_concept_id','concept'),
         (generate_term,'input_term_id','term'),
         (generate_relation,'input_concept_relation_id','concept_relation'),
         (generate_forward_edges,'input_concept_id','concept'),
         (generate_backward_edges,'input_concept_id','backward_edges'),
         (generate_concepts_from_term_id,'input_concept_id','"concepts_connected_to_terms"'),
         (generate_terms_from_concept_id,'input_term_id','terms_connected_to_concept')
]

@pytest.mark.parametrize("output_generator,test_input_field,test_output_field", tests)
def lkb_test(output_generator,test_input_field,test_output_field,lkb,test_data_output):
    assert output_generator(lkb,test_data_output[test_input_field]) == test_data_output[test_output_field]
    


