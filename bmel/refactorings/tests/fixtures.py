from lexical_knowledge_base import *
from data import *
from functools import lru_cache as cache

import pytest

@cache
def get_small_knowledge_data():
    concepts = {Concept(id = "C1",info=hashdict({"tst_attr":3})),
                Concept(id = "C2",info=hashdict({"tst_attr":4})),
                Concept(id = "C3",info=hashdict({"tst_attr":5})),
    }
    terms = {Term(id="T1",string="Test Term 1"),
            Term(id="T2",string="Test Term 2"),
            Term(id="T3",string="Test Term 3")
    }
    conceptual_relations = {Conceptual_Relation(id="R1",string="Test Relation 1"),
                            Conceptual_Relation(id="R2",string="Test Relation 2")
    }

    conceptual_edges = {Conceptual_Edge(concept_id_1= "C1",concept_id_2= "C2",rel_id="R1"),
                        Conceptual_Edge(concept_id_1= "C1",concept_id_2= "C3",rel_id="R2"),
                        Conceptual_Edge(concept_id_1= "C3",concept_id_2= "C2",rel_id="R1")
    }
    lexical_edges = {Lexical_Edge(concept_id="C1",term_id="T1"),
                    Lexical_Edge(concept_id="C2",term_id="T1"),
                    Lexical_Edge(concept_id="C3",term_id="T2"),
                    Lexical_Edge(concept_id="C3",term_id="T3"),
    }


    small_example_knowledge_data = Knowledge_Data(concepts=concepts,terms=terms,conceptual_relations=conceptual_relations,conceptual_edges=conceptual_edges,lexical_edges=lexical_edges)

    return small_example_knowledge_data

@cache
def get_small_doc_data():
    text = "This is a good example of a test sentence."
    doc_id = "123456"
    span_dicts = [{"start_index":15,"end_index":21,"concept_id":"C1"},
                {"start_index":15,"end_index":40,"concept_id":"C2"},
                {"start_index":0,"end_index":3,"concept_id":"C3"}
            ]
    return Document(text,doc_id,span_dicts)


doc_datasets = [get_small_doc_data]
ids = ["small_doc_dataset"]
@pytest.fixture(params=doc_datasets,ids = ids,scope="session")
def doc_dataset(request):
    return request.param()


lkb_types = [Basic_Lexical_Knowledge_Base,RDF_Lexical_Knowledge_Base]
@pytest.fixture(params=lkb_types,scope="session")
def lkb_type(request):
    return request.param

knowledge_data_egs = [get_small_knowledge_data]
ids = ["small_example_data"]
@pytest.fixture(params=knowledge_data_egs,ids=ids,scope="session")
def knowledge_data(request):
    return request.param()

@pytest.fixture(scope="session")
def lkb_with_initial_data(lkb_type,knowledge_data):
    return lkb_type(knowledge_data),knowledge_data

@pytest.fixture(scope="session")
def lkb(lkb_with_initial_data):
    return lkb_with_initial_data[0]

