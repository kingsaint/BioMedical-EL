import pytest
from lexical_knowledge_base import *
from hashdict import hashdict

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


small_example = Knowledge_Data(concepts=concepts,terms=terms,conceptual_relations=conceptual_relations,conceptual_edges=conceptual_edges,lexical_edges=lexical_edges)

basic_lkb = Basic_Lexical_Knowledge_Base(small_example)
rdf_lkb = RDF_Lexical_Knowledge_Base(small_example)
def get_basic_lkb():
    return basic_lkb
def get_rdf_lkb():
    return rdf_lkb
@pytest.fixture(params=[get_basic_lkb,get_rdf_lkb])
def LKB_Implementation(request):
    return request.param
def test_data_getting(LKB_Implementation):
    assert LKB_Implementation().get_data() == small_example
def test_concept_getting(LKB_Implementation):
    assert LKB_Implementation().get_concept("C1") == Concept(id = "C1",info={"tst_attr":3})
def test_term_getting(LKB_Implementation):
    assert LKB_Implementation().get_term("T1") == Term(id="T1",string="Test Term 1")
def test_relation_getting(LKB_Implementation):
    assert LKB_Implementation().get_relation("R1") == Conceptual_Relation(id="R1",string="Test Relation 1")
def test_forward_edge_getting(LKB_Implementation):
    correct_result = sorted([(Conceptual_Relation(id="R1",string="Test Relation 1"),Concept(id = "C2",info=hashdict({"tst_attr":4}))),
                             (Conceptual_Relation(id="R2",string="Test Relation 2"),Concept(id = "C3",info=hashdict({"tst_attr":5})))
                             ],
                            key = (lambda tup: (tup[0].id,tup[1].id)))
    result = sorted(LKB_Implementation().get_forward_edges("C1"),key = (lambda tup: (tup[0].id,tup[1].id)))
    assert result == correct_result

def test_backward_edge_getting(LKB_Implementation):
    correct_result = sorted([(Conceptual_Relation(id="R1",string="Test Relation 1"),Concept(id = "C1",info={"tst_attr":3})),
                              (Conceptual_Relation(id="R1",string="Test Relation 1"),Concept(id = "C3",info={"tst_attr":5}))
                            ],
                            key = (lambda tup: (tup[0].id,tup[1].id)))
    result =sorted(LKB_Implementation().get_backward_edges("C2"), key = (lambda tup: (tup[0].id,tup[1].id)))
    assert result == correct_result

def test_get_terms_for_concept(LKB_Implementation):
    correct_result = {Term(id="T2",string="Test Term 2"),
                      Term(id="T3",string="Test Term 3")
                    }
    result = set(LKB_Implementation().get_terms_from_concept_id("C3"))
    assert result == correct_result
def test_get_concepts_for_terms(LKB_Implementation):
    correct_result = sorted([Concept(id = "C1",info={"tst_attr":3}),
                             Concept(id = "C2",info={"tst_attr":4})
                             ],
                            key = (lambda concept: concept.id))
    result = sorted(LKB_Implementation().get_concepts_from_term_id("T1"),key = (lambda concept: concept.id))
    assert result == correct_result



        