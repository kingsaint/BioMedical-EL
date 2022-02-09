
from misc.hashdict import hashdict
from fixtures import *

def test_data_getting(lkb_with_initial_data):
    lkb,knowledge_data =  lkb_with_initial_data
    assert lkb.get_data() == knowledge_data
def test_concept_getting(lkb):
    assert lkb.get_concept("C1") == Concept(id = "C1",info={"tst_attr":3})
def test_term_getting(lkb):
    assert lkb.get_term("T1") == Term(id="T1",string="Test Term 1")
def test_relation_getting(lkb):
    assert lkb.get_relation("R1") == Conceptual_Relation(id="R1",string="Test Relation 1")
def test_forward_edge_getting(lkb):
    correct_result = sorted([(Conceptual_Relation(id="R1",string="Test Relation 1"),Concept(id = "C2",info=hashdict({"tst_attr":4}))),
                             (Conceptual_Relation(id="R2",string="Test Relation 2"),Concept(id = "C3",info=hashdict({"tst_attr":5})))
                             ],
                            key = (lambda tup: (tup[0].id,tup[1].id)))
    result = sorted(lkb.get_forward_edges("C1"),key = (lambda tup: (tup[0].id,tup[1].id)))
    assert result == correct_result

def test_backward_edge_getting(lkb):
    correct_result = sorted([(Conceptual_Relation(id="R1",string="Test Relation 1"),Concept(id = "C1",info={"tst_attr":3})),
                              (Conceptual_Relation(id="R1",string="Test Relation 1"),Concept(id = "C3",info={"tst_attr":5}))
                            ],
                            key = (lambda tup: (tup[0].id,tup[1].id)))
    result =sorted(lkb.get_backward_edges("C2"), key = (lambda tup: (tup[0].id,tup[1].id)))
    assert result == correct_result

def test_get_terms_for_concept(lkb):
    correct_result = {Term(id="T2",string="Test Term 2"),
                      Term(id="T3",string="Test Term 3")
                    }
    result = set(lkb.get_terms_from_concept_id("C3"))
    assert result == correct_result
def test_get_concepts_for_terms(lkb):
    correct_result = sorted([Concept(id = "C1",info={"tst_attr":3}),
                             Concept(id = "C2",info={"tst_attr":4})
                             ],
                            key = (lambda concept: concept.id))
    result = sorted(lkb.get_concepts_from_term_id("T1"),key = (lambda concept: concept.id))
    assert result == correct_result



        