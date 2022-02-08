test_with_pytest.py
from dataclasses import asdict
from datasets import *
import pytest
@pytest.fixture
def example():
    text = "This is a good example of a test sentence."
    span_dicts = [{"start_index":15,"end_index":21,"kb_entity_identifier":"C123456"},
                      {"start_index":15,"end_index":40,"kb_entity_identifier":"C11111"},
                      {"start_index":0,"end_index":3,"kb_entity_identifier":"C222222"}
                ]
    kb = {"C222222":["This"],
          "C11111":["Ex","Eg","Example"]
        }
    return {"text":text,"span_dicts":span_dicts,"kb":kb}
    

def test_linked_message_creation(example):
    correctly_produced_dict = {"text":"This is a good example of a test sentence.",
                                "spans": [{"start_index":0,"end_index":3,"kb_entity_identifier":"C222222"},
                                          {"start_index":15,"end_index":21,"kb_entity_identifier":"C123456"},
                                        {"start_index":15,"end_index":40,"kb_entity_identifier":"C11111"}
                                        ]
                                }
    
    spans = [Span(**dictionary) for dictionary in example["span_dicts"]]
    assert  asdict(Linked_Message(example["text"],spans)) == asdict(Linked_Message(example["text"],example["span_dicts"])) == correctly_produced_dict

def test_overlap_removal(example):
    correctly_produced_dict = {"text":"This is a good example of a test sentence.",
                            "spans": [{"start_index":0,"end_index":3,"kb_entity_identifier":"C222222"},
                                      {"start_index":15,"end_index":40,"kb_entity_identifier":"C11111"}
                                ]
                            }

    lm = Linked_Message(example["text"],example["span_dicts"])
    lm.remove_overlaps()
    assert asdict(lm) == correctly_produced_dict

def test_overlap_check(example):
    lm = Linked_Message(example["text"],example["span_dicts"])
    assert lm.check_for_overlap()
    lm.remove_overlaps()
    assert not lm.check_for_overlap()

def test_dataset_creation(example):
    correctly_produced_dict = {"linked_messages":[{"text":"This is a good example of a test sentence.",
                                                   "spans": [{"start_index":0,"end_index":3,"kb_entity_identifier":"C222222"},
                                                            {"start_index":15,"end_index":21,"kb_entity_identifier":"C123456"},
                                                            {"start_index":15,"end_index":40,"kb_entity_identifier":"C11111"}
                                                            ]
                                                  }
                                                ],
                                "kb":None
                                 }
    el_dataset = EL_Dataset([example["text"]],[example["span_dicts"]])
             
    assert asdict(el_dataset) == correctly_produced_dict

def test_no_overlap_dataset_creation(example):
    correctly_produced_dict = {"linked_messages":[{"text":"This is a good example of a test sentence.",
                                                   "spans": [{"start_index":0,"end_index":3,"kb_entity_identifier":"C222222"},
                                                            {"start_index":15,"end_index":40,"kb_entity_identifier":"C11111"}
                                                   ]
                                                  }
                                                ],
                                "kb":{'id_to_name_map': {'C11111': ['Ex', 'Eg', 'Example'], 'C222222': ['This']}}
                                }
    el_dataset = EL_Dataset([example["text"]],[example["span_dicts"]],kb=Knowledge_Base(example["kb"]),remove_overlaps=True)          
    assert asdict(el_dataset) == correctly_produced_dict
    
def test_get_verbose(example):
    correctly_produced_dict = {"linked_messages":[{"text":"This is a good example of a test sentence.",
                                                   "spans": [{"start_index":0,"end_index":3,"kb_entity_identifier":"C222222","span_text": "This","concept_names":["This"]},
                                                            {"start_index":15,"end_index":40,"kb_entity_identifier":"C11111","span_text": "example of a test sentence","concept_names":["Ex","Eg","Example"]}
                                                   ]
                                                  }
                                                ],
                                "kb":{'id_to_name_map': {'C11111': ['Ex', 'Eg', 'Example'], 'C222222': ['This']}}

                                }
    el_dataset = EL_Dataset([example["text"]],[example["span_dicts"]],kb=Knowledge_Base(example["kb"]),remove_overlaps=True)    

    assert EL_Dataset.get_verbose_dictionary(el_dataset) == correctly_produced_dict 