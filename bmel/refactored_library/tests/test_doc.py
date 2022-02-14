from dataclasses import asdict
from el_toolkit.document import *

# def test_document_creation(doc_dataset):
#     text,


# def test_overlap_removal(doc):

from collections import namedtuple
DOC_DATASET_NAMES = ["eg_1"]
Doc_test = namedtuple("Doc_test",["test_name","expected_output_generator"])

DOC_TESTS = [Doc_test("segment_document",lambda doc: asdict(segment_document(doc)))]

def generate_doc_test_data(dataset_name):
    doc = Document.read_json(f"tests/test_data/doc_test_data/{dataset_name}/doc_data.json")
    all_test_data = {}
    for test in DOC_TESTS:
        test_data = {}
        test_data["expected_output"] = test.expected_output_generator(doc)
        all_test_data[test.test_name] = test_data
    with open(f"tests/test_data/doc_test_data/{dataset_name}/test_data.json","w") as filename:
        json.dump(all_test_data,filename,indent = 2)

# test_parameters

# @pytest.mark.parametrize("test,input_id,expected_output,lkb", test_parameters,ids =ids,indirect=["lkb"])
# def test_doc(test,input_id,expected_output,lkb):
#     assert test.get_expected_output(input_id,lkb) == expected_output

    

# def test_linked_message_creation(example):
#     correctly_produced_dict = {"text":"This is a good example of a test sentence.",
#                                 "spans": [{"start_index":0,"end_index":3,"concept_id":"C222222"},
#                                           {"start_index":15,"end_index":21,"concept_id":"C123456"},
#                                         {"start_index":15,"end_index":40,"concept_id":"C11111"}
#                                         ]
#                                 }
    
#     spans = [Span(**dictionary) for dictionary in example["span_dicts"]]
#     assert  asdict(Linked_Message(example["text"],spans)) == asdict(Linked_Message(example["text"],example["span_dicts"])) == correctly_produced_dict

# def test_overlap_removal(example):
#     correctly_produced_dict = {"text":"This is a good example of a test sentence.",
#                             "spans": [{"start_index":0,"end_index":3,"concept_id":"C222222"},
#                                       {"start_index":15,"end_index":40,"concept_id":"C11111"}
#                                 ]
#                             }

#     lm = Linked_Message(example["text"],example["span_dicts"])
#     lm.remove_overlaps()
#     assert asdict(lm) == correctly_produced_dict

# def test_overlap_check(example):
#     lm = Linked_Message(example["text"],example["span_dicts"])
#     assert lm.check_for_overlap()
#     lm.remove_overlaps()
#     assert not lm.check_for_overlap()

# def test_dataset_creation(example):
#     correctly_produced_dict = {"linked_messages":[{"text":"This is a good example of a test sentence.",
#                                                    "spans": [{"start_index":0,"end_index":3,"concept_id":"C222222"},
#                                                             {"start_index":15,"end_index":21,"concept_id":"C123456"},
#                                                             {"start_index":15,"end_index":40,"concept_id":"C11111"}
#                                                             ]
#                                                   }
#                                                 ],
#                                 "kb":None
#                                  }
#     el_dataset = EL_Dataset([example["text"]],[example["span_dicts"]])
             
#     assert asdict(el_dataset) == correctly_produced_dict

# def test_no_overlap_dataset_creation(example):
#     correctly_produced_dict = {"linked_messages":[{"text":"This is a good example of a test sentence.",
#                                                    "spans": [{"start_index":0,"end_index":3,"concept_id":"C222222"},
#                                                             {"start_index":15,"end_index":40,"concept_id":"C11111"}
#                                                    ]
#                                                   }
#                                                 ],
#                                 "kb":{'id_to_name_map': {'C11111': ['Ex', 'Eg', 'Example'], 'C222222': ['This']}}
#                                 }
#     el_dataset = EL_Dataset([example["text"]],[example["span_dicts"]],kb=Knowledge_Base(example["kb"]),remove_overlaps=True)          
#     assert asdict(el_dataset) == correctly_produced_dict
    
# def test_get_verbose(example):
#     correctly_produced_dict = {"linked_messages":[{"text":"This is a good example of a test sentence.",
#                                                    "spans": [{"start_index":0,"end_index":3,"concept_id":"C222222","span_text": "This","concept_names":["This"]},
#                                                             {"start_index":15,"end_index":40,"concept_id":"C11111","span_text": "example of a test sentence","concept_names":["Ex","Eg","Example"]}
#                                                    ]
#                                                   }
#                                                 ],
#                                 "kb":{'id_to_name_map': {'C11111': ['Ex', 'Eg', 'Example'], 'C222222': ['This']}}

#                                 }
#     el_dataset = EL_Dataset([example["text"]],[example["span_dicts"]],kb=Knowledge_Base(example["kb"]),remove_overlaps=True)    

#     assert EL_Dataset.get_verbose_dictionary(el_dataset) == correctly_produced_dict 