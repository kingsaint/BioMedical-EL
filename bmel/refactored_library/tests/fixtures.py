from el_toolkit.lexical_knowledge_base import *
from el_toolkit.data import Knowledge_Data
from functools import lru_cache as cache
import json
import os

import pytest

DATASET_NAMES = ["small_example"]
LKB_TYPES = [Basic_Lexical_Knowledge_Base,RDF_Lexical_Knowledge_Base]

@cache
def get_small_doc_data():
    text = "This is a good example of a test sentence."
    doc_id = "123456"
    span_dicts = [{"start_index":15,"end_index":21,"concept_id":"C1"},
                  {"start_index":15,"end_index":40,"concept_id":"C2"},
                  {"start_index":0,"end_index":3,"concept_id":"C3"}
            ]
    return Document(text,doc_id,span_dicts)


@pytest.fixture(scope="session")
def kb_expected_outputs():
    expected_outputs = {}
    def _kb_expected_output(dataset_name):
        if dataset_name not in expected_outputs.keys():
            expected_outputs[dataset_name] = json.load(f"tests/test_data/lkb_test_data/{dataset_name}/knowledge_data.json)")
        return expected_outputs[dataset_name]
    yield _kb_expected_output
    expected_outputs = {}

@pytest.fixture(scope="session")
def knowledge_datasets():
    knowledge_datasets = {}
    def _knowledge_dataset(dataset_name):
        if dataset_name not in knowledge_datasets.keys():
            knowledge_datasets[dataset_name] = Knowledge_Data.read_json(f"tests/test_data/lkb_test_data/{dataset_name}/knowledge_data.json")
        return knowledge_datasets[dataset_name]
    yield _knowledge_dataset

@pytest.fixture(scope="session")
def lkb(request,knowledge_datasets):
    lkb_type = request.param[0]
    dataset_name = request.param[1]
    knowledge_dataset = knowledge_datasets(dataset_name)
    return lkb_type(knowledge_dataset)

