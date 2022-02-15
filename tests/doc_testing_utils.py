from dataclasses import asdict
from os import remove
from el_toolkit.document import *
from transformers import BertTokenizer
from itertools import product
import pytest
from collections import namedtuple
from el_toolkit.data_processors import segment_document,remove_overlaps


biomed_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='monologg/biobert_v1.1_pubmed',
                                            do_lower_case=False, cache_dir=None)

Doc_test = namedtuple("Doc_test",["test_name","expected_output_generator","with_overlaps"])
NO_OVERLAP_DOC_DATASET_NAMES = ["no_overlaps_3_mentions","no_overlaps_14_mentions","no_overlaps_32_mentions","no_overlaps_hard"]
OVERLAP_DOC_DATASET_NAMES = ["overlaps"]
DOC_TESTS = [Doc_test("segment_document",lambda doc: [asdict(segmented_doc) for segmented_doc in segment_document(doc,biomed_tokenizer,8)],False),
             Doc_test("remove_overlaps",lambda doc: asdict(remove_overlaps(doc)),True)]

def generate_doc_test_data():
    for dataset_name in NO_OVERLAP_DOC_DATASET_NAMES + OVERLAP_DOC_DATASET_NAMES:
        doc = Document.read_json(f"tests/test_data/doc_function_test_data/{dataset_name}/doc_data.json")
        all_test_data = {}
        for test in DOC_TESTS:
            if test.with_overlaps or dataset_name in NO_OVERLAP_DOC_DATASET_NAMES:
                test_data = {}
                test_data["expected_output"] = test.expected_output_generator(doc)
                all_test_data[test.test_name] = test_data
        with open(f"tests/test_data/doc_function_test_data/{dataset_name}/test_data.json","w+") as filename:
            json.dump(all_test_data,filename,indent = 2)


def get_doc_test_parameters():
    test_parameters = []
    ids = []
    for test,dataset_name in product(DOC_TESTS,NO_OVERLAP_DOC_DATASET_NAMES+OVERLAP_DOC_DATASET_NAMES):
        if test.with_overlaps or dataset_name in NO_OVERLAP_DOC_DATASET_NAMES:
            input_doc = Document.read_json(f"tests/test_data/doc_function_test_data/{dataset_name}/doc_data.json")
            with open(f"tests/test_data/doc_function_test_data/{dataset_name}/test_data.json") as file_name:
                expected_output = json.load(file_name)[test.test_name]["expected_output"]
            test_parameters.append((test,input_doc,expected_output))
            ids.append(f"{test.test_name}:{str(dataset_name)}")
    return test_parameters,ids
