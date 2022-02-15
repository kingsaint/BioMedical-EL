from el_toolkit.lexical_knowledge_base import Knowledge_Data
from tests.doc_testing_utils import get_doc_test_parameters
from tests.lkb_testing_utils import get_lkb_test_parameters

import pytest

@pytest.fixture(scope="session")
def knowledge_datasets(request):
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
    knowledge_dataset =  knowledge_datasets(dataset_name)
    return lkb_type(knowledge_dataset)
    
#Tests on Document Functions
doc_test_parameters,doc_test_ids = get_doc_test_parameters()
@pytest.mark.parametrize("test,input_doc,expected_output", doc_test_parameters,ids=doc_test_ids)
def test_doc(test,input_doc,expected_output):
    assert test.expected_output_generator(input_doc) == expected_output


#Tests on Basic LKB Methods
lkb_test_parameters,lkb_test_ids = get_lkb_test_parameters()
@pytest.mark.parametrize("test,input_id,expected_output,lkb", lkb_test_parameters,ids=lkb_test_ids,indirect=["lkb"])
def test_lkb(test,input_id,expected_output,lkb):
    assert test.expected_output_generator(input_id,lkb) == expected_output

#Domain Deriving Test
