from tests.doc_testing_utils import get_doc_test_parameters
from tests.lkb_testing_utils import get_lkb_test_parameters
from tests.fixtures import *

import pytest


##Doc Tests
doc_test_parameters,doc_test_ids = get_doc_test_parameters()
@pytest.mark.parametrize("test,input_doc,expected_output", doc_test_parameters,ids=doc_test_ids)
def test_doc(test,input_doc,expected_output):
    assert test.expected_output_generator(input_doc) == expected_output


#LKB Tests
lkb_test_parameters,lkb_test_ids = get_lkb_test_parameters()
@pytest.mark.parametrize("test,input_id,expected_output,lkb", lkb_test_parameters,ids=lkb_test_ids,indirect=["lkb"])
def test_lkb(test,input_id,expected_output,lkb):
    assert test.expected_output_generator(input_id,lkb) == expected_output

