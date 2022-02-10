from el_toolkit.data import Knowledge_Data
from tests.test_lkb import generate_test_data
import json
knowledge_data = Knowledge_Data.read_json("tests/test_data/small_example")
print(knowledge_data)
test_result = generate_test_data(knowledge_data)
