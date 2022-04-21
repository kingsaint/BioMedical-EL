import pytest
from el_toolkit.documents.document import Document
from el_toolkit.entity_linkers.string_matcher.entity_linker import AhoCorasickEntityLinker
from trendnet.trendnet import LKB_factory
trendnet = LKB_factory("TRENDNet")
el = AhoCorasickEntityLinker(lkb=trendnet)
def test_linking():
    document = el.link([Document("Does anyone else have a Prader-Willi Syndrome child that has been diagnosed with fabry disease.")])[0]
    print(document)
    assert False