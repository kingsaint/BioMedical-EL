from __future__ import annotations
from ipymarkup.demo import format_span_line_markup
from el_toolkit.lexical_knowledge_base import Lexical_Knowledge_Base
from el_toolkit.document import Document

class Displayer():
    def display(doc:Document,lkb:Lexical_Knowledge_Base=None):
        spans = [(mention.start_index,mention.end_index,mention.concept_id) for mention in doc.mentions]
        if lkb == None:
            spans = [[spans[0],spans[1],lkb.get_synonyms(spans[2])[0]] for span in spans]
        html = "".join(format_span_line_markup(doc.message, spans))
        return html