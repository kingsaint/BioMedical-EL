from __future__ import annotations
from ipymarkup.demo import format_span_line_markup
from el_toolkit.lkb.lexical_knowledge_base import Lexical_Knowledge_Base
from el_toolkit.document import Document

class Displayer():
    def display(doc:Document,lkb:Lexical_Knowledge_Base=None):
        spans = doc.mention_data
        if lkb != None:
            spans = [(span.start_index,span.end_index,lkb.get_terms_from_concept_id(span.concept_id)[0].string) for span in spans]
        html = "".join(format_span_line_markup(doc.message, spans))
        return html