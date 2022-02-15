
from ipymarkup.demo import format_span_line_markup
from el_toolkit.lexical_knowledge_base import Lexical_Knowledge_Base
from el_toolkit.document import Document
class Displayer():
    def display(doc:Document,lkb:Lexical_Knowledge_Base=None):
        spans = [(mention.start_index,mention.stop_index,mention.concept_id) for mention in doc.mentions]
        html_lines =format_span_line_markup(doc.message, spans)
        html = "".join(format_span_line_markup(text, spans))
        return html