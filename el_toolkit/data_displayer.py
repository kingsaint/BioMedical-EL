from __future__ import annotations

from lkb.basic_lkb import Lexical_Knowledge_Base
from IPython.display import display, HTML

class Displayable():
    def jupyter_display(self,*args):
        display(HTML(self.produce_html(*args)))
    def databricks_display(self):
        raise NotImplementedError
