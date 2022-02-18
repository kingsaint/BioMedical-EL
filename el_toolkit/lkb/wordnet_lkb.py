from collections import namedtuple
from el_toolkit.lkb.lexical_knowledge_base import Concept,  Lexical_Knowledge_Base, Term

ConceptEdge = namedtuple("ConceptEdge",["relation","concept"])

class Concept_Node:
    def __init__(self,concept:Concept):
        self.concept = concept
        self._term_nodes = set()
        self._outward_related_concept_nodes = []
        self._inward_related_concept_nodes = []
    def add_related_concept_node(self,relation,concept_node,direction:str):
        if direction == "outward":
            self._outward_related_concept_nodes.append(ConceptEdge(relation,concept_node))
        elif direction == "inward":
            self._inward_related_concept_nodes.append(ConceptEdge(relation,concept_node))
    def add_term_node(self,term):
        self._term_nodes.add(term)
    def get_term_nodes(self):
        return self._term_nodes
    def get_related_concepts(self,direction:str="outward"):
        if direction == "outward":
            return self._outward_related_concept_nodes
        elif direction == "inward":
            return self._inward_related_concept_nodes
    def __repr__(self):
        return f"{self.concept}"


class Term_Node:
    def __init__(self,term:Term):
        self.term = term
        self._concept_nodes = set()
    def add_concept_node(self,concept_node:Concept_Node):
        self._concept_nodes.add(concept_node)
    def get_concept_nodes(self):
        return self._concept_nodes
    def __repr__(self):
        return f"{self.term}"

class WordNet_Lexical_Knowledge_Base(Lexical_Knowledge_Base):#good for complex, possibly recursive queries.
    def __init__(self,concepts,terms,conceptual_relations,lexical_edges,conceptual_edges):
        self.id_to_concept_node = {concept.id:Concept_Node(concept) for concept in concepts}
        self.id_to_term_node= {term.id:Term_Node(term) for term in terms}
        self.id_to_concept_relation= {relation.id:relation for relation in conceptual_relations}
        for lexical_edge in lexical_edges:
            concept_node = self.get_concept_node(lexical_edge.concept_id)
            #print(f"{concept_node.concept},{concept_node.get_term_nodes()}")
            term_node = self.get_term_node(lexical_edge.term_id)
            concept_node.add_term_node(term_node)
            term_node.add_concept_node(concept_node)
        for conceptual_edge in conceptual_edges:
            src_concept_node = self.get_concept_node(conceptual_edge.concept_id_1)
            dst_concept_node = self.get_concept_node(conceptual_edge.concept_id_2)
            relation = self.id_to_concept_relation[conceptual_edge.rel_id]
            src_concept_node.add_related_concept_node(relation,dst_concept_node,direction="outward")
            dst_concept_node.add_related_concept_node(relation,src_concept_node,direction="inward")
    def get_concept(self,concept_id:str):
        return self.get_concept_node(concept_id).concept
    def get_term(self,term_id):
        return self.get_term_node(term_id).term
    def get_relation(self,rel_id):
        return self.id_to_concept_relation[rel_id]
    def get_outward_edges(self,subject_concept_id:str):
        return [(rel,object_concept_node.concept) for rel,object_concept_node in self.get_concept_node(subject_concept_id).get_related_concepts("outward")]
    def get_inward_edges(self,object_concept_id:str):
        return [(rel,subject_concept_node.concept)for rel,subject_concept_node in self.get_concept_node(object_concept_id).get_related_concepts("inward")]
    def get_terms_from_concept_id(self,concept_id:str):
        return [term_node.term for term_node in self.get_concept_node(concept_id).get_term_nodes()]
    def get_concepts_from_term_id(self,term_id:str):
        return[concept_node.concept for concept_node in  self.get_term_node(term_id).get_concept_nodes()]
    def get_data(self):
        raise NotImplementedError
    def get_concept_node(self,concept_id):
        return self.id_to_concept_node[concept_id]
    def get_term_node(self,term_id):
        return self.id_to_term_node[term_id]
    def derive_domain(self,ancestor_concept_id:str,isa_relation_label:str):
        ancestor_concept = self.get_concept_node(ancestor_concept_id)
        # Function to collect the nodes in a breadth-first traversal
        def get_descendants(concept_node):
            descendants = []
            visited_ids = set()
            queue = []
            queue.append(concept_node)
            visited_ids.add(concept_node.concept.id)
            while queue:
                concept_node = queue.pop(0)
                concept_id = concept_node.concept.id
                descendants.append(concept_node)
                for rel,concept_node in concept_node.get_related_concepts("inward"):
                    if concept_id not in visited_ids:
                        if rel.string==isa_relation_label:
                            queue.append(concept_node)
                            descendants.append(concept_node.concept)
                        visited_ids.add(concept_id)
            return descendants
        return get_descendants(ancestor_concept)
        