from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF,XSD,RDFS
from rdflib.plugins import sparql
from misc.hashdict import hashdict
from data import *



class Basic_Lexical_Knowledge_Base:#Good for simple queries, 
    def __init__(self,data):
        self.id_to_concept = {concept.id:concept for concept in data.concepts}
        self.id_to_term= {term.id:term for term in data.terms}
        self.id_to_concept_relations = {relation.id:relation for relation in data.conceptual_relations}
        self.concept_relation_name_to_concept_relation = {relation.string:relation for relation in data.conceptual_relations}
        self.forward_conceptual_edges = defaultdict(list)
        self.backward_conceptual_edges = defaultdict(list)
        self.concept_id_to_term_ids = defaultdict(list)
        self.term_id_to_concept_ids = defaultdict(list)
        for edge in data.conceptual_edges:
            print("Creating conceptual relation index")
            self.forward_conceptual_edges[edge.concept_id_1].append((self.id_to_concept_relations[edge.rel_id],self.id_to_concept[edge.concept_id_2]))
            self.backward_conceptual_edges[edge.concept_id_2].append((self.id_to_concept_relations[edge.rel_id],self.id_to_concept[edge.concept_id_1]))
        for edge in data.lexical_edges:
            self.concept_id_to_term_ids[edge.concept_id].append(edge.term_id)
            self.term_id_to_concept_ids[edge.term_id].append(edge.concept_id)
    def get_concept(self,concept_id:str):
        return self.id_to_concept[concept_id]  
    def get_term(self,term_id):
        return self.id_to_term[term_id]  
    def get_relation(self,rel_id):
        return self.id_to_concept_relations[rel_id]
    def get_forward_edges(self,subject_concept_id:str):
        return self.forward_conceptual_edges[subject_concept_id]
    def get_backward_edges(self,object_concept_id:str):
        return self.backward_conceptual_edges[object_concept_id]    
    def get_terms_from_concept_id(self,concept_id:str):
        return [self.get_term(term_id) for term_id in self.concept_id_to_term_ids[concept_id]]
    def get_concepts_from_term_id(self,term_id:str):
        return [self.get_concept(term_id) for term_id in self.term_id_to_concept_ids[term_id]]
    def get_data(self):
        concepts =  set(self.id_to_concept.values())
        terms = set(self.id_to_term.values())
        conceptual_relations = set(self.id_to_concept_relations.values())
        conceptual_edges = set()
        for concept_id_1, related_concepts in self.forward_conceptual_edges.items():
            for rel,concept_2 in related_concepts:
                conceptual_edges.add(Conceptual_Edge(concept_id_1,concept_2.id,rel.id))
        lexical_edges = set()
        for concept_id,term_ids in self.concept_id_to_term_ids.items():
            for term_id in term_ids:
                lexical_edges.add(Lexical_Edge(concept_id,term_id))
        return Knowledge_Data(concepts,terms,conceptual_edges,lexical_edges,conceptual_relations)
class RDF_Lexical_Knowledge_Base:#good for complex,possibly recursive queries.
    def __init__(self,data):
        g = Graph()
        self.id_to_concept = {concept.id:concept for concept in data.concepts}
        self.id_to_term= {term.id:term for term in data.terms}
        self.id_to_concept_relations = {relation.id:relation for relation in data.conceptual_relations}
        g.parse("misc/lkb_vocab.ttl")
        VOCAB = Namespace('http://id.trendnet/vocab#')
        LKB = Namespace('http://id.trendnet/lkb/')
        g.bind('http://id.trendnet/vocab#', VOCAB)
        g.bind('http://id.trendnet/lkb/',LKB)
        concept_attr_names = set()
        for concept in data.concepts:
            concept_uri =  URIRef(f"http://id.trendnet/lkb/Concept/{concept.id}")
            g.add((concept_uri,RDF.type,VOCAB.Concept))
            g.add((concept_uri,VOCAB.id,Literal(concept.id, datatype=XSD.string)))
            for concept_attr_name,concept_attr in concept.info.items():
                if concept_attr_name not in concept_attr_names:
                    concept_attr_names.add(concept_attr_name)
                    g.add((URIRef(f'http://id.trendnet/vocab#{concept_attr_name}'),RDF.type,VOCAB.Additional_Concept_Attribute))
                    g.add((URIRef(f'http://id.trendnet/vocab#{concept_attr_name}'),RDFS.label,Literal(concept_attr_name)))
                g.add((concept_uri,URIRef(f'http://id.trendnet/vocab#{concept_attr_name}'),Literal(concept_attr)))
        for term in data.terms:
            term_uri =  URIRef(f"http://id.trendnet/lkb/Term/{term.id}")
            g.add((term_uri,RDF.type,VOCAB.Term))
            g.add((term_uri,VOCAB.id,Literal(term.id,datatype=XSD.string)))
            g.add((term_uri,VOCAB.string,Literal(term.string,datatype=XSD.string)))
        for concept_relation in data.conceptual_relations:
            concept_relation_uri = URIRef(f"http://id.trendnet/lkb/Concept_Relation/{concept_relation.id}")
            g.add((concept_relation_uri,RDF.type,VOCAB.Concept_Relation))
            g.add((concept_relation_uri,VOCAB.id,Literal(concept_relation.id,datatype=XSD.string)))
            g.add((concept_relation_uri,VOCAB.string,Literal(concept_relation.string,datatype=XSD.string)))
        for edge in data.conceptual_edges:
            concept_one_uri = URIRef(f"http://id.trendnet/lkb/Concept/{edge.concept_id_1}")
            concept_two_uri = URIRef(f"http://id.trendnet/lkb/Concept/{edge.concept_id_2}")
            rel_uri = URIRef(f"http://id.trendnet/lkb/Concept_Relation/{edge.rel_id}")
            g.add((concept_one_uri,rel_uri,concept_two_uri))
        for edge in data.lexical_edges:
            concept_uri =  URIRef(f"http://id.trendnet/lkb/Concept/{edge.concept_id}")
            term_uri = URIRef(f"http://id.trendnet/lkb/Term/{edge.term_id}")
            g.add((concept_uri,VOCAB.vocab_term,term_uri))
        self.g = g
        self.VOCAB = VOCAB
    def get_concept(self,concept_id:str):
        return self.id_to_concept[concept_id]
    def get_term(self,term_id):
        return self.id_to_term[term_id]
    def get_relation(self,rel_id):
        return self.id_to_concept_relations[rel_id]
    def get_forward_edges(self,subject_concept_id:str):
        q = """
            SELECT ?object_id ?rel_id WHERE {?subject_concept ?concept_relation ?object_concept .
                                            ?concept_relation RDF:type VOCAB:Concept_Relation .
                                            ?subject_concept VOCAB:id ?subject_id .
                                            ?object_concept VOCAB:id ?object_id .
                                            ?concept_relation VOCAB:id ?rel_id
                                     }
        """
        qres = self.sparql_query(q,initBindings={'subject_id':Literal(subject_concept_id, datatype=XSD.string)})
        return [(self.get_relation(str(row.rel_id)),self.get_concept(str(row.object_id))) for row in qres]
    def get_backward_edges(self,object_concept_id):
        q = """
            SELECT ?subject_id ?rel_id WHERE {?subject_concept ?concept_relation ?object_concept .
                                            ?concept_relation RDF:type VOCAB:Concept_Relation .
                                            ?subject_concept VOCAB:id ?subject_id .
                                            ?object_concept VOCAB:id ?object_id .
                                            ?concept_relation VOCAB:id ?rel_id
                                     }
        """
        qres = self.sparql_query(q,initBindings={'object_id':Literal(object_concept_id, datatype=XSD.string)})
        return [(self.get_relation(str(row.rel_id)),self.get_concept(str(row.subject_id))) for row in qres]
    def get_terms_from_concept_id(self,concept_id:str):
        q = """
            SELECT ?term_id WHERE {?concept VOCAB:vocab_term ?term .
                                   ?term VOCAB:id ?term_id .
                                   ?concept VOCAB:id ?concept_id
                              }
        """
        qres = self.sparql_query(q,initBindings={'concept_id':Literal(concept_id, datatype=XSD.string)})
        return [self.get_term(str(row.term_id)) for row in qres]
    def get_concepts_from_term_id(self,term_id:str):
        q = """
            SELECT ?concept_id WHERE {?concept VOCAB:vocab_term ?term .
                                      ?term VOCAB:id ?term_id .
                                      ?concept VOCAB:id ?concept_id
                              }
        """
        qres = self.sparql_query(q,initBindings={'term_id':Literal(term_id, datatype=XSD.string)})
        return [self.get_concept(str(row.concept_id)) for row in qres]
    def sparql_query(self,query:str,initBindings:dict=None):   
        q = sparql.prepareQuery(query,initNs = {"VOCAB": self.VOCAB, "RDF": RDF, "RDFS": RDFS})
        qres = self.g.query(q, initBindings=initBindings)
        return qres
    def get_g(self):
        return self.g
    def get_data(self):
        concepts =  set(self.id_to_concept.values())
        terms = set(self.id_to_term.values())
        conceptual_relations = set(self.id_to_concept_relations.values())
        q1 = """
            SELECT ?object_id ?rel_id ?subject_id WHERE {?subject_concept ?concept_relation ?object_concept .
                                                        ?concept_relation RDF:type VOCAB:Concept_Relation .
                                                        ?subject_concept VOCAB:id ?subject_id .
                                                        ?object_concept VOCAB:id ?object_id .
                                                        ?concept_relation VOCAB:id ?rel_id
                                     }
        """
        qres = self.sparql_query(q1)
        conceptual_edges = set(Conceptual_Edge(str(row.subject_id),str(row.object_id),str(row.rel_id),) for row in qres)
        q2 = """
            SELECT ?concept_id ?term_id WHERE {?concept VOCAB:vocab_term ?term .
                                               ?term VOCAB:id ?term_id .
                                               ?concept VOCAB:id ?concept_id
                                              }
        """
        qres = self.sparql_query(q2)
        lexical_edges = set(Lexical_Edge(str(row.concept_id),str(row.term_id)) for row in qres)
        return Knowledge_Data(concepts,terms,conceptual_edges,lexical_edges,conceptual_relations)




