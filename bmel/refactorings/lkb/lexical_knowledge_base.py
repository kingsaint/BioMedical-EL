from __future__ import annotations
from argparse import ArgumentError
from dataclasses import dataclass
import json
from collections import defaultdict
from rdflib import Graph, URIRef, Literal, BNode, Namespace, term
from rdflib.namespace import RDF,XSD,RDFS
from rdflib.plugins import sparql
from click import Argument


@dataclass(frozen=True)
class Concept:
    id: str
    info: dict

@dataclass(frozen=True)
class Term:
    id: str
    string: str

@dataclass(frozen=True)
class Conceptual_Edge:
    concept_id_1: str
    concept_id_2: str
    rel_id: str

@dataclass(frozen=True)
class Lexical_Edge:
    concept_id: str
    term_id: str

@dataclass(frozen=True)
class Conceptual_Relation:
    id: str
    string: str #relation name

@dataclass(frozen=True)
class Knowledge_Data:#Intended to be a single data format for sparse storage. Used to initialize and LKB
    concepts: list[Concept]
    terms: list[Term]
    conceptual_edges: list[Conceptual_Edge]
    lexical_edges: list[Lexical_Edge]
    conceptual_relations: list[Conceptual_Relation]

class Lexical_Knowledge_Base:#Base Class; Intended to be the class that handles queries on the data and editing of knowledge data
    def __init__(self,data):
        raise NotImplementedError
    def get_concept(self,concept_id:str):
        raise NotImplementedError
    def get_term(self,term_id):
        raise NotImplementedError
    def get_forward_edges(self,concept_id:str):
        raise NotImplementedError
    def get_backward_edges(self,concept_id):
        raise NotImplementedError
    def get_terms_from_concept_id(self,concept_id:str):
        raise NotImplementedError
    def get_concepts_from_term_info(self,term_id:str):
        raise NotImplementedError
    def add_concept(self,concept):
        raise NotImplementedError
    def add_term(self,term):
        raise NotImplementedError
    def add_conceptual_relation(self,conceptual_relation):
        raise NotImplementedError
    def add_conceptual_edge(self,concept_id_1,concept_id_2,relation_id):
        raise NotImplementedError
    def add_lexical_edge(self,concept_id, term_id): 
        raise NotImplementedError

class Basic_Lexical_Knowledge_Base(Lexical_Knowledge_Base):
    def __init__(self,data):
        #not storage efficient
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
    def add_concept(self,concept):
        self.id_to_concept[concept.id] = concept 
    def add_term(self,term):
        self.id_to_concept[term.id] = term 
    def add_conceptual_relation(self,conceptual_relation):
        assert conceptual_relation.id not in self.id_to_concept_relations.keys()
        self.id_to_concept_relations[conceptual_relation.id] = conceptual_relation
    def add_conceptual_edge(self,concept_id_1,concept_id_2,relation_id = None, relation_name= None):
        if not relation_id:
            if not relation_name: 
                raise ArgumentError
            else:
                relation = self.concept_relation_name_to_concept_relation[relation_name]
        else:
            relation = self.id_to_concept_relations[relation_id]
        self.forward_conceptual_edges[concept_id_1].append(Conceptual_Edge(concept_id_1,concept_id_2,relation))
        self.backward_conceptual_edges[concept_id_2].append(Conceptual_Edge(concept_id_1,concept_id_2,relation))
    def add_lexical_edge(self,concept_id, term_id): 
        self.concept_id_to_term_ids[concept_id].append(term_id)
        self.term_ids_to_concept_ids[term_id].append(concept_id)

class RDF_Lexical_Knowledge_Base(Lexical_Knowledge_Base):
    def __init__(self,data):
        g = Graph()
        g.parse("lkb_vocab.ttl")
        VOCAB = Namespace('http://id.trendnet/vocab#')
        LKB = Namespace('http://id.trendnet/lkb/')
        g.bind('http://id.trendnet/vocab#', VOCAB)
        g.bind('http://id.trendnet/lkb/',LKB)
        for concept_attr_name,concept_attr in data.concepts[0].info.items():
            g.add((URIRef(f'http://id.trendnet/vocab#{concept_attr_name}'),RDF.type,VOCAB.Additional_Concept_Attribute))
            g.add((URIRef(f'http://id.trendnet/vocab#{concept_attr_name}'),RDFS.label,Literal(concept_attr_name)))
        for concept in data.concepts:
            concept_uri =  URIRef(f"http://id.trendnet/lkb/Concept/{concept.id}")
            g.add((concept_uri,RDF.type,VOCAB.Concept))
            g.add((concept_uri,VOCAB.id,Literal(concept.id, datatype=XSD.string)))
            for concept_attr_name,concept_attr in concept.info.items():
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
        q = """
            SELECT ?attr_label ?attr WHERE {?concept VOCAB:id ?id . 
                                            ?concept ?attr_name ?attr . 
                                            ?attr_name RDF:type VOCAB:Additional_Concept_Attribute .
                                            ?attr_name RDFS:label ?attr_label .
                                            ?concept RDF:type VOCAB:Concept 
                                            }
        """
        qres = self.sparql_query(q,initBindings={'id':Literal(concept_id, datatype=XSD.string)})
        return Concept(id = concept_id, info = {str(row.attr_label):term._castLexicalToPython(row.attr, row.attr.datatype) for row in qres})

    def get_term(self,term_id):
        q = """
            SELECT ?str WHERE {?term VOCAB:id ?id .
                              ?term VOCAB:string ?str .
                              ?term RDF:type VOCAB:Term
                              }
        """
        qres = self.sparql_query(q,initBindings={'id':Literal(term_id, datatype=XSD.string)})
        for row in qres:
            return Term(id = term_id, string = str(row.str))
    def get_relation(self,rel_id):
        q = """
            SELECT ?str WHERE {?relation VOCAB:id ?id .
                               ?relation VOCAB:string ?str .
                               ?relation RDF:type VOCAB:Concept_Relation
                               }
        """
        qres = self.sparql_query(q,initBindings={'id':Literal(rel_id, datatype=XSD.string)})
        for row in qres:
            return Conceptual_Relation(id = rel_id, string = str(row.str))
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
    def add_concept(self,concept):
        raise NotImplementedError
    def add_term(self,term):
        raise NotImplementedError
    def add_conceptual_relation(self,conceptual_relation):
        raise NotImplementedError
    def add_conceptual_edge(self,concept_id_1,concept_id_2,relation_id):
        raise NotImplementedError
    def add_lexical_edge(self,concept_id, term_id): 
        raise NotImplementedError
    def sparql_query(self,query:str,initBindings:dict):   
        q = sparql.prepareQuery(query,initNs = {"VOCAB": self.VOCAB, "RDF": RDF, "RDFS": RDFS})
        qres = self.g.query(q, initBindings=initBindings)
        return qres
    def get_g(self):
        return self.g