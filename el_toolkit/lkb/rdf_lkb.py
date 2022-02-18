from __future__ import annotations
from el_toolkit.lkb.lexical_knowledge_base import Conceptual_Edge,Lexical_Edge, Lexical_Knowledge_Base
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF,XSD,RDFS
from rdflib.plugins import sparql
from dataclasses import asdict, dataclass, field


class RDF_Lexical_Knowledge_Base(Lexical_Knowledge_Base):#good for complex,possibly recursive queries.
    def __init__(self,concepts,terms,conceptual_relations,lexical_edges,conceptual_edges):
        g = Graph()
        self.id_to_concept = {concept.id:concept for concept in concepts}
        self.id_to_term= {term.id:term for term in terms}
        self.id_to_concept_relations = {relation.id:relation for relation in conceptual_relations}
        g.parse("el_toolkit/lkb/lkb_vocab.ttl")
        VOCAB = Namespace('http://id.trendnet/vocab#')
        LKB = Namespace('http://id.trendnet/lkb/')
        g.bind('http://id.trendnet/vocab#', VOCAB)
        g.bind('http://id.trendnet/lkb/',LKB)
        concept_attr_names = set()
        for concept in concepts:
            concept_uri =  URIRef(f"http://id.trendnet/lkb/Concept/{concept.id}")
            g.add((concept_uri,RDF.type,VOCAB.Concept))
            g.add((concept_uri,VOCAB.id,Literal(concept.id, datatype=XSD.string)))
            for concept_attr_name,concept_attr in concept.info.items():
                if concept_attr_name not in concept_attr_names:
                    concept_attr_names.add(concept_attr_name)
                    g.add((URIRef(f'http://id.trendnet/vocab#{concept_attr_name}'),RDF.type,VOCAB.Additional_Concept_Attribute))
                    g.add((URIRef(f'http://id.trendnet/vocab#{concept_attr_name}'),RDFS.label,Literal(concept_attr_name)))
                g.add((concept_uri,URIRef(f'http://id.trendnet/vocab#{concept_attr_name}'),Literal(concept_attr)))
        for term in terms:
            term_uri =  URIRef(f"http://id.trendnet/lkb/Term/{term.id}")
            g.add((term_uri,RDF.type,VOCAB.Term))
            g.add((term_uri,VOCAB.id,Literal(term.id,datatype=XSD.string)))
            g.add((term_uri,VOCAB.string,Literal(term.string,datatype=XSD.string)))
        for concept_relation in conceptual_relations:
            concept_relation_uri = URIRef(f"http://id.trendnet/lkb/Concept_Relation/{concept_relation.id}")
            g.add((concept_relation_uri,RDF.type,VOCAB.Concept_Relation))
            g.add((concept_relation_uri,VOCAB.id,Literal(concept_relation.id,datatype=XSD.string)))
            g.add((concept_relation_uri,VOCAB.string,Literal(concept_relation.string,datatype=XSD.string)))
        for edge in conceptual_edges:
            concept_one_uri = URIRef(f"http://id.trendnet/lkb/Concept/{edge.concept_id_1}")
            concept_two_uri = URIRef(f"http://id.trendnet/lkb/Concept/{edge.concept_id_2}")
            rel_uri = URIRef(f"http://id.trendnet/lkb/Concept_Relation/{edge.rel_id}")
            g.add((concept_one_uri,rel_uri,concept_two_uri))
        for edge in lexical_edges:
            concept_uri =  URIRef(f"http://id.trendnet/lkb/Concept/{edge.concept_id}")
            term_uri = URIRef(f"http://id.trendnet/lkb/Term/{edge.term_id}")
            g.add((concept_uri,VOCAB.vocab_term,term_uri))
        self.g = g
        self.VOCAB = VOCAB
    def get_concept(self,concept_id):
        return self.id_to_concept[concept_id]
    def get_term(self,term_id):
        return self.id_to_term[term_id]
    def get_relation(self,rel_id):
        return self.id_to_concept_relations[rel_id]
    def get_outward_edges(self,subject_concept_id:str):
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
    def get_inward_edges(self,object_concept_id):
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
        return concepts,terms,conceptual_relations,lexical_edges,conceptual_edges