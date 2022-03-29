from __future__ import annotations
from abc import abstractmethod
from collections import defaultdict, namedtuple
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF,XSD,RDFS
from rdflib.plugins import sparql
import json
from dataclasses import asdict, dataclass, field
import json
@dataclass(frozen=True)
class Concept:
    id: str
    #concept_name: str
    # info: dict = field(default_factory=dict)

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

class Lexical_Knowledge_Base_BC:
    @abstractmethod
    def __init__(self,concepts,terms,conceptual_edges,lexical_edges,conceptual_relations):
        pass
    @abstractmethod
    def get_concept(self,concept_id:str):
        pass 
    @abstractmethod
    def get_term(self,term_id):
        pass
    @abstractmethod
    def get_relation(self,rel_id):
        pass
    @abstractmethod
    def get_outward_edges(self,subject_concept_id:str):
        pass
    @abstractmethod
    def get_inward_edges(self,object_concept_id:str):
        pass
    @abstractmethod
    def get_terms_from_concept_id(self,concept_id:str):
        pass
    @abstractmethod
    def get_concepts_from_term_id(self,term_id:str):
        pass
    @abstractmethod
    def get_data(self):
        pass
    def get_named_concepts(self):
        #return a list of concepts, and a string corresponding to their canonical term.
        raise NotImplementedError
    @classmethod
    def read_json(cls,file_path):
        with open(file_path, 'r') as infile:
            dictionary = json.load(infile)
            concepts = [Concept(concept["id"]) for concept in dictionary["concepts"]]
            terms = [Term(**term) for term in dictionary["terms"]]
            conceptual_edges = [Conceptual_Edge(**conceptual_edge) for conceptual_edge in dictionary["conceptual_edges"]]
            lexical_edges = [Lexical_Edge(**lexical_edge) for lexical_edge in dictionary["lexical_edges"]]
            conceptual_relations = [Conceptual_Relation(**conceptual_relation) for conceptual_relation in dictionary["conceptual_relations"]]
            return cls(concepts,terms,conceptual_relations,lexical_edges,conceptual_edges)
    def write_json(self,file_path):
        concepts,terms,conceptual_edges,lexical_edges,conceptual_relations = self.get_data()
        dictionary = {"concepts":list(concepts),"terms":list(terms),"conceptual_edges":list(conceptual_edges),"lexical_edges":list(lexical_edges),"conceptual_relations":list(conceptual_relations)}
        jsonified = {key:[asdict(entity) for entity in value] for key,value in dictionary.items()}
        with open(file_path, 'w') as outfile:
            json.dump(jsonified,outfile,indent = 2)
    
        
class Lexical_Knowledge_Base(Lexical_Knowledge_Base_BC):#Good for simple queries, not memory-efficient
    def __init__(self,concepts,terms,conceptual_relations,lexical_edges,conceptual_edges):
        self.id_to_concept = {concept.id:concept for concept in concepts}
        self.id_to_term= {term.id:term for term in terms}
        self.id_to_concept_relations = {relation.id:relation for relation in conceptual_relations}
        self.concept_relation_name_to_concept_relation = {relation.string:relation for relation in conceptual_relations}
        self.outward_conceptual_edges = defaultdict(list)
        self.inward_conceptual_edges = defaultdict(list)
        self.concept_id_to_term_ids = defaultdict(list)
        self.term_id_to_concept_ids = defaultdict(list)
        print("Creating conceptual edge index")
        for edge in conceptual_edges:
            self.outward_conceptual_edges[edge.concept_id_1].append((self.id_to_concept_relations[edge.rel_id],self.id_to_concept[edge.concept_id_2]))
            self.inward_conceptual_edges[edge.concept_id_2].append((self.id_to_concept_relations[edge.rel_id],self.id_to_concept[edge.concept_id_1]))
        print("Creating lexical edge index")
        for edge in lexical_edges:
            self.concept_id_to_term_ids[edge.concept_id].append(edge.term_id)
            self.term_id_to_concept_ids[edge.term_id].append(edge.concept_id)
    # def contains_concept(self,concept_id):
    #     return concept_id in self.id_to_concept
    def get_concept(self,concept_id:str):
        return self.id_to_concept[concept_id]  
    def get_term(self,term_id):
        return self.id_to_term[term_id]  
    def get_relation(self,rel_id):
        return self.id_to_concept_relations[rel_id]
    def get_outward_edges(self,subject_concept_id:str):
        return self.outward_conceptual_edges[subject_concept_id]
    def get_inward_edges(self,object_concept_id:str):
        return self.inward_conceptual_edges[object_concept_id]    
    def get_terms_from_concept_id(self,concept_id:str):
        return [self.get_term(term_id) for term_id in self.concept_id_to_term_ids[concept_id]]
    def get_concepts_from_term_id(self,term_id:str):
        return [self.get_concept(term_id) for term_id in self.term_id_to_concept_ids[term_id]]
    def get_data(self):
        concepts =  set(self.id_to_concept.values())
        terms = set(self.id_to_term.values())
        conceptual_relations = set(self.id_to_concept_relations.values())
        conceptual_edges = set()
        for concept_id_1, related_concepts in self.outward_conceptual_edges.items():
            for rel,concept_2 in related_concepts:
                conceptual_edges.add(Conceptual_Edge(concept_id_1,concept_2.id,rel.id))
        lexical_edges = set()
        for concept_id,term_ids in self.concept_id_to_term_ids.items():
            for term_id in term_ids:
                lexical_edges.add(Lexical_Edge(concept_id,term_id))
        return concepts,terms,conceptual_edges,lexical_edges,conceptual_relations
    def get_concept_ids(self):
        return set(self.id_to_concept.keys())

    






