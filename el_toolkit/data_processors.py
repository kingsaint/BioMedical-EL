from el_toolkit.document import Document,Mention
from el_toolkit.lexical_knowledge_base import Concept, Knowledge_Data,RDF_Lexical_Knowledge_Base
from rdflib import Graph, URIRef, Literal, Namespace

def remove_overlaps(doc:Document,broad_strategy:bool=True) -> Document:
    if broad_strategy:
        no_overlap_mentions = []
        for mention_1 in doc.mentions:
            contained=False
            for mention_2 in doc.mentions:
                if mention_1.start_index >= mention_2.start_index and mention_1.end_index <= mention_2.end_index and mention_1 != mention_2:
                    contained = True
            if not contained:
                no_overlap_mentions.append(mention_1)
        return Document(doc.doc_id,doc.message,no_overlap_mentions)
    else:
        raise NotImplementedError


def segment_document(doc:Document,tokenizer,max_mention_per_new_doc) -> list[Document]:
    assert not doc.check_for_span_overlaps()
    def segment_recursive(remaining_text,remaining_mentions,doc_number=0,prev_seq_len=0):
        new_doc_id = doc.doc_id + "_" + str(doc_number)
        omitted_mentions = 0
        new_document_mentions = []
        mentions_added = 0 
        segment_text = ""
        while mentions_added < len(remaining_mentions):#we've processed all remaining mentions
            mention_to_add =  remaining_mentions[mentions_added]
            mention_start_index_in_new_doc = mention_to_add.start_index - prev_seq_len
            mention_end_index_in_new_doc = mention_to_add.end_index - prev_seq_len
            tentative_segment_text = remaining_text[:mention_end_index_in_new_doc]
            tokens = tokenizer.tokenize(tentative_segment_text)
            if (len(new_document_mentions) < max_mention_per_new_doc and len(['[CLS]'] + tokens + ['[SEP]']) < 256):
                segment_text = tentative_segment_text
                #new_mention_id = new_doc_id + str((len(new_document_mentions) + omitted_mentions) % max_mention_per_new_doc)
                if mention_start_index_in_new_doc < mention_end_index_in_new_doc  and mention_start_index_in_new_doc>= 0 and mention_end_index_in_new_doc > 0:
                    new_document_mentions.append(Mention(mention_start_index_in_new_doc,mention_end_index_in_new_doc,mention_to_add.concept_id))
                else:
                    omitted_mentions += 1
                mentions_added += 1
            else:
                segmented_doc = Document(new_doc_id,segment_text,new_document_mentions)
                remaining_segmented_docs,remaining_omitted_mentions = segment_recursive(remaining_text[len(segment_text):],remaining_mentions[mentions_added:],doc_number+1,prev_seq_len+len(segment_text))
                return [segmented_doc] + remaining_segmented_docs, omitted_mentions + remaining_omitted_mentions
        segmented_doc = Document(new_doc_id,segment_text,new_document_mentions)
        return [segmented_doc],omitted_mentions#base case
    segmented_docs,omitted_mentions = segment_recursive(doc.message,doc.mentions)
    print(f"Mentions Omitted: {omitted_mentions}")
    return segmented_docs


def derive_domain(kd:Knowledge_Data,concept_id:str,isa_relation_name:str) -> list[Concept]:
    lkb = RDF_Lexical_Knowledge_Base(kd)
    q = """
            SELECT ?object_id ?rel_id WHERE {?subject_concept ?concept_relation ?object_concept .
                                            ?concept_relation RDF:type VOCAB:Concept_Relation .
                                            ?subject_concept VOCAB:id ?subject_id .
                                            ?object_concept VOCAB:id ?object_id .
                                            ?concept_relation VOCAB:label ?rel_label
                                     }
        """
    qres = lkb.sparql_query(q,initBindings={'rel_label':Literal(isa_relation_name, datatype=XSD.string)})
    return [(lkb.get_relation(str(row.rel_id)),lkb.get_concept(str(row.object_id))) for row in qres]
    
