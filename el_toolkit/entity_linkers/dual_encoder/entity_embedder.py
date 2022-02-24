import functools
from el_toolkit.mpi_utils import partition

class Entity_Embedder:
    def __init__(self,model,hvd=None):
        self._model = model
        self._hvd = hvd
        self._distributed = self._hvd == None
    def embed_entity(self,encoded_entity):
        candidate_embeddings = []
        #logger.info("INFO: Collecting candidate embeddings.")
        with torch.no_grad():
            candidate_token_ids = torch.LongTensor([encoded_entity.entity_token_ids]).to(self._model.device)
            candidate_token_masks = torch.LongTensor([encoded_entity.entity_tokens_masks]).to(self._model.device)
            candidate_outputs = self._model.bert_candidate.bert(input_ids=candidate_token_ids,attention_mask=candidate_token_masks)
            candidate_embedding = candidate_outputs[1]
            candidate_embeddings.append(candidate_embedding)
            #logger.info(str(candidate_embedding))
        candidate_embeddings = torch.cat(candidate_embeddings, dim=0)
        #logger.info("INFO: Collected candidate embeddings.")
        return candidate_embeddings
    def embed_all_entities(self,encoded_entities):#distributed
        if self._distributed:
            entity_keys = partition(list(encoded_entities.keys()),self._hvd.size(),self._hvdhvd.rank())
            single_node_encoded_entities = {encoded_entities[entity_key].concept_id:self._entity_embedder.embed_entity(encoded_entities[entity_key]) for entity_key in entity_keys}
            all_entity_embeddings = self._hvd.allgather(single_node_encoded_entities)
            entity_embeddings = functools.reduce(lambda dict_1,dict_2: {**dict_1,**dict_2},all_entity_embeddings)
            return entity_embeddings
        else:
            entity_embeddings = {entity.concept_id:self._entity_encoder.embed_entity(encoded_entities[entity]) for entity in encoded_entities}
        return Entity_Embeddings(all_entity_embeddings)

class Entity_Embeddings:
    def __init__(self,entity_embeddings):
        self._entity_embeddings = entity_embeddings
    def get_most_similar(self,encoded_doc,k=8):
        #NOTE:Could use batches of encoded_docs? Currently takes batch of mentions,
        if hasattr(self._model, "module"):
            hidden_size = self._model.module.hidden_size
        else:
            hidden_size = self._model.hidden_size
        mention_embeddings = self._document_embedder.get_mention_embeddings(encoded_doc)
        # Perform similarity search
        num_m = mention_embeddings.size(0)  #
        all_candidate_embeddings_ = self._entity_embeddings.unsqueeze(0).expand(num_m, -1, hidden_size) # M X C_all X H

        # candidate_indices = candidate_indices[0]  # original size 1 X 10 -> 10
        similarity_scores = torch.bmm(mention_embeddings,
                                    all_candidate_embeddings_.transpose(1, 2))  # M X 1 X C_all
        similarity_scores = similarity_scores.squeeze(1)  # M X C_all
        # print(similarity_scores)
        _, candidate_indices = torch.topk(similarity_scores, k)

        return candidate_indices.cpu().detach().numpy().tolist()
    
    