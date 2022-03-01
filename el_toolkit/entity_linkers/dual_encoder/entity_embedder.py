from el_toolkit.mpi_utils import partition


class Entity_Embedder:#takes an lkb and produces embeddin
    def __init__(self):
        raise NotImplementedError

    def embed_entity(self,encoded_entity):
        raise NotImplementedError

    def embed_entities(self,encoded_entity):#return Entity_Embeddings object
        raise NotImplementedError

class Entity_Embeddings:
    def __init__(self,entity_embedding_tensor,embedding_idx):
        self._entity_embeddings_tensor = entity_embeddings
    def get_most_similar_concept_idx(self,encoded_doc):
        num_m = mention_embeddings.size(0)  #
        all_candidate_embeddings_ = self._entity_embeddings_tensor.unsqueeze(0).expand(num_m, -1, hidden_size) # M X C_all X H

        # candidate_indices = candidate_indices[0]  # original size 1 X 10 -> 10
        similarity_scores = torch.bmm(mention_embeddings,
                                    all_candidate_embeddings_.transpose(1, 2))  # M X 1 X C_all
        similarity_scores = similarity_scores.squeeze(1)  # M X C_all
        # print(similarity_scores)
        _, candidate_indices = torch.topk(similarity_scores, k)
        return candidate_indices

class Dual_Encoder_Entity_Embedder(Entity_Embedder):
    def __init__(self,model,entity_encoder,hvd=None):
        self._model = model
        self._hvd = hvd
        self._distributed = self._hvd == None
    def embed_entity(self,encoded_entity):
        #encode entity first
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
    def embed_entities(self,encoded_entities):#distributed
        if self._distributed:
            encoded_entities =  partition(encoded_entities,self._hvd.size(),self._hvd.rank())
        entity_embeddings = [self.embed_entity(encoded_entity) for encoded_entity in encoded_entities]
        if self._distributed:
            entity_embeddings = self._hvd.allgather(entity_embeddings)
        return Entity_Embeddings(entity_embeddings)


    
    