from argparse import ArgumentError
from el_toolkit.models.dual_encoder.encoder import Entity_Encoder
from el_toolkit.models.dual_encoder.featurizer import Encoder
from mpi_utils import partition

class Entity_Embedder:
    def __init__(self,model,tokenizer,max_seq_length = 8):
        self._model = model
        self._encoder = Entity_Encoder(tokenizer,max_seq_length)
    def get_candidate_embedding(self,encoded_entity=None,entity=None):
        if encoded_entity == None:
            if entity == None:
                raise ArgumentError
            else: 
                encoded_entity = self._encoder(entity)
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
    def get_all_candidate_embeddings(encoded_entities,entities):#distributed

    
    