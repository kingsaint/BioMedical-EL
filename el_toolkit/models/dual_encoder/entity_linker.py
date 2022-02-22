from el_toolkit.models.dual_encoder.document_embedder import Document_Embedder
from el_toolkit.models.dual_encoder.entity_embedder import Entity_Embedder
from el_toolkit.mpi_utils import partition


class Entity_Linker:
    def __init__(self,tokenizer,model,entities,hvd):
        self._document_embedder = Document_Embedder(tokenizer,model)
        self._entity_embedder = Entity_Embedder(tokenizer,model)
        self._all_candidate_embeddings =  self.embed_all_entities_distributed(entities=entities)
        self._hvd = hvd
    def get_most_similar(self,encoded_doc = None,doc = None):
        #NOTE:Could use batches of encoded_docs? Currently takes batch of mentions,
        if encoded_doc == None:
            if doc == None:
                raise ArgumentError
            else: 
                encoded_doc = self._encoder(doc)
        if hasattr(self._model, "module"):
            hidden_size = self._model.module.hidden_size
        else:
            hidden_size = self._model.hidden_size
        mention_embeddings = self._document_embedder.get_mention_embeddings(encoded_doc)
        # Perform similarity search
        num_m = mention_embeddings.size(0)  #
        all_candidate_embeddings_ = self._all_candidate_embeddings.unsqueeze(0).expand(num_m, -1, hidden_size) # M X C_all X H

        # candidate_indices = candidate_indices[0]  # original size 1 X 10 -> 10
        similarity_scores = torch.bmm(mention_embeddings,
                                    all_candidate_embeddings_.transpose(1, 2))  # M X 1 X C_all
        similarity_scores = similarity_scores.squeeze(1)  # M X C_all
        # print(similarity_scores)
        _, candidate_indices = torch.topk(similarity_scores, k=args.num_candidates)

        return candidate_indices.cpu().detach().numpy().tolist()

    def embed_all_entities(self,encoded_entities=None,entities=None):#distributed
        if encoded_entities == None:
            if entities == None:
                raise ArgumentError
            else: 
                entity_partition = partition(entities,self._hvd.size(),self._hvd.rank())
                encoded_entities = self.encode_all_entities(entities)
        else:
            encoded_entities = partition(entities,self._hvd.size(),self._hvd.rank())
        entity_embeddings = self._entity_embedder.embed_entities(encoded_entities)
        all_entity_embeddings = self._hvd.allgather(entity_embeddings)
        return all_entity_embeddings

