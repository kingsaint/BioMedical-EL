#get device from model?


from argparse import ArgumentError

from el_toolkit.models.dual_encoder.encoder import Doc_Encoder


class Document_Embedder:
    def __init__(self,model,tokenizer,max_seq_length=8):
        self._model = model
        self._tokenizer = tokenizer
        self._document_embedder = Doc_Encoder(tokenizer,max_seq_length)
    def get_last_hidden_states(self,encoded_doc):
        # Hard negative candidate mining
        # print("Performing hard negative candidate mining ...")
        # Get mention embeddings
        input_token_ids = torch.LongTensor([encoded_doc.token_ids]).to(self._model.device)
        input_token_masks = torch.LongTensor([encoded_doc.doc_tokens_mask]).to(self._model.device)
        # Forward pass through the mention encoder of the dual encoder
        with torch.no_grad():
            if hasattr(self._model, "module"):
                mention_outputs = self._model.module.bert_mention.bert(
                    input_ids=input_token_ids,
                    attention_mask=input_token_masks,
                )
            else:
                mention_outputs = self._model.bert_mention.bert(
                    input_ids=input_token_ids,
                    attention_mask=input_token_masks,
                )
        last_hidden_states = mention_outputs[0]  # B X L X H
        return last_hidden_states
        
    def get_mention_embeddings(self,encoded_doc=None,doc=None):
        if encoded_doc == None:
            if doc == None:
                raise ArgumentError
            else: 
                encoded_doc = self._encoder(doc)
        last_hidden_states = self.embed(encoded_doc)
        mention_embeddings = []
        for i, (s_idx, e_idx) in enumerate(zip(encoded_doc.mention_start_markers, encoded_doc.mention_end_markers)):
            m_embd = torch.mean(last_hidden_states[:, s_idx:e_idx+1, :], dim=1)
            mention_embeddings.append(m_embd)
        return torch.cat(mention_embeddings, dim=0).unsqueeze(1)
        