import copy
from transformers import BertModel
import torch.nn as nn
import torch

class BertMentionDetectorModel(nn.Module):#Bert Embeddings + Span Scores
    #takes tokens, spits out hidden layer for each token
    def __init__(self,bert_model,max_mention_length=256):
        super().__init__()
        self.bert_mention = bert_model
        self.hidden_size = bert_model.config.hidden_size
        self.max_mention_length = max_mention_length
        self.bound_classifier = nn.Linear(self.hidden_size, 3)
        self.init_modules()
        self.loss_fn_ner = nn.BCEWithLogitsLoss()
        self.BIGINT = 1e31
        self.BIGFLOAT = float(self.BIGINT)
    @classmethod
    def from_pretrained_bert_filepath(cls,filepath,max_mention_length=256):
        bert_model = BertModel.from_pretrained(filepath)
        return cls(bert_model,max_mention_length)
    def init_modules(self):
        for module in self.bound_classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.bert_mention.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    def forward(self,
                doc_token_ids,#B x DL
                doc_token_masks,#B x DL
                mention_start_indices=None,#B x MN
                mention_end_indices=None,#B x MN
                ):
        mention_outputs = self.bert_mention.bert(
                input_ids=doc_token_ids,
                attention_mask=doc_token_masks,
            )
        last_hidden_states = mention_outputs[0] #B x DL x H

        logits = self.bound_classifier(last_hidden_states)  #B x DL x 3
        start_scores, end_scores, mention_scores = logits.split(1, dim=-1) #B x DL x 1
        start_scores = start_scores.squeeze(-1)#B x DL
        end_scores = end_scores.squeeze(-1)#B x DL
        mention_scores = mention_scores.squeeze(-1)#B x DL
        mask = torch.where(doc_token_masks==0,-self.BIGFLOAT,1.0).to(doc_token_masks.device)#B x DL
        # Exclude The masked tokens from start or end mentions
        masked_start_scores = start_scores+mask#B x DL
        masked_end_scores =end_scores+mask#B x DL
        masked_mention_scores = mention_scores+mask#B x DL
        cumulative_mention_scores = torch.zeros(doc_token_masks.size()).to(doc_token_masks.device)#B x DL
        cumulative_mention_scores[torch.where(doc_token_masks == 0)] = -self.BIGINT#B x DL
        cumulative_mention_scores[:, 0] = masked_mention_scores[:, 0]
        for i in range(1, doc_token_masks.size(1)):
            cumulative_mention_scores[:, i] = cumulative_mention_scores[:, i-1] + masked_mention_scores[:, i]#B x DL

        all_span_mention_scores = cumulative_mention_scores.unsqueeze(1) - cumulative_mention_scores.unsqueeze(2)#B x 1 x DL - B x DL x 1 => B x DL x DL (Broadcasting)
        all_span_mention_scores += masked_mention_scores.unsqueeze(2).expand_as(all_span_mention_scores)#B x DL x DL

        # Add the start scores and end scores
        all_span_start_end_scores = masked_start_scores.unsqueeze(2) + masked_end_scores.unsqueeze(1)# B x DL x DL

        # Add the mention span scores with the sum of start scores and end scores
        all_span_scores = all_span_start_end_scores + all_span_mention_scores# B x DL x DL

        # Mention end cannot have a higher index than mention start
        impossible_mask = torch.zeros((all_span_scores.size(1), all_span_scores.size(2)), dtype=torch.int32).to(doc_token_masks.device)#DL x DL
        possible_rows, possible_cols = torch.triu_indices(all_span_scores.size(1), all_span_scores.size(2)) #(DL^2+ DL)/2,(DL^2+ DL)/2
        impossible_mask[(possible_rows, possible_cols)] = 1 #DL x DL
        impossible_mask = impossible_mask.unsqueeze(0).expand_as(all_span_scores)# B x DL x DL
        all_span_scores[torch.where(impossible_mask == 0)] = -self.BIGINT# B x DL x DL

        # Spans with logprobs  minus "Inf" are invalid
        all_spans = torch.where(all_span_scores > -self.BIGINT)#3 x S
        all_doc_indices = all_spans[0] #S
        all_start_indices = all_spans[1] #S
        all_end_indices = all_spans[2] #S

        # Spans longer than `max_mention_length` are invalid
        span_lengths = all_end_indices - all_start_indices + 1 #S
        valid_span_indices = torch.where(span_lengths <= self.max_mention_length)#S (after pruning spans which are too long)
        valid_doc_indices = all_doc_indices[valid_span_indices]#S (after pruning spans which are too long)
        valid_start_indices = all_start_indices[valid_span_indices]#S (after pruning spans which are too long)
        valid_end_indices = all_end_indices[valid_span_indices]#S (after pruning spans which are too long)

        # Valid spans and their scores
        valid_spans = torch.stack((valid_doc_indices, valid_start_indices, valid_end_indices), dim=0)#3 x S
        valid_span_scores = all_span_scores[(valid_doc_indices, valid_start_indices, valid_end_indices)]#S

        if self.training:
            # Target
            targets = torch.zeros(valid_spans.size(-1), dtype=torch.float32).to(valid_spans.device)
            gold_spans = set()
            gold_start_end_indices = torch.stack([mention_start_indices, mention_end_indices], dim=1)
            for i in range(gold_start_end_indices.size(0)):
                for j in range(gold_start_end_indices.size(-1)):
                    # (0, 0) can't be a mention span because 0-th index corresponds to [CLS]
                    if gold_start_end_indices[i][0][j].item() == 0 and gold_start_end_indices[i][1][j].item() == 0:
                        continue
                    # -1 is the paddding index
                    if gold_start_end_indices[i][0][j].item() == -1:
                        continue
                    else:
                        gold_spans.add(
                            (i, gold_start_end_indices[i][0][j].item(), gold_start_end_indices[i][1][j].item()))
            for i in range(valid_spans.size(-1)):
                pred_span = (valid_spans[0][i].item(), valid_spans[1][i].item(), valid_spans[2][i].item())
                if pred_span in gold_spans:
                    targets[i] = 1.0
            # Binary Cross Entropy loss
            ner_loss = self.loss_fn_ner(valid_span_scores, targets)
            #logger.info(f"ner_loss:{ner_loss}")
            return last_hidden_states,ner_loss,None,None,None
        else:
            # Inference supports batch_size=1
            inferred_doc_indices,inferred_start_indices,inferred_end_indices = self.infer(valid_doc_indices,valid_start_indices, valid_end_indices, valid_span_scores)
            
            return last_hidden_states,None,inferred_doc_indices,inferred_start_indices,inferred_end_indices
    def get_training_mention_embeddings(self,
                                        mention_start_indices,#B x MN
                                        mention_end_indices,#B x MN
                                        last_hidden_states #B x DL x H
                        ):
        # Pool the mention representations
        mention_embeddings = []
        for i in range(mention_start_indices.size(0)):
            for j in range(mention_start_indices.size(1)):
                s_idx = mention_start_indices[i][j].item()
                if s_idx != -1:#padding mentionx
                    e_idx = mention_end_indices[i][j].item()
                    m_embd = torch.mean(last_hidden_states[i, s_idx:e_idx + 1, :], dim=1)
                    mention_embeddings.append(m_embd)
        mention_embeddings = torch.cat(mention_embeddings, dim=0)
        return mention_embeddings # (B*MN) x H all mentions
    def get_eval_mention_embeddings(self,inferred_doc_indices,inferred_start_indices,inferred_end_indices,last_hidden_states):
        mention_embeddings = []
        for i in range(inferred_doc_indices(0)):
            doc_idx = inferred_doc_indices[i]
            s_idx = inferred_start_indices[i]
            e_idx = inferred_end_indices[i]
            m_embd = torch.mean(last_hidden_states[doc_idx, s_idx:e_idx + 1, :], dim=1)
            mention_embeddings.append(m_embd)
        return mention_embeddings # (MB) x H
    def infer(valid_doc_indices,valid_start_indices, valid_end_indices, valid_span_scores,threshold=.8):
        inferred_span_indices = torch.where(valid_span_scores > threshold)
        return valid_doc_indices[inferred_span_indices],valid_start_indices[inferred_span_indices],valid_end_indices[inferred_span_indices]

class DualEmbedderModel(nn.Module):
    def __init__(self,bert_mention,candidate_embedder):
        super().__init__()
        self.candidate_embedder = candidate_embedder
        self.span_detector = BertMentionDetectorModel(bert_mention)
        self.loss_fn_linker = nn.CrossEntropyLoss(ignore_index=-1)
        self.BIGINT = 1e31
    def forward(self,
                doc_token_ids,#B x DL
                doc_token_masks,#B x DL
                mention_start_indices=None,#B x MN
                mention_end_indices=None,#B x MN
                all_candidate_embeddings=None, #C x H (Not batched)
                labels=None,#B x MN,
                candidate_masks=None,#B x (MN * C) 
                **kwargs
                
                ):
        if self.training:
            assert mention_start_indices is not None
            assert mention_end_indices is not None
            assert labels is not None
            last_hidden_states,ner_loss,_,_,_ = self.span_detector(doc_token_ids=doc_token_ids,
                                                                   doc_token_masks=doc_token_masks,
                                                                   mention_start_indices=mention_start_indices,
                                                                   mention_end_indices=mention_end_indices
                                                                    ) #(B*MN) * H
            mention_embeddings = self.span_detector.get_training_mention_embeddings(mention_start_indices,mention_end_indices,last_hidden_states)
            
            pooled_candidate_outputs,n_c = self.get_candidate_embeddings(**kwargs)
            n_c = self.get_number_candidates(**kwargs)
            candidate_embeddings = pooled_candidate_outputs.reshape(-1, n_c, self.span_detector.hidden_size) #(B*MN) X C X H
            linker_logits = self.similarity_score(mention_embeddings,candidate_embeddings)#(B*MN) x C
            candidate_masks = candidate_masks.reshape(-1,n_c)# (B*MN) x C
            #Mask off Padding Candidate
            linker_logits = linker_logits - (1.0 - candidate_masks) * self.BIGINT# (B*MN) x C
            
            labels.reshape(-1)# MB
            linking_loss = self.loss_fn_linker(linker_logits, labels)
            return  linker_logits,ner_loss + linking_loss
        else:
            assert all_candidate_embeddings is not None
            last_hidden_states,_,inferred_doc_indices,inferred_start_indices,inferred_end_indices = self.span_detector(doc_token_ids=doc_token_ids,
                                                                                                                       doc_token_masks=doc_token_masks
                                                                                                                      )
            mention_embeddings = self.span_detector.get_eval_mention_embeddings(inferred_doc_indices,inferred_start_indices,inferred_end_indices)
            linker_logits = self.all_similarity_score(self,mention_embeddings,all_candidate_embeddings)
            return linker_logits,None
    def similarity_score(mention_embeddings,#(MB) x H
                         candidate_embeddings,#(MB) x C x H
                        ):
        #logger.info(str(all_candidate_embeddings.size()))
        linker_logits = torch.bmm(mention_embeddings.unsqueeze(1), candidate_embeddings.transpose(1, 2))# (MB) x 1 x C
        linker_logits = linker_logits.squeeze(1) #(MB) x C
        return linker_logits
    def all_similarity_score(self,
                            mention_embeddings,#(MB) x H
                            candidate_embeddings,#C x H
                            ):
        mention_batch_size = mention_embeddings.size(0)
        candidate_embeddings = candidate_embeddings.expand(mention_batch_size,-1,-1)#(MB) x C x H
        return self.similarity_score(mention_embeddings,candidate_embeddings) #(MB) x C
    def get_candidate_embeddings(self,**kwargs):
        raise NotImplementedError
    def get_number_of_candidates(self,**kwargs):
        raise NotImplementedError

class BertDualEmbedderModel(DualEmbedderModel):
    def __init__(self,*args):
        super().__init__(*args)
        assert type(self.candidate_embedder) == BertModel
    @classmethod
    def from_pretrained_bert_filepath(cls,filepath):
        bert_mention = BertModel.from_pretrained(filepath)
        bert_candidate = copy.deepcopy(bert_mention)
        return cls(bert_mention,bert_candidate)
    
    def get_candidate_embeddings(self,
                                 candidate_token_ids,#B x (MN * C) x CL
                                 candidate_token_masks#B x (MN * C) x CL (Not sure why they originally flattened this tensor)   
                                 ):
        seq_len = candidate_token_ids.size(2)
        candidate_token_ids = candidate_token_ids.reshape(-1, seq_len)  # (B*MN*C) X CL (flatten)
        candidate_token_masks = candidate_token_masks.reshape(-1, seq_len)  # (B*MN*C) X CL (flatten)

        candidate_outputs = self.bert_candidate.bert(
            input_ids=candidate_token_ids,
            attention_mask=candidate_token_masks,
        )
        return candidate_outputs[1]
    def get_number_of_candidates(self,candidate_token_ids):
        return candidate_token_ids.size()[1]

class MLPEmbedderModel(DualEmbedderModel):
    def __init__(self,*args):
        super().__init__(*args)
        assert type(self.candidate_embedder) == torch.nn.Linear
    def get_candidate_embeddings(self,pretrained_concept_embedding):#get fine-tuned concept embedding 
            return self.candidate_embedder(pretrained_concept_embedding)
    def get_number_of_candidates(self,pretrained_concept_embedding):
        return pretrained_concept_embedding.size(1)