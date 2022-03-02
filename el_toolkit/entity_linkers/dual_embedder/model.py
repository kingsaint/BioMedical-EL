class BertEmbedder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # self.bert = BertForPreTraining(config)

class BertMentionDetectorModel(BertPreTrainedModel):#Bert Embeddings + Span Scores
    #takes tokens, spits out hidden layer for each token
    def __init__(self, config, pretrained_bert):
        super().__init__(config)
        self.config = config
        self.num_tags = 3
        self.bert_mention = pretrained_bert
        self.hidden_size = config.hidden_size
        self.bound_classifier = nn.Linear(config.hidden_size, 3)
        self.init_modules()
        self.loss_fn_ner = nn.BCEWithLogitsLoss()
        self.BIGINT = 1e31
        self.BIGFLOAT = float(self.BIGINT)

    def init_modules(self):
        for module in self.bound_classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self,
                mention_token_ids,
                mention_token_masks,
                mention_start_indices=None,
                mention_end_indices=None
                ):
        mention_outputs = self.bert_mention.bert(
                input_ids=mention_token_ids,
                attention_mask=mention_token_masks,
            )
        last_hidden_states = mention_outputs[0]

        logits = self.bound_classifier(last_hidden_states)  # B X L X H --> B X L X 3
        start_scores, end_scores, mention_scores = logits.split(1, dim=-1)
        start_scores = start_scores.squeeze(-1)
        end_scores = end_scores.squeeze(-1)
        mention_scores = mention_scores.squeeze(-1)
        mask = torch.where(mention_token_masks==0,-self.BIGFLOAT,1.0).to(mention_token_masks.device)
        # Exclude The masked tokens from start or end mentions
        masked_start_scores = start_scores+mask
        masked_end_scores =end_scores+mask
        masked_mention_scores = mention_scores+mask
        cumulative_mention_scores = torch.zeros(mention_token_masks.size()).to(mention_token_masks.device)
        cumulative_mention_scores[torch.where(mention_token_masks == 0)] = -self.BIGINT
        cumulative_mention_scores[:, 0] = masked_mention_scores[:, 0]
        for i in range(1, mention_token_masks.size(1)):
            cumulative_mention_scores[:, i] = cumulative_mention_scores[:, i-1] + masked_mention_scores[:, i]

        all_span_mention_scores = cumulative_mention_scores.unsqueeze(1) - cumulative_mention_scores.unsqueeze(2)
        all_span_mention_scores += masked_mention_scores.unsqueeze(2).expand_as(all_span_mention_scores)

        # Add the start scores and end scores
        all_span_start_end_scores = masked_start_scores.unsqueeze(2) + masked_end_scores.unsqueeze(1)

        # Add the mention span scores with the sum of start scores and end scores
        all_span_scores = all_span_start_end_scores + all_span_mention_scores

        # Mention end cannot have a higher index than mention start
        impossible_mask = torch.zeros((all_span_scores.size(1), all_span_scores.size(2)), dtype=torch.int32).to(mention_token_masks.device)
        possible_rows, possible_cols = torch.triu_indices(all_span_scores.size(1), all_span_scores.size(2))
        impossible_mask[(possible_rows, possible_cols)] = 1
        impossible_mask = impossible_mask.unsqueeze(0).expand_as(all_span_scores)
        all_span_scores[torch.where(impossible_mask == 0)] = -self.BIGINT

        # Spans with logprobs  minus "Inf" are invalid
        all_spans = torch.where(all_span_scores > -self.BIGINT)
        all_doc_indices = all_spans[0]
        all_start_indices = all_spans[1]
        all_end_indices = all_spans[2]

        # Spans longer than `max_mention_length` are invalid
        span_lengths = all_end_indices - all_start_indices + 1
        valid_span_indices = torch.where(span_lengths <= args.max_mention_length)
        valid_doc_indices = all_doc_indices[valid_span_indices]
        valid_start_indices = all_start_indices[valid_span_indices]
        valid_end_indices = all_end_indices[valid_span_indices]

        # Valid spans and their scores
        valid_spans = torch.stack((valid_doc_indices, valid_start_indices, valid_end_indices), dim=0)
        valid_span_scores = all_span_scores[(valid_doc_indices, valid_start_indices, valid_end_indices)]

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
            return (ner_loss,) + (last_hidden_states,)
        else:
            # Inference supports batch_size=1
            return (None,) + (last_hidden_states,)

class BertDualEmbedderModel(BertPreTrainedModel):
    #takes tokens, spits out hidden layer for each token
    def __init__(self, config, pretrained_bert,span_detector):
        super().__init__(config)
        self.config = config
        self.num_tags = 3
        self.bert_mention = pretrained_bert
        self.span_detector = span_detector
        self.hidden_size = config.hidden_size
        self.bound_classifier = nn.Linear(config.hidden_size, 3)
        self.init_modules()
        self.loss_fn_linker = nn.CrossEntropyLoss(ignore_index=-1)
        self.BIGINT = 1e31
        self.BIGFLOAT = float(self.BIGINT)

    def init_modules(self):
        for module in self.bound_classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self,
                num_candidates,
                mention_token_ids,
                mention_token_masks,
                mention_start_indices=None,
                mention_end_indices=None,
                candidate_token_ids,
                candidate_token_masks,
                all_candidate_embeddings=None,
                labels=None,
                ):
        ner_loss,last_hidden_states = self.span_detector(mention_token_ids=mention_token_ids,
                                                    ention_token_masks=mention_token_masks,
                                                    mention_start_indices=mention_start_indices,
                                                    mention_end_indices=mention_end_indices
                                                    )
        
        ''' For random negative training and  For tf-idf candidates based training and evaluation'''
        if self.training:
            assert all_candidate_embeddings is None
            assert mention_start_indices is not None
            assert labels is not None
            candidate_mask = torch.sum(candidate_token_ids, dim=2)  # B X C
            non_zeros = torch.where(candidate_mask > 0)
            candidate_mask[non_zeros] = 1  # B X C
            candidate_mask = candidate_mask.float()
            # Pool the mention representations
            mention_embeddings = []
            for i in range(mention_start_indices.size(0)):
                for j in range(mention_start_indices.size(1)):
                    s_idx = mention_start_indices[i][j].item()
                    e_idx = mention_end_indices[i][j].item()
                    m_embd = torch.mean(last_hidden_states[:, s_idx:e_idx + 1, :], dim=1)
                    mention_embeddings.append(m_embd)
            mention_embeddings = torch.cat(mention_embeddings, dim=0).unsqueeze(1)

            b_size, n_c, seq_len = candidate_token_ids.size()
            candidate_token_ids = candidate_token_ids.reshape(-1, seq_len)  # B(N*C) X L
            candidate_token_masks = candidate_token_masks.reshape(-1, seq_len)  # B(N*C) X L

            candidate_outputs = self.bert_candidate.bert(
                input_ids=candidate_token_ids,
                attention_mask=candidate_token_masks,
            )
            pooled_candidate_outputs = candidate_outputs[1]

            candidate_embeddings = pooled_candidate_outputs.reshape(-1, num_candidates, self.hidden_size) #BN X C X H
            linker_logits = torch.bmm(mention_embeddings, candidate_embeddings.transpose(1, 2))
            linker_logits = linker_logits.squeeze(1)  # BN X C

            # logits = logits.reshape(b_size, n_c)  # B X C

            # Mask off the padding candidates
            candidate_mask = candidate_mask.reshape(-1, 2 * num_candidates)
            linker_logits = linker_logits - (1.0 - candidate_mask) * 1e31
            #logger.info(f"ned linking_logits:{linker_logits}"
            labels = labels.reshape(-1)
            #logger.info(f"ned labels:{labels}")
            linking_loss = self.loss_fn_linker(linker_logits, labels)
            return ner_loss + linking_loss, linker_logits
        else:
            b_size = mention_embeddings.size(0)
            all_candidate_embeddings = all_candidate_embeddings[0].unsqueeze(0).expand(b_size, -1, -1)  # B X C_all X H
            #logger.info(str(all_candidate_embeddings.size()))
            linker_logits = torch.bmm(mention_embeddings, all_candidate_embeddings.transpose(1, 2))
            linker_logits = linker_logits.squeeze(1)  # BN X C
            return None, linker_logits

class SimilarityScore(nn.Module):
    def __init__(self):
        self.loss_fn_linker = nn.CrossEntropyLoss(ignore_index=-1)
    def forward(mention_embeddings,
                candidate_embeddings,
                candidate_mask=None,
                labels=None):
        b_size = mention_embeddings.size(0)
        candidate_embeddings = candidate_embeddings[0].unsqueeze(0).expand(b_size, -1, -1)  # B X C X H
        #logger.info(str(all_candidate_embeddings.size()))
        linker_logits = torch.bmm(mention_embeddings, candidate_embeddings.transpose(1, 2))
        linker_logits = linker_logits.squeeze(1)  # BN X C
         # Mask off the padding candidates
        if candidate_mask != None:
            candidate_mask = candidate_mask.reshape(-1, 2 * num_candidates)#HOW
            linker_logits = linker_logits - (1.0 - candidate_mask) * 1e31
        #logger.info(f"ned linking_logits:{linker_logits}"
        labels = labels.reshape(-1)
        #logger.info(f"ned labels:{labels}")
        linking_loss = self.loss_fn_linker(linker_logits, labels)
        return linking_loss, linker_logits