import os
import json
import random
import math
import logging
from mpi4py import MPI
import numpy as np
from torch.utils.data.dataset import TensorDataset
logger = logging.getLogger(__name__)
import torch
from .modeling_bert import BertModel
from .tokenization_bert import BertTokenizer
from .configuration_bert import BertConfig
from .modeling_e2e_span import DualEncoderBert, PreDualEncoder

import horovod.torch as hvd
import mlflow



MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
}
comm = None

def get_comm_magic():
    global comm
    if comm is None:
      comm = MPI.COMM_WORLD
    return comm

def load_data(data_dir, mode):
    logger.info("***getting data***")
    if 'NCBI' in data_dir:
        entity_path = './data/NCBI_Disease/raw_data/entities.txt'
    elif 'BC5CDR' in data_dir:
        entity_path = '/dbfs/Workspace/Repos/cflowers@trend.community/BioMedical-EL/data/BC5CDR/raw_data/entities.txt'
    elif 'st21pv' in data_dir:
        entity_path = './data/MM_st21pv_CUI/raw_data/entities.txt'
    elif 'aida' in data_dir:
        entity_path = './data/aida-yago2-dataset/raw_data/entities.txt'
    elif 'dummy' in data_dir:
        entity_path = './data/dummy_data/raw_data/entities.txt'
    else:
        entity_path = './data/MM_full_CUI/raw_data/entities.txt'
    entities = {}
    with open(entity_path, encoding='utf-8') as f:
        for line in f:
            if 'BC5CDR' in data_dir:
                e, text = line.strip().split('\t')
            else:
                e, _, text = line.strip().split('\t')
            entities[e] = text

    file_path = os.path.join(data_dir, mode, 'documents/documents.json')
    docs = {}
    with open(file_path, encoding='utf-8') as f:
        print("documents dataset is loading......")
        for line in f:
            fields = json.loads(line.strip())
            docs[fields["document_id"]] = {"text": fields["text"]}
        print("documents dataset is done :)")

    # doc_ids = list(docs.keys())
    file_path = os.path.join(data_dir, mode, 'mentions/mentions.json')
    ments = {}
    with open(file_path, encoding='utf-8') as f:
        print("mentions {} dataset is loading......".format(mode))
        # doc_idx = 0
        for line in f:
            # doc_id = doc_ids[doc_idx]
            doc_mentions = json.loads(line.strip())
            if len(doc_mentions) > 0:
                doc_id = doc_mentions[0]["content_document_id"]
                ments[doc_id] = json.loads(line)
            # for line in f:
            #     fields = json.loads(line)
            #     ments[fields["mention_id"]] = {k: v for k, v in fields.items() if k != "mention_id"}
        print("mentions {} dataset is done :)".format(mode))
    return ments, docs, entities

def get_window(prefix, mention, suffix, max_size):
    if len(mention) >= max_size:
        window = mention[:max_size]
        return window, 0, len(window) - 1

    leftover = max_size - len(mention)
    leftover_half = int(math.ceil(leftover / 2))

    if len(prefix) >= leftover_half:
        prefix_len = leftover_half if len(suffix) >= leftover_half else \
                     leftover - len(suffix)
    else:
        prefix_len = len(prefix)

    prefix = prefix[-prefix_len:]  # Truncate head of prefix
    window = prefix + ['[Ms]'] + mention + ['[Me]'] + suffix
    window = window[:max_size]  # Truncate tail of suffix

    mention_start_index = len(prefix)
    mention_end_index = len(prefix) + len(mention) - 1

    return window, mention_start_index, mention_end_index


def get_mention_window(mention_id, mentions, docs,  max_seq_length, tokenizer):
    max_len_context = max_seq_length - 2 # number of characters
    # Get "enough" context from space-tokenized text.
    content_document_id = mentions[mention_id]['content_document_id']
    context_text = docs[content_document_id]['text']
    start_index = mentions[mention_id]['start_index']
    end_index = mentions[mention_id]['end_index']
    prefix = context_text[max(0, start_index - max_len_context): start_index]
    suffix = context_text[end_index: end_index + max_len_context]
    extracted_mention = context_text[start_index: end_index]

    assert extracted_mention == mentions[mention_id]['text']

    # Get window under new tokenization.
    return get_window(tokenizer.tokenize(prefix),
                      tokenizer.tokenize(extracted_mention),
                      tokenizer.tokenize(suffix),
                      max_len_context)


def get_marked_mentions(document_id, mentions, docs,  max_seq_length, tokenizer, args):
    # print("Num mention in this doc =", len(mentions[document_id]))
    for m in mentions[document_id]:
        assert m['content_document_id'] == document_id

    context_text = docs[document_id]['text'].lower() if args.do_lower_case else docs[document_id]['text']
    tokenized_text = [tokenizer.cls_token]
    mention_start_markers = []
    mention_end_markers = []
    sequence_tags = []

    # print(len(context_text))
    # print(len(mentions[document_id]))
    prev_end_index = 0
    for m in mentions[document_id]:
        start_index = m['start_index']
        end_index = m['end_index']
        # print(start_index, end_index)
        if start_index >= len(context_text):
            continue
        extracted_mention = context_text[start_index: end_index]

        # Text between the end of last mention and the beginning of current mention
        prefix = context_text[prev_end_index: start_index]
        # Tokenize prefix and add it to the tokenized text
        prefix_tokens = tokenizer.tokenize(prefix)
        tokenized_text += prefix_tokens
        # The sequence tag for prefix tokens is 'O' , 'DNT' --> 'Do Not Tag'
        for j, token in enumerate(prefix_tokens):
            sequence_tags.append('O' if not token.startswith('##') else 'DNT')
        # Add mention start marker to the tokenized text
        mention_start_markers.append(len(tokenized_text))
        # tokenized_text += ['[Ms]']
        # Tokenize the mention and add it to the tokenized text
        mention_tokens = tokenizer.tokenize(extracted_mention)
        tokenized_text += mention_tokens
        # Sequence tags for mention tokens -- first token B, other tokens I
        for j, token in enumerate(mention_tokens):
            if j == 0:
                sequence_tags.append('B')
            else:
                sequence_tags.append('I' if not token.startswith('##') else 'DNT')

        # Add mention end marker to the tokenized text
        mention_end_markers.append(len(tokenized_text) - 1)
        # tokenized_text += ['[Me]']
        # Update prev_end_index
        prev_end_index = end_index

    suffix = context_text[prev_end_index:]
    if len(suffix) > 0:
        suffix_tokens = tokenizer.tokenize(suffix)
        tokenized_text += suffix_tokens
        # The sequence tag for suffix tokens is 'O'
        for j, token in enumerate(suffix_tokens):
            sequence_tags.append('O' if not token.startswith('##') else 'DNT')
    tokenized_text += [tokenizer.sep_token]

    return tokenized_text, mention_start_markers, mention_end_markers, sequence_tags


def get_entity_window(entity_text, max_entity_len, tokenizer):
    entity_tokens = tokenizer.tokenize(entity_text)
    if len(entity_tokens) > max_entity_len:
        entity_tokens = entity_tokens[:max_entity_len]
    return entity_tokens


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, mention_token_ids, mention_token_masks,
                 candidate_token_ids_1, candidate_token_masks_1,
                 candidate_token_ids_2, candidate_token_masks_2,
                 label_ids, mention_start_indices, mention_end_indices,
                 num_mentions, seq_tag_ids):
        self.mention_token_ids = mention_token_ids
        self.mention_token_masks = mention_token_masks
        self.candidate_token_ids_1 = candidate_token_ids_1
        self.candidate_token_masks_1 = candidate_token_masks_1
        self.candidate_token_ids_2 = candidate_token_ids_2
        self.candidate_token_masks_2 = candidate_token_masks_2
        self.label_ids = label_ids
        self.mention_start_indices = mention_start_indices
        self.mention_end_indices = mention_end_indices
        self.num_mentions = num_mentions
        self.seq_tag_ids = seq_tag_ids

tag_to_id_map = {'O': 0, 'B': 1, 'I': 2, 'DNT': -100}
def convert_tags_to_ids(seq_tags):
    seq_tag_ids = [-100]  # corresponds to the [CLS] token
    for t in seq_tags:
        seq_tag_ids.append(tag_to_id_map[t])
    seq_tag_ids.append(-100)  # corresponds to the [SEP] token
    return seq_tag_ids

def load_and_cache_examples(
    args, 
    tokenizer, 
    model=None
):  

    max_seq_length=args.max_seq_length
    mode = 'train' if args.do_train else 'test'
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
    args.data_dir,
    "cached_{}_{}".format(
        mode,
        list(filter(None, args.base_model_name_or_path.split("/"))).pop()),
    )
    if not os.path.exists(cached_features_file) or args.overwrite_cache:
        mentions, docs, entities = load_data(args.data_dir, mode)
        all_entities = list(entities.keys())
        all_entity_token_ids = []
        all_entity_token_masks = []
        for c_idx, c in enumerate(all_entities):
            entity_text = entities[c].lower() if args.do_lower_case else entities[c]
            max_entity_len = max_seq_length // 4  # Number of tokens
            entity_window = get_entity_window(entity_text, max_entity_len, tokenizer)
            # [CLS] candidate text [SEP]
            candidate_tokens = [tokenizer.cls_token] + entity_window + [tokenizer.sep_token]
            candidate_tokens = tokenizer.convert_tokens_to_ids(candidate_tokens)
            if len(candidate_tokens) > max_seq_length:
                candidate_tokens = candidate_tokens[:max_seq_length]
                candidate_masks = [1] * max_seq_length
            else:
                candidate_len = len(candidate_tokens)
                pad_len = max_seq_length - candidate_len
                candidate_tokens += [tokenizer.pad_token_id] * pad_len
                candidate_masks = [1] * candidate_len + [0] * pad_len

            assert len(candidate_tokens) == max_seq_length
            assert len(candidate_masks) == max_seq_length

            all_entity_token_ids.append(candidate_tokens)
            all_entity_token_masks.append(candidate_masks)   
        if args.use_hard_negatives or args.use_hard_and_random_negatives:
            if model is None:
                raise ValueError("`model` parameter cannot be None")
            all_candidate_embeddings = get_all_candidate_embeddings(args,model,all_entity_token_ids,all_entity_token_masks)
            if os.path.exists(os.path.join(args.data_dir, 'mention_hard_negatives.json')):
                with open(os.path.join(args.data_dir, 'mention_hard_negatives.json')) as f_hn:
                    mention_hard_negatives = json.load(f_hn)
            else:
                mention_hard_negatives = {}
        # Process the mentions
        features = []
        num_longer_docs = 0
        all_document_ids = []
        all_label_candidate_ids = []
        for (ex_index, document_id) in enumerate(partition(list(mentions.keys()),hvd.size(),hvd.rank())):
            if ex_index % 1000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(mentions))

            doc_tokens_, mention_start_markers, mention_end_markers, seq_tags = get_marked_mentions(document_id,
                                                                                mentions,
                                                                                docs,
                                                                                max_seq_length,
                                                                                tokenizer,
                                                                                args)

            doc_tokens = tokenizer.convert_tokens_to_ids(doc_tokens_)
            seq_tag_ids = convert_tags_to_ids(seq_tags)

            assert len(doc_tokens) == len(seq_tag_ids)


            if len(doc_tokens) > max_seq_length:
                print(len(doc_tokens))
                doc_tokens = doc_tokens[:max_seq_length]
                seq_tag_ids = seq_tag_ids[:max_seq_length]
                doc_tokens_mask = [1] * max_seq_length
                num_longer_docs += 1
                continue
            else:
                mention_len = len(doc_tokens)
                pad_len = max_seq_length - mention_len
                doc_tokens += [tokenizer.pad_token_id] * pad_len
                doc_tokens_mask = [1] * mention_len + [0] * pad_len
                seq_tag_ids += [-100] * pad_len

            assert len(doc_tokens) == max_seq_length
            assert len(doc_tokens_mask) == max_seq_length
            assert len(seq_tag_ids) == max_seq_length

            # Build list of candidates
            label_candidate_ids = []
            for m in mentions[document_id]:
                label_candidate_ids.append(m['label_candidate_id'])
                all_document_ids.append(document_id)
                all_label_candidate_ids.append(m['label_candidate_id'])

            candidates = []
            candidates_2 = None
            if args.do_train:
                if args.use_random_candidates:  # Random negatives
                    for m_idx, m in enumerate(mentions[document_id]):
                        m_candidates = []
                        m_candidates.append(label_candidate_ids[m_idx])  # positive candidate
                        candidate_pool = set(entities.keys()) - set([label_candidate_ids[m_idx]])
                        negative_candidates = random.sample(candidate_pool, args.num_candidates - 1)
                        m_candidates += negative_candidates
                        candidates.append(m_candidates)

                elif args.use_tfidf_candidates:  # TF-IDF negatives
                    for m_idx, m in enumerate(mentions[document_id]):
                        m_candidates = []
                        m_candidates.append(label_candidate_ids[m_idx])  # positive candidate
                        for c in m["tfidf_candidates"]:
                            if c != label_candidate_ids[m_idx] and len(m_candidates) < args.num_candidates:
                                m_candidates.append(c)
                        candidates.append(m_candidates)

                elif args.use_hard_and_random_negatives:
                    # First get the random negatives
                    for m_idx, m in enumerate(mentions[document_id]):
                        m_candidates = []
                        m_candidates.append(label_candidate_ids[m_idx])  # positive candidate
                        candidate_pool = set(entities.keys()) - set([label_candidate_ids[m_idx]])
                        negative_candidates = random.sample(candidate_pool, args.num_candidates - 1)
                        m_candidates += negative_candidates
                        candidates.append(m_candidates)

                    # Then get the hard negative
                    if model is None:
                        raise ValueError("`model` parameter cannot be None")
                    # Hard negative candidate mining
                    # print("Performing hard negative candidate mining ...")
                    # Get mention embeddings
                    input_token_ids = torch.LongTensor([doc_tokens]).to(args.device)
                    input_token_masks = torch.LongTensor([doc_tokens_mask]).to(args.device)
                    # Forward pass through the mention encoder of the dual encoder
                    with torch.no_grad():
                        if hasattr(model, "module"):
                            mention_outputs = model.module.bert_mention.bert(
                                input_ids=input_token_ids,
                                attention_mask=input_token_masks,
                            )
                        else:
                            mention_outputs = model.bert_mention.bert(
                                input_ids=input_token_ids,
                                attention_mask=input_token_masks,
                            )
                    last_hidden_states = mention_outputs[0]  # B X L X H
                    # Pool the mention representations
                    # mention_start_indices = torch.LongTensor([mention_start_markers]).to(args.device)
                    # mention_end_indices = torch.LongTensor([mention_end_markers]).to(args.device)
                    #
                    if hasattr(model, "module"):
                        hidden_size = model.module.hidden_size
                    else:
                        hidden_size = model.hidden_size
                    #
                    # mention_start_indices = mention_start_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
                    # mention_end_indices = mention_end_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
                    # mention_start_embd = last_hidden_states.gather(1, mention_start_indices)
                    # mention_end_embd = last_hidden_states.gather(1, mention_end_indices)
                    # if hasattr(model, "module"):
                    #     mention_embeddings = model.module.mlp(torch.cat([mention_start_embd, mention_end_embd], dim=2))
                    # else:
                    #     mention_embeddings = model.mlp(torch.cat([mention_start_embd, mention_end_embd], dim=2))
                    # mention_embeddings = mention_embeddings.reshape(-1, 1, hidden_size) # M X 1 X H

                    mention_embeddings = []
                    # print(mention_start_markers, mention_end_markers)
                    for i, (s_idx, e_idx) in enumerate(zip(mention_start_markers, mention_end_markers)):
                        m_embd = torch.mean(last_hidden_states[:, s_idx:e_idx+1, :], dim=1)
                        mention_embeddings.append(m_embd)
                    mention_embeddings = torch.cat(mention_embeddings, dim=0).unsqueeze(1)

                    # Perform similarity search
                    num_m = mention_embeddings.size(0)  #
                    all_candidate_embeddings_ = all_candidate_embeddings.unsqueeze(0).expand(num_m, -1, hidden_size) # M X C_all X H

                    # distance, candidate_indices = all_candidate_index.search(mention_embedding, args.num_candidates)
                    # candidate_indices = candidate_indices[0]  # original size 1 X 10 -> 10
                    # print(mention_embeddings)
                    similarity_scores = torch.bmm(mention_embeddings,
                                                all_candidate_embeddings_.transpose(1, 2))  # M X 1 X C_all
                    similarity_scores = similarity_scores.squeeze(1)  # M X C_all
                    # print(similarity_scores)
                    distance, candidate_indices = torch.topk(similarity_scores, k=args.num_candidates)

                    candidate_indices = candidate_indices.cpu().detach().numpy().tolist()
                    # print(candidate_indices)

                    # print(len(mentions[document_id]))
                    for m_idx, m in enumerate(mentions[document_id]):
                        mention_id = m["mention_id"]
                        # Update the list of hard negatives for this `mention_id`
                        if mention_id not in mention_hard_negatives:
                            mention_hard_negatives[mention_id] = []
                        # print(m_idx)
                        for i, c_idx in enumerate(candidate_indices[m_idx]):
                            c = all_entities[c_idx]
                            if not c == m["label_candidate_id"]:  # Positive candidate position
                                if c not in mention_hard_negatives[mention_id]:
                                    mention_hard_negatives[mention_id].append(c)

                    candidates_2 = []
                    # candidates_2.append(label_candidate_id)  # positive candidate
                    # Append hard negative candidates
                    for m_idx, m in enumerate(mentions[document_id]):
                        mention_id = m["mention_id"]
                        if len(mention_hard_negatives[mention_id]) < args.num_candidates:  # args.num_candidates - 1
                            m_hard_candidates = mention_hard_negatives[mention_id]
                        else:
                            candidate_pool = mention_hard_negatives[mention_id]
                            m_hard_candidates = random.sample(candidate_pool, args.num_candidates)  # args.num_candidates - 1
                        candidates_2.append(m_hard_candidates)

            elif args.do_eval:
                if args.use_all_candidates:
                    all_candidate_embedding_path = os.path.join(args.output_dir, 'all_candidate_embeddings.pt')
                    if os.path.exists(all_candidate_embedding_path):
                        all_candidate_embeddings =torch.load(all_candidate_embedding_path)
                    else:
                        all_candidate_embeddings = get_all_candidate_embeddings(args, model, all_entity_token_ids, all_entity_token_masks)
                for m_idx, m in enumerate(mentions[document_id]):
                    m_candidates = []

                    if args.include_positive:
                        m_candidates.append(label_candidate_ids[m_idx])  # positive candidate
                        for c in m["tfidf_candidates"]:
                            if c != label_candidate_ids[m_idx] and len(m_candidates) < args.num_candidates:
                                m_candidates.append(c)
                    elif args.use_tfidf_candidates:
                        for c in m["tfidf_candidates"]:
                            m_candidates.append(c)
                    elif args.use_all_candidates:
                        m_candidates = all_entities

                    candidates.append(m_candidates)

            # Number of mentions in the documents
            num_mentions = len(mentions[document_id])

            if args.use_all_candidates:
                # If all candidates are considered during inference,
                # then place dummy candidate tokens and candidate masks
                candidate_token_ids_1 = None
                candidate_token_masks_1 = None
                candidate_token_ids_2 = None
                candidate_token_masks_2 = None

            else:
                candidate_token_ids_1 = [[tokenizer.pad_token_id] * max_entity_len] * (args.num_max_mentions * args.num_candidates)
                candidate_token_masks_1 = [[0]*max_entity_len] * (args.num_max_mentions * args.num_candidates)
                candidate_token_ids_2 = None
                candidate_token_masks_2 = None

                c_idx = 0
                for m_idx, m_candidates in enumerate(candidates):
                    if m_idx >= args.num_max_mentions:
                        logger.warning("More than {} mentions in doc, mentions after {} are ignored".format(
                                args.num_max_mentions, args.num_max_mentions))
                        break
                    for c in m_candidates:
                        if c in entities:
                            entity_text = entities[c].lower() if args.do_lower_case else entities[c]
                            max_entity_len = max_seq_length // 4  # Number of tokens
                            entity_window = get_entity_window(entity_text, max_entity_len, tokenizer)
                            # [CLS] candidate text [SEP]
                            candidate_tokens = [tokenizer.cls_token] + entity_window + [tokenizer.sep_token]
                            candidate_tokens = tokenizer.convert_tokens_to_ids(candidate_tokens)
                        else:
                            candidate_tokens = [0]*max_entity_len
                        if len(candidate_tokens) > max_entity_len:
                            candidate_tokens = candidate_tokens[:max_entity_len]
                            candidate_masks = [1] * max_entity_len
                        else:
                            candidate_len = len(candidate_tokens)
                            pad_len = max_entity_len - candidate_len
                            candidate_tokens += [tokenizer.pad_token_id] * pad_len
                            candidate_masks = [1] * candidate_len + [0] * pad_len

                        assert len(candidate_tokens) == max_entity_len
                        assert len(candidate_masks) == max_entity_len
                        candidate_token_ids_1[c_idx] = candidate_tokens
                        candidate_token_masks_1[c_idx] = candidate_masks
                        c_idx += 1

                # This second set of candidates is required for Gillick et al. hard negative training
                if candidates_2 is not None:
                    candidate_token_ids_2 = [[tokenizer.pad_token_id] * max_entity_len] * (
                            args.num_max_mentions * args.num_candidates)
                    candidate_token_masks_2 = [[0] * max_entity_len] * (args.num_max_mentions * args.num_candidates)

                    for m_idx, m_hard_candidates in enumerate(candidates_2):
                        if m_idx >= args.num_max_mentions:
                            logger.warning("More than {} mentions in doc, mentions after {} are ignored".format(
                                    args.num_max_mentions, args.num_max_mentions))
                            break
                        c_idx = m_idx * args.num_candidates
                        for c in m_hard_candidates:
                            entity_text = entities[c].lower() if args.do_lower_case else entities[c]
                            max_entity_len = max_seq_length // 4  # Number of tokens
                            entity_window = get_entity_window(entity_text, max_entity_len, tokenizer)
                            # [CLS] candidate text [SEP]
                            candidate_tokens = [tokenizer.cls_token] + entity_window + [tokenizer.sep_token]
                            candidate_tokens = tokenizer.convert_tokens_to_ids(candidate_tokens)
                            if len(candidate_tokens) > max_entity_len:
                                candidate_tokens = candidate_tokens[:max_entity_len]
                                candidate_masks = [1] * max_entity_len
                            else:
                                candidate_len = len(candidate_tokens)
                                pad_len = max_entity_len - candidate_len
                                candidate_tokens += [tokenizer.pad_token_id] * pad_len
                                candidate_masks = [1] * candidate_len + [0] * pad_len

                            assert len(candidate_tokens) == max_entity_len
                            assert len(candidate_masks) == max_entity_len

                            candidate_token_ids_2[c_idx] = candidate_tokens
                            candidate_token_masks_2[c_idx] = candidate_masks
                            c_idx += 1

            # Target candidate
            #logger.info(str(candidates))
            label_ids = [-1] * args.num_max_mentions
            for m_idx, m_candidates in enumerate(candidates):
                if m_idx >= args.num_max_mentions:
                    logger.warning("More than {} mentions in doc, mentions after {} are ignored".format(
                        args.num_max_mentions, args.num_max_mentions))
                    break
                if label_candidate_ids[m_idx] in m_candidates:
                    label_ids[m_idx] = m_candidates.index(label_candidate_ids[m_idx])
                else:
                    label_ids[m_idx] = -100 # when target candidate not in candidate set

            # Pad the mention start and end indices
            mention_start_indices = [0] * args.num_max_mentions
            mention_end_indices = [0] * args.num_max_mentions
            if num_mentions <= args.num_max_mentions:
                mention_start_indices[:num_mentions] = mention_start_markers
                mention_end_indices[:num_mentions] = mention_end_markers
            else:
                mention_start_indices = mention_start_markers[:args.num_max_mentions]
                mention_end_indices = mention_end_markers[:args.num_max_mentions]
                
            assert len(mention_start_indices) == args.num_max_mentions, f"{num_mentions},{mention_start_indices},{args.num_max_mentions}"
            assert len(mention_end_indices) == args.num_max_mentions, f"{mention_end_indices}, {args.num_max_mentions}"
            


            features.append(
                InputFeatures(mention_token_ids=doc_tokens,
                            mention_token_masks=doc_tokens_mask,
                            candidate_token_ids_1=candidate_token_ids_1,
                            candidate_token_masks_1=candidate_token_masks_1,
                            candidate_token_ids_2=candidate_token_ids_2,
                            candidate_token_masks_2=candidate_token_masks_2,
                            label_ids=label_ids,
                            mention_start_indices=mention_start_indices,
                            mention_end_indices=mention_end_indices,
                            num_mentions=num_mentions,
                            seq_tag_ids=seq_tag_ids,
                            )
            )
        logger.info("Saving features into cached file %s", cached_features_file)
        
        #gather across all nodes
        features=[features for node_features in comm.allgather(features) for features in node_features]#FLATTENED
        all_document_ids =[document_ids for node_document_ids in comm.allgather(all_document_ids) for document_ids in node_document_ids]#FLATTENED
        all_label_candidate_ids = [candidate_ids for node_candidate_ids in comm.allgather(all_label_candidate_ids) for candidate_ids in node_candidate_ids]#FLATTENED
        num_longer_docs = comm.allreduce(num_longer_docs,op=MPI.SUM)
        print(num_longer_docs)
        if hvd.rank()==0:
            torch.save(features, cached_features_file)
            np.save(os.path.join(args.data_dir, 'all_entities.npy'),
                        np.array(all_entities))
            np.save(os.path.join(args.data_dir, 'all_entity_token_ids.npy'),
                    np.array(all_entity_token_ids))
            np.save(os.path.join(args.data_dir, 'all_entity_token_masks.npy'),
                    np.array(all_entity_token_masks))
            np.save(os.path.join(args.data_dir, 'all_document_ids.npy'),
                    np.array(all_document_ids))
            np.save(os.path.join(args.data_dir, 'all_label_candidate_ids.npy'),
            np.array(all_label_candidate_ids))
            if args.use_hard_and_random_negatives:

                mention_hard_negatives_list = comm.gather(mention_hard_negatives)
                mention_hard_negatives ={}
                for dictionary in mention_hard_negatives_list:
                    mention_hard_negatives.update(dictionary)
                with open(os.path.join(args.data_dir, 'mention_hard_negatives.json'), 'w+') as f_hn:
                    json.dump(mention_hard_negatives, f_hn)
                f_hn.close()
    else:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        all_document_ids = np.load(os.path.join(args.data_dir, 'all_document_ids.npy'))
        all_label_candidate_ids = np.load(os.path.join(args.data_dir, 'all_label_candidate_ids.npy'))
        all_entities = np.load(os.path.join(args.data_dir, 'all_entities.npy'))
        all_entity_token_ids = np.load(os.path.join(args.data_dir, 'all_entity_token_ids.npy'))
        all_entity_token_masks = np.load(os.path.join(args.data_dir, 'all_entity_token_masks.npy'))
        logger.info("Finished loading features from cached file %s", cached_features_file)
    



    # Convert to Tensors and build dataset
    all_mention_token_ids = torch.tensor([f.mention_token_ids for f in features], dtype=torch.long)
    #reduced dataset size to 10 for debugging
    all_mention_token_masks = torch.tensor([f.mention_token_masks for f in features], dtype=torch.long)
    all_candidate_token_ids_1 = torch.tensor([f.candidate_token_ids_1 if f.candidate_token_ids_1 is not None else [0] for f in features], dtype=torch.long)
    all_candidate_token_masks_1 = torch.tensor([f.candidate_token_masks_1 if f.candidate_token_masks_1 is not None else [0] for f in features], dtype=torch.long)
    all_candidate_token_ids_2 = torch.tensor([f.candidate_token_ids_2 if f.candidate_token_ids_2 is not None else [0] for f in features], dtype=torch.long)
    all_candidate_token_masks_2 = torch.tensor([f.candidate_token_masks_2 if f.candidate_token_masks_2 is not None else [0] for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    #print([len(f.mention_end_indices) for f in features])
    all_mention_start_indices = torch.tensor([f.mention_start_indices for f in features], dtype=torch.long)
    all_mention_end_indices = torch.tensor([f.mention_end_indices for f in features], dtype=torch.long)
    all_num_mentions = torch.tensor([f.num_mentions for f in features], dtype=torch.long)
    all_seq_tag_ids = torch.tensor([f.seq_tag_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_mention_token_ids,
                            all_mention_token_masks,
                            all_candidate_token_ids_1,
                            all_candidate_token_masks_1,
                            all_candidate_token_ids_2,
                            all_candidate_token_masks_2,
                            all_labels,
                            all_mention_start_indices,
                            all_mention_end_indices,
                            all_num_mentions,
                            all_seq_tag_ids,
                            )
    return dataset, (all_entities, all_entity_token_ids, all_entity_token_masks,all_candidate_embeddings), (all_document_ids, all_label_candidate_ids)

def save_checkpoint(args,epoch_num,tokenizer,tokenizer_class,model,optimizer,scheduler,all_candidate_embeddings):
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # Create output directory if needed
    epoch_num = epoch_num + 1 
    run = mlflow.active_run()
    training_run_dir = os.path.join(args.output_dir,f"training_run_{run.info.run_id}".format(args.n_gpu,epoch_num))
    final = args.num_train_epochs == epoch_num
    if final:
        output_dir = os.path.join(training_run_dir, "checkpoint-{}-FINAL".format(epoch_num))
    else:
        output_dir = os.path.join(training_run_dir, "checkpoint-{}".format(epoch_num))

    logger.info("Saving model checkpoint to %s", output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training

    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    torch.save(all_candidate_embeddings, os.path.join(output_dir, "all_candidate_embeddings.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)
    
    # Load a trained model and vocabulary that you have fine-tuned to ensure proper
    if final:
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        tokenizer = tokenizer_class.from_pretrained(output_dir)
        model.to(args.device)
    mlflow.log_artifacts(training_run_dir)    
    logger.info("Saved model checkpoint to %s", output_dir)
    
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        


def get_all_candidate_embeddings(args, model, all_entity_token_ids, all_entity_token_masks):
    single_node_candidate_embeddings = []
    logger.info("INFO: Collecting all candidate embeddings.")
    with torch.no_grad():
        node_entity_token_ids = partition(all_entity_token_ids,hvd.size(),hvd.rank())
        node_entity_token_masks = partition(all_entity_token_masks,hvd.size(),hvd.rank())
        for i, _ in enumerate(node_entity_token_ids):
            #logger.info(all_entity_token_ids.index(node_entity_token_ids[i]))
            entity_tokens = node_entity_token_ids[i]
            entity_tokens_masks = node_entity_token_masks[i]
            candidate_token_ids = torch.LongTensor([entity_tokens]).to(args.device)
            candidate_token_masks = torch.LongTensor([entity_tokens_masks]).to(args.device)
            candidate_outputs = model.bert_candidate.bert(
                    input_ids=candidate_token_ids,
                    attention_mask=candidate_token_masks,
                )
            candidate_embedding = candidate_outputs[1]
            single_node_candidate_embeddings.append(candidate_embedding)
            #logger.info(str(candidate_embedding))
    single_node_candidate_embeddings = torch.cat(single_node_candidate_embeddings, dim=0)
    all_candidate_embeddings = hvd.allgather(single_node_candidate_embeddings)
    logger.info("INFO: Collected all candidate embeddings.")
    print("Tensor size = ", all_candidate_embeddings.size())
    return all_candidate_embeddings

def get_mention_spans(mention_token_ids, predicted_tags, doc_lens,tokenizer):
        b_size = predicted_tags.size(0)
        b_start_indices = []
        b_end_indices = []
        for b_idx in range(b_size):
            tags = predicted_tags[b_idx].cpu().numpy()
            start_indices = []
            end_indices = []
            start_index = 0
            end_index = 0
            # for j in range(doc_lens[b_idx]):
            #     if tags[j] == 1:  # If the token tag is 1, this is the beginning of a mention
            #         start_index = j
            #         end_index = j
            #     elif tags[j] == 2:
            #         if j == 0: # It is the first token (ideally shouldn't be though as it corresponds to the [CLS] token
            #             start_index = j
            #             end_index = j
            #         else:
            #             if tags[j-1] == 1 or tags[j-1] == 2:  # If the previous token is 1, then it's a part of a mention
            #                 end_index += 1
            #             elif tags[j-1] == 0:  # If the previous token is 0, it's the start of a mention (imperfect though)
            #                 start_index = j
            #                 end_index = j
            #     elif tags[j] == 0 and (tags[j-1] == 1 or tags[j-1] == 2): # End of mention
            #         start_indices.append(start_index)
            #         end_indices.append(end_index)
            mention_found = False
            for j in range(1, doc_lens[b_idx] - 1): # Excluding [CLS], [SEP]
                if tags[j] == 1:  # If the token tag is 1, this is the beginning of a mention B
                    start_index = j
                    end_index = j
                    for k in range(j+1, doc_lens[b_idx] - 1):
                        if tokenizer.convert_ids_to_tokens([mention_token_ids[b_idx][k]])[0].startswith('##'):
                            j += 1
                            end_index += 1
                        else:
                            break
                    mention_found = True
                elif tags[j] == 2:
                    if tags[j-1] == 0:  # If the previous token is 0, it's the start of a mention (imperfect though)
                            start_index = j
                            end_index = j
                    else:
                        end_index += 1
                    for k in range(j+1, doc_lens[b_idx] - 1):
                        if tokenizer.convert_ids_to_tokens([mention_token_ids[b_idx][k]])[0].startswith('##'):
                            j += 1
                            end_index += 1
                        else:
                            break
                    mention_found = True
                elif tags[j] == 0 and mention_found: # End of mention
                    start_indices.append(start_index)
                    end_indices.append(end_index)
                    mention_found = False

            # If the last token(s) are a mention
            if mention_found:
                start_indices.append(start_index)
                end_indices.append(end_index)

            b_start_indices.append(start_indices)
            b_end_indices.append(end_indices)
        return b_start_indices, b_end_indices

def find_partially_overlapping_spans(pred_mention_start_indices, pred_mention_end_indices,\
                                        gold_mention_start_indices, gold_mention_end_indices, doc_lens):
    b_size = gold_mention_start_indices.shape[0]
    num_mentions = gold_mention_start_indices.shape[1]

    # Get the Gold mention spans as tuples
    gold_mention_spans = [[(gold_mention_start_indices[b_idx][j], gold_mention_end_indices[b_idx][j]) \
                                        for j in range(num_mentions)]
                            for b_idx in range(b_size)]

    # Get the predicted mention spans as tuples
    predicted_mention_spans = [[] for b_idx in range(b_size)]
    for b_idx in range(b_size):
        num_pred_mentions = len(pred_mention_start_indices[b_idx])
        for j in range(num_pred_mentions):
            predicted_mention_spans[b_idx].append((pred_mention_start_indices[b_idx][j], pred_mention_end_indices[b_idx][j]))

    unmatched_gold_mentions = 0
    extraneous_predicted_mentions = 0
    b_overlapping_start_indices = []
    b_overlapping_end_indices = []
    b_which_gold_spans = []
    for b_idx in range(b_size):
        overlapping_start_indices = []
        overlapping_end_indices = []
        which_gold_spans = []
        p_mention_spans = predicted_mention_spans[b_idx]
        g_mention_spans = gold_mention_spans[b_idx]
        for span_num, (g_s, g_e) in enumerate(g_mention_spans):
            found_overlapping_pred = False
            for (p_s, p_e) in p_mention_spans:
                if p_s >= doc_lens[b_idx]: # If the predicted start index is beyond valid tokens
                    break
                elif g_s <= p_s <= g_e: # The beginning of prediction is within the gold span
                    overlapping_start_indices.append(p_s)
                    if g_e <= p_e:
                        overlapping_end_indices.append(g_e)
                    else:
                        overlapping_end_indices.append(p_e)
                    which_gold_spans.append(span_num)
                    found_overlapping_pred = True
                elif g_s <= p_e <= g_e: # The end of the predicted span is within the gold span
                    if g_s >= p_s:
                        overlapping_start_indices.append(g_s)
                    else:
                        overlapping_start_indices.append(p_s)
                    overlapping_end_indices.append(p_e)
                    which_gold_spans.append(span_num)
                    found_overlapping_pred = True
            if not found_overlapping_pred:
                unmatched_gold_mentions += 1

        for (p_s, p_e) in p_mention_spans:
            if p_s >= doc_lens[b_idx]:  # If the start index is beyond valid tokens
                break
            found_overlapping_pred = False
            for (g_s, g_e) in g_mention_spans:
                if g_s <= p_s <= g_e:  # The beginning of prediction is withing the gold span
                    found_overlapping_pred = True
                elif g_s <= p_e <= g_e:  # The end of the predicted span is within the gold span
                    found_overlapping_pred = True
            if not found_overlapping_pred:
                extraneous_predicted_mentions += 1

        b_overlapping_start_indices.append(overlapping_start_indices)
        b_overlapping_end_indices.append(overlapping_end_indices)
        b_which_gold_spans.append(which_gold_spans)

    return unmatched_gold_mentions, extraneous_predicted_mentions, \
            b_overlapping_start_indices, b_overlapping_end_indices, b_which_gold_spans

def get_marked_mentions(document_id, mentions, docs,  max_seq_length, tokenizer, args):
    # print("Num mention in this doc =", len(mentions[document_id]))
    for m in mentions[document_id]:
        assert m['content_document_id'] == document_id

    context_text = docs[document_id]['text'].lower() if args.do_lower_case else docs[document_id]['text']
    tokenized_text = [tokenizer.cls_token]
    mention_start_markers = []
    mention_end_markers = []
    sequence_tags = []

    # print(len(context_text))
    # print(len(mentions[document_id]))
    prev_end_index = 0
    for m in mentions[document_id]:
        start_index = m['start_index']
        end_index = m['end_index']
        # print(start_index, end_index)
        if start_index >= len(context_text):
            continue
        extracted_mention = context_text[start_index: end_index]

        # Text between the end of last mention and the beginning of current mention
        prefix = context_text[prev_end_index: start_index]
        # Tokenize prefix and add it to the tokenized text
        prefix_tokens = tokenizer.tokenize(prefix)
        tokenized_text += prefix_tokens
        # The sequence tag for prefix tokens is 'O' , 'DNT' --> 'Do Not Tag'
        for j, token in enumerate(prefix_tokens):
            sequence_tags.append('O' if not token.startswith('##') else 'DNT')
        # Add mention start marker to the tokenized text
        mention_start_markers.append(len(tokenized_text))
        # tokenized_text += ['[Ms]']
        # Tokenize the mention and add it to the tokenized text
        mention_tokens = tokenizer.tokenize(extracted_mention)
        tokenized_text += mention_tokens
        # Sequence tags for mention tokens -- first token B, other tokens I
        for j, token in enumerate(mention_tokens):
            if j == 0:
                sequence_tags.append('B')
            else:
                sequence_tags.append('I' if not token.startswith('##') else 'DNT')

        # Add mention end marker to the tokenized text
        mention_end_markers.append(len(tokenized_text) - 1)
        # tokenized_text += ['[Me]']
        # Update prev_end_index
        prev_end_index = end_index

    suffix = context_text[prev_end_index:]
    if len(suffix) > 0:
        suffix_tokens = tokenizer.tokenize(suffix)
        tokenized_text += suffix_tokens
        # The sequence tag for suffix tokens is 'O'
        for j, token in enumerate(suffix_tokens):
            sequence_tags.append('O' if not token.startswith('##') else 'DNT')
    tokenized_text += [tokenizer.sep_token]

    return tokenized_text, mention_start_markers, mention_end_markers, sequence_tags

def get_base_model(args):
    comm = get_comm_magic()
    if hvd.rank()!=0:
        comm.barrier()  # Make sure only the first process in distributed training will download model & vocab
        

    args.base_model_type = args.base_model_type.lower()
    config_class, _, tokenizer_class = MODEL_CLASSES[args.base_model_type]
    config = config_class.from_pretrained(
            args.config_name if args.config_name else args.base_model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.base_model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

    pretrained_bert = PreDualEncoder.from_pretrained(
            args.base_model_name_or_path,
            from_tf=bool(".ckpt" in args.base_model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        # Add new special tokens '[Ms]' and '[Me]' to tag mention
    new_tokens = ['[Ms]', '[Me]']
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    pretrained_bert.resize_token_embeddings(len(tokenizer))

    if hvd.rank()==0:
        comm.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model = DualEncoderBert(config, pretrained_bert)
    return tokenizer_class,tokenizer,model  

def get_trained_model(args):
    tokenizer_class,tokenizer,model = get_base_model(args)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model.bin')))
    tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    return tokenizer,model

def partition(a, n, i):
    #a=list
    #n=number_of_partitions
    #i=partition_number
    k, m = divmod(len(a), n)
    return a[i*k+min(i, m):(i+1)*k+min(i+1, m)]