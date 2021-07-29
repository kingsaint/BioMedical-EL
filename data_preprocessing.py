import os
import json
import argparse
from transformers import BertTokenizer, BertModel
import copy
import re

# with open('./data/NCBI_Disease/mentions_candidates_tfidf.json') as f:
#     mentions_candidates_tfidf = json.load(f)


def preprocess_data(data_dir):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='./biobert_v1.1_pubmed',
                                              do_lower_case=False, cache_dir=None)
    print(len(tokenizer))

    regex = re.compile('^\d+\|[a|t]\|')

    raw_data_dir = os.path.join(data_dir, "raw_data")
    save_dir = os.path.join(data_dir, "processed_data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(raw_data_dir)
    print(save_dir)
    input_files = os.listdir(raw_data_dir)
    print(input_files)
    for file_name in input_files:
        documents = dict()
        mentions = dict()
        print(os.path.join(raw_data_dir, file_name))
        with open(os.path.join(raw_data_dir, file_name), encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if regex.match(line):
                    match_span = regex.match(line).span()
                    start_span_idx = match_span[0]
                    end_span_idx = match_span[1]
                    document_id = line[start_span_idx:end_span_idx].split("|")[0]
                    text = line[end_span_idx:]
                    if document_id not in documents:
                        documents[document_id] = text  # Title is added
                    else:
                        documents[document_id] = documents[document_id] + ' ' + text  # Abstract is added
                    print("Document id = {}".format(document_id))
                else:
                    cols = line.strip().split('\t')
                    if len(cols) == 6:
                        if cols[5] == '-1':
                            continue
                        document_id = cols[0]
                        if document_id not in mentions:
                            mentions[document_id] = []
                        mention_id = document_id + "_" + str(len(mentions[document_id]) + 1)
                        start_index = int(cols[1])
                        end_index = int(cols[2])

                        assert start_index >= 0
                        assert end_index >= 0
                        assert start_index <= end_index

                        mention_text = cols[3]

                        mention_type = cols[4]
                        candidate_id = cols[5]
                        if '+' in candidate_id:  # some concepts are combination of two concepts and hence two ids are concatenated by + or |
                            candidate_id = candidate_id.split('+')[0].strip()
                        elif '|' in cols[5]:
                            candidate_id = candidate_id.split('|')[0].strip()

                        tfidf_candidates = []
                        # for c in mentions_candidates_tfidf[document_id][mention_id]["all_candidates"]:
                        #     tfidf_candidates.append(c["candidate_id"])

                        mentions[document_id].append({"mention_id": mention_id,
                                                 "start_index": start_index,
                                                 "end_index": end_index,
                                                 "text": mention_text,
                                                 "type": mention_type,
                                                 "content_document_id": document_id,
                                                 "label_candidate_id": candidate_id}
                                                    ) # "tfidf_candidates": tfidf_candidates

                    else:  # Empty lines
                        continue
        print("Segmentation starts ...")

        all_documents = copy.deepcopy(documents)
        omitted_mentions = 0

        for document_id in all_documents:
            print("Doc ==>", document_id, "Doc length ==>", len(documents[document_id]))
            start_index_new_doc = 0
            end_index_new_doc = 0
            segment_id = 0
            segment_text = ""
            num_mentions = len(mentions[document_id])
            print("Num mentions ==>", num_mentions)
            cumulative_seg_len = [0]
            max_mention_per_new_doc = 8
            num_mentions_new_doc = 0

            for i in range(num_mentions):
                end_index_new_doc = max(end_index_new_doc, mentions[document_id][i]["end_index"])
                tentative_segment_text = segment_text + documents[document_id][start_index_new_doc:end_index_new_doc]
                tokens = tokenizer.tokenize(tentative_segment_text)
                if num_mentions_new_doc != max_mention_per_new_doc and len(['[CLS]'] + tokens + ['[SEP]']) < 256:
                    num_mentions_new_doc += 1
                    # print(num_mentions_new_doc)
                    segment_text = segment_text + documents[document_id][start_index_new_doc:end_index_new_doc]
                    start_index_new_doc = end_index_new_doc
                    # Add the mention to `mentions`
                    new_document_id = document_id + '_' + str(segment_id)
                    if new_document_id not in mentions:
                        mentions[new_document_id] = []
                    mention_id = new_document_id + '_' + str(i % max_mention_per_new_doc)
                    new_mention = copy.deepcopy(mentions[document_id][i])
                    new_mention["mention_id"] = mention_id
                    new_mention["content_document_id"] = new_document_id
                    new_mention["start_index"] = new_mention["start_index"] - cumulative_seg_len[segment_id]
                    new_mention["end_index"] = new_mention["end_index"] - cumulative_seg_len[segment_id]
                    if new_mention["start_index"] < new_mention["end_index"] and  new_mention["start_index"] >= 0 and new_mention["end_index"] > 0:
                        mentions[new_document_id].append(new_mention)
                    else:
                        omitted_mentions += 1
                    continue
                else:
                    # Write the new segment
                    new_document_id = document_id + '_' + str(segment_id)
                    if new_document_id not in mentions:
                        mentions[new_document_id] = []
                    documents[new_document_id] = segment_text
                    cumulative_seg_len.append(cumulative_seg_len[-1] + len(segment_text))
                    print("New doc id ==> {}, # mentions ==> {}".format(new_document_id, len(mentions[new_document_id])))

                    # Reset everything
                    num_mentions_new_doc = 0
                    segment_text = ""

                    # Increment segment number
                    segment_id += 1

                    # Take care of the current mention for which if condition returned False
                    num_mentions_new_doc += 1
                    segment_text = segment_text + documents[document_id][start_index_new_doc:end_index_new_doc]
                    start_index_new_doc = end_index_new_doc

                    # Add the mention to `mentions`
                    new_document_id = document_id + '_' + str(segment_id)
                    if new_document_id not in mentions:
                        mentions[new_document_id] = []
                    mention_id = new_document_id + '_' + str(i % max_mention_per_new_doc)
                    new_mention = copy.deepcopy(mentions[document_id][i])
                    new_mention["mention_id"] = mention_id
                    new_mention["content_document_id"] = new_document_id

                    new_mention["start_index"] = new_mention["start_index"] - cumulative_seg_len[segment_id]
                    new_mention["end_index"] = new_mention["end_index"] - cumulative_seg_len[segment_id]

                    if new_mention["start_index"] < new_mention["end_index"] and new_mention["start_index"] >= 0 and new_mention["end_index"] > 0:
                        mentions[new_document_id].append(new_mention)
                    else:
                        omitted_mentions += 1

            # Last few mentions and the remaining text
            segment_text = segment_text + documents[document_id][start_index_new_doc:]
            new_document_id = document_id + '_' + str(segment_id)
            documents[new_document_id] = segment_text
            cumulative_seg_len.append(cumulative_seg_len[-1] + len(segment_text))

            # Delete the original document from `documents`
            del documents[document_id]
            # Delete the original mentions from `mentions`
            del mentions[document_id]
        print("Number of omitted mentions ==>", omitted_mentions)

        if 'train' in file_name:
            split = 'train'
        elif 'test' in file_name:
            split = 'test'
        else:
            split = 'dev'

        if not os.path.exists(os.path.join(save_dir, split, "documents")):
            os.makedirs(os.path.join(save_dir, split, "documents"))
        with open(os.path.join(save_dir, split, "documents/documents.json"), 'w+') as f:
            for document_id in documents:
                dict_to_write = {"document_id": document_id, "text": documents[document_id]}
                dict_to_write = json.dumps(dict_to_write)
                f.write(dict_to_write + '\n')
        f.close()

        if not os.path.exists(os.path.join(save_dir, split, "mentions")):
            os.makedirs(os.path.join(save_dir, split, "mentions"))
        with open(os.path.join(save_dir, split, "mentions/mentions.json"), 'w+') as f:
            for document_id in mentions:
                dict_to_write = json.dumps(mentions[document_id])
                f.write(dict_to_write + '\n')
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True, type=str, help="Path to data dir"
    )

    args = parser.parse_args()

    preprocess_data(args.data_dir)