from dataclasses import asdict
from el_toolkit.document import Document,Mention
import re
from transformers import BertTokenizer
import json

biomed_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='monologg/biobert_v1.1_pubmed',
                                            do_lower_case=False, cache_dir=None)

Document(message="This is a good example of a test sentence.",doc_id="1234",mentions = [Mention(**{"start_index":0,"end_index":4,"concept_id":"C222222"}),
                                                                                        Mention(**{"start_index":15,"end_index":22,"concept_id":"C123456"}),
                                                                                        Mention(**{"start_index":27,"end_index":41,"concept_id":"C11111"})
                                                                                                ]
).write_json(f"tests/test_data/test_docs/no_overlaps_3_mentions/doc_data.json")


Document(message="This is a good example of a test sentence.",doc_id="1234",mentions = [Mention(**{"start_index":0,"end_index":4,"concept_id":"C222222"}),
                                                                                        Mention(**{"start_index":15,"end_index":22,"concept_id":"C123456"}),
                                                                                        Mention(**{"start_index":15,"end_index":41,"concept_id":"C11111"})
                                                                                                ]
).write_json(f"tests/test_data/doc_test_data/overlaps/doc_data.json")

doc = Document.read_json(f"tests/test_data/test_docs/no_overlaps_3_mentions/doc_data.json")

regex = re.compile('^\d+\|[a|t]\|')

def generate_bc5cdr_docs():
        docs = []
        mentions = []
        for partition_name in ["train","dev","test"]:
                with open(f"/Users/coltonflowers/Work/TREND_repos/BioMedical-EL/data/BC5CDR/raw_data/{partition_name}_corpus.txt") as f:
                        for line in f:
                                line = line.strip()
                                if regex.match(line):
                                        match_span = regex.match(line).span()
                                        start_span_idx = match_span[0]
                                        end_span_idx = match_span[1]
                                        document_id = line[start_span_idx:end_span_idx].split("|")[0]
                                        if line[start_span_idx:end_span_idx].split("|")[1] == "t":
                                                text = line[end_span_idx:]
                                        else:
                                                text = text + ' ' + line[end_span_idx:]  # Abstract is added
                                        
                                elif line:
                                        cols = line.strip().split('\t')
                                        if len(cols) == 6:
                                                if cols[5] == '-1':
                                                        continue
                                                document_id = cols[0]
                                                start_index = int(cols[1])
                                                end_index = int(cols[2])

                                                assert start_index >= 0
                                                assert end_index >= 0
                                                assert start_index <= end_index
                                                concept_id = cols[5]
                                                if '+' in concept_id:  # some concepts are combination of two concepts and hence two ids are concatenated by + or |
                                                        concept_id = concept_id.split('+')[0].strip()
                                                elif '|' in cols[5]:
                                                        concept_id = concept_id.split('|')[0].strip()

                                                mentions.append(Mention(start_index,end_index,concept_id))
                                else:
                                        docs.append(Document(document_id,text,mentions))
                                        text = ""
                                        mentions = []
        with open(f"./datasets/bc5cdr/documents/original_docs.json","w+") as f:
                json.dump([asdict(doc) for doc in docs],f)
        docs[0].write_json(f"./tests/test_data/doc_test_data/no_overlaps_hard/doc_data.json")
        docs[-1].write_json(f"./tests/test_data/doc_test_data/no_overlaps_32_mentions/doc_data.json")
        docs[-4].write_json(f"./tests/test_data/doc_test_data/no_overlaps_14_mentions/doc_data.json")

generate_bc5cdr_docs()