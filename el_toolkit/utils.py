def partition(a, n, i):
    #a=list
    #n=number_of_partitions
    #i=partition_number
    k, m = divmod(len(a), n)
    
def truncate(token_ids,pad_token_id,max_seq_len):
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]
        tokens_mask = [1] * max_seq_len
    else:
        token_number = len(token_ids)
        pad_len = max_seq_len - token_number
        token_ids += [pad_token_id] * pad_len
        tokens_mask = [1] * token_number + [0] * pad_len
    return token_ids,tokens_mask