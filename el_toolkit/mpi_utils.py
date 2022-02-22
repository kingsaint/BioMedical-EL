def partition(a, n, i):
    #a=list
    #n=number_of_partitions
    #i=partition_number
    k, m = divmod(len(a), n)
    return a[i*k+min(i, m):(i+1)*k+min(i+1, m)]