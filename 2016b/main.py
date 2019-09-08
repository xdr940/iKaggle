import pandas as pd
import pandas_profiling
from path import Path
import numpy as np
from scipy.stats import chi2_contingency
from collections import Counter

root = Path('/home/roit/datasets/kaggle/2016b')


dump_path = root/'dump'

ge_info = root/'gene_info'
exitnpy = False
if exitnpy==False:
    genes_dic = []
    genes = []
    snps_sorted = pd.read_csv(dump_path/'sorted_cols_series.csv')
    cnt=0
    for file in ge_info.files():
        gene = open(file).read()#str: rs1\n rs2\n...
        ls = gene.split('\n')
        ls.pop()
        cnt+=len(ls)
        genes_dic+=ls
        genes.append(ls)

    print(cnt)


    vecs = np.zeros([len(genes),len(snps_sorted)])

    for i in range(len(genes)):
        for j in range(len(genes[i])):
            col = genes_dic.index(genes[i][j])
            vecs[i][col] = 1
    np.save('vecs.npy',vecs)

else:
    vecs = np.load('vecs.npy')

sum_vec =vecs.sum(axis=0)#行相加

print(sum_vec.sum())