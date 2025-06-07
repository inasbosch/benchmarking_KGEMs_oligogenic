from numpy import ndarray, asarray, dtype
from typing import Any, Optional
import pickle as pck

from sklearn.model_selection import StratifiedKFold
from config.embedding_source_paths import ROOT, DATASET_PATH

class CrossFold():
    def __init__(self, stratified: bool = False,
                 k: int = 10, random_seed: int = 10,
                 file_path: Optional[str] = None):

        self.k = k

        if stratified:
            self.splits = GenePairFold(file_path)
        else:
            self.splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)

    def split(self, input: ndarray[Any, dtype[Any]], output: ndarray[Any, dtype[Any]]) -> enumerate[Any]:
       return enumerate(self.splits.split(input,output))



class GenePairFold():
    def __init__(self, file_path: Optional[str] = None):

        self.file_path = file_path + "/kmeans_"

    def split(self, input: ndarray[Any, dtype[Any]], output: Optional[ndarray[Any, dtype[Any]]] = None) \
            -> ndarray[Any, dtype[Any]]:

        hard = False
        f = open(DATASET_PATH + "/Datasets/neutrals_1KGP_100x.tsv", "r")
        for pair, out in zip(input, output):
            if int(out) == 0:
                found = False
                for row in f:
                    if "gene" in row:
                        continue
                    gene1 = row.split("\t")[0]
                    gene2 = row.split("\t")[1]

                    if (gene1 == pair[0] and gene2 == pair[1]) or \
                        (gene1 == pair[1] and gene2 == pair[0]):
                        found = True
                        break

                if not found:
                    hard = True
                    break


        if hard:
            neutral_add = "hard_"
        else:
            neutral_add = ""

        with open(self.file_path+"pathogenic_partition.bin","rb") as f:
            positive_recombined = pck.load(f)
        with open(self.file_path+neutral_add+"neutral_partition.bin","rb") as f:
            neutral_recombined = pck.load(f)

        index_split = [[[], []] for _ in range(10)]

        for j, pair in enumerate(input):
            for i in range(10):
                pos = positive_recombined[i]
                neg = neutral_recombined[i]
                if (((pair[0], pair[1]) in pos) or ((pair[1], pair[0]) in pos)) \
                        or (((pair[0], pair[1]) in neg) or ((pair[1], pair[0]) in neg)):
                    index_split[i][1].append(j)
                else:
                    index_split[i][0].append(j)

        index_split = [[asarray(index_split[i][0]),asarray(index_split[i][1])] for i in range(10)]
        return index_split
