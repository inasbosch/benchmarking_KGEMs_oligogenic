import random
import sys

from config.embedding_source_paths import DATASET_PATH
from node_model.models.model import Model

import pickle as pck
import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from itertools import combinations
from copy import deepcopy
from numpy import inner as numpy_inner
from numpy import isrealobj, conj, ndarray, asarray, mean
import numpy as np
from numpy.linalg import norm
from typing import List, Tuple, Optional, Dict
from math import pi, cos
from random import seed, sample
from sklearn import decomposition
from sklearn.cluster import KMeans, MiniBatchKMeans

colors = []
cmap = plt.get_cmap('Set1')
for c in range(cmap.N):
    if to_hex(cmap(c)) not in colors:
        colors.append(to_hex(cmap(c)))
cmap = plt.get_cmap('Set3')
for c in range(cmap.N):
    if to_hex(cmap(c)) not in colors:
        colors.append(to_hex(cmap(c)))
cmap = plt.get_cmap('tab10')
for c in range(cmap.N):
    if to_hex(cmap(c)) not in colors:
        colors.append(to_hex(cmap(c)))

def cosine_similarity(a: ndarray, b: ndarray):
    norm_a = norm(a)
    norm_b = norm(b)
    return inner(a,b)/(norm_a*norm_b)

def inner(a: ndarray, b: ndarray):
   if isrealobj(a):
       return numpy_inner(a,b)
   else:
       return real(numpy_inner(a, conj(b)))


class ClusteringAlg():
    method_key: str

    def __init__(self, model: Model, pathogenic: bool = True):

        model = deepcopy(model)
        model.stratified = False
        model.holdout = False
        self.model = model
        self.pathogenic = pathogenic
        self.recombined = False

        seed(10)

        gene_data = self.model.read_gene_set()

        if pathogenic:
            output_val = 1
        else:
            output_val = 0

        self.vertices = []
        for row in gene_data:
            if int(row[2]) == output_val:
                sorted_row = sorted([row[0], row[1]])
                self.vertices.append(sorted_row[0] + "," + sorted_row[1])

        if self.pathogenic:
            with open(DATASET_PATH + "Datasets/holdout_separate.tsv", "r") as f:
                for row in f:
                    split_row = row.split("\t")
                    split_row[1] = split_row[1][:-1]
                    split_row = sorted(split_row)
                    self.vertices.append(split_row[0] + "," + split_row[1])

        self.partition = None

        embeddings = self.model.embedding_model.load_embeddings()
        label_to_id = self.model.embedding_model.load_label_to_id()

        self.embedding_dict = {}
        for pair_str in self.vertices:
            pair = pair_str.split(",")
            if pair[0] in label_to_id and pair[1] in label_to_id:
                emb1 = embeddings[label_to_id[pair[0]]]
                emb2 = embeddings[label_to_id[pair[1]]]
                self.embedding_dict[pair_str] = model.embedding_transform.compute(emb1, emb2)[0]

        to_remove = []
        for pair in self.vertices:
            if pair not in self.embedding_dict:
                to_remove.append(pair)
        for pair in to_remove:
            self.vertices.remove(pair)

        self.holdout_pairs = self.compute_holdout()

        n = len(self.vertices)
        self.label_to_id = {self.vertices[i]: i for i in range(n)}

        if n <= 250:
            self.valid = False
        else:
            self.valid = True

    def compute_holdout(self):
        holdout_pairs = []

        if self.pathogenic:
            file = open(DATASET_PATH + "Datasets/holdout_separate.tsv", "r")
        else:
            if self.num_neutral_holdout != 1:
                num_str = str(self.num_neutral_holdout)
            else:
                num_str = ""
            file = open(DATASET_PATH + "Datasets/holdout_neutral_separate"+num_str+".tsv", "r")

        for row in file:
            split_row = row.split("\t")
            split_row[1] = split_row[1][:-1]
            split_row = sorted(split_row)
            holdout_pairs.append(split_row[0] + "," + split_row[1])

        to_remove = []
        for pair in holdout_pairs:
            split_pair = pair.split(",")
            for pair2 in self.vertices:
                split_pair2 = pair2.split(",")
                if split_pair[0] in split_pair2 or split_pair[1] in split_pair2:
                    if pair2 not in to_remove:
                        to_remove.append(pair2)

        for pair in to_remove:
            self.vertices.remove(pair)

        if self.model.max_sim_val != 1 / 9:
            self.max_sim_str = str(round(self.model.max_sim_val, 2)) + "_"
        else:
            self.max_sim_str = ""

        print("Initial number of gene pairs:", len(self.vertices))
        to_remove = []
        for pair in self.vertices:
            for pair2 in holdout_pairs:
                temp_sim = cosine_similarity(
                    self.embedding_dict[pair],
                    self.embedding_dict[pair2]
                )

                if temp_sim > cos(pi * self.model.max_sim_val):
                    to_remove.append(pair)
                    break

        for pair in to_remove:
            self.vertices.remove(pair)

        print("Number of gene pairs:", len(self.vertices))

        return holdout_pairs


    def pca(self, removed: Optional[List[int]] = None):
        if removed is None:
            removed = []

        entity_array = []

        for cluster in self.partition:
            for pair in cluster:
                entity_array.append(self.embedding_dict[self.vertices[pair]])

        for pair in self.holdout_pairs:
            entity_array.append(self.embedding_dict[pair])

        entity_array = asarray(entity_array)

        PCA_res = decomposition.PCA(n_components=2)
        entity_2d = PCA_res.fit_transform(entity_array)

        PCA_res = decomposition.PCA(n_components=3)
        entity_3d = PCA_res.fit_transform(entity_array)

        color_list = []
        for i, cluster in enumerate(self.partition):
            for pair in cluster:
                if pair in removed:
                    color_list.append("#ffffff")
                else:
                    if i >= len(colors):
                        color_list.append("grey")
                    else:
                        color_list.append(colors[i])

        for _ in self.holdout_pairs:
            color_list.append("black")

        plt.scatter(entity_2d[:, 0], entity_2d[:, 1], color=color_list)
        plt.show()


        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(entity_3d[:, 0], entity_3d[:, 1], entity_3d[:,2], color=color_list)
        plt.show()

    def fit(self):
        pass

    def cosine_part(self, part: List[int]):
        cosine_sims = {}
        for double_pairs in combinations(part, 2):
            pair1 = self.vertices[double_pairs[0]]
            pair2 = self.vertices[double_pairs[1]]
            emb1 = self.embedding_dict[pair1]
            emb2 = self.embedding_dict[pair2]
            cosine_sims[(double_pairs[0], double_pairs[1])] = cosine_similarity(emb1, emb2)

        return cosine_sims

    def max_similarity(self, cosine_sims: Dict[Tuple[str, str], float]):
        all_sims = {}

        for pair in cosine_sims:
            pair1, pair2 = pair
            if pair1 not in all_sims:
                all_sims[pair1] = [cosine_sims[pair]]
            else:
                all_sims[pair1].append(cosine_sims[pair])

            if pair2 not in all_sims:
                all_sims[pair2] = [cosine_sims[pair]]
            else:
                all_sims[pair2].append(cosine_sims[pair])

        avg_sims = {pair: mean(all_sims[pair]) for pair in all_sims}
        return max(avg_sims.keys(), key=lambda x: avg_sims[x])

    def undersampling(self, mu: float = 0.4, plot: bool = False):
        self.num_pairs = sum([len(part) for part in self.partition])
        new_partition = []
        removed = []

        for part in tqdm.tqdm(self.partition,desc="Undersampling"):
            cosine_sims = self.cosine_part(part)
            while len(part) > (self.num_pairs / 10) * (1 + mu):
                max_cos = max(cosine_sims, key=lambda x: cosine_sims[x])
                to_remove = sample(max_cos,1)[0]

                part.remove(to_remove)
                removed.append(to_remove)
                self.num_pairs -= 1

                remove_list = []
                for key in cosine_sims:
                    if to_remove in key:
                        remove_list.append(key)

                for key in remove_list:
                    del cosine_sims[key]

            new_partition.append(part)

        self.under_partition = new_partition
        print([len(part) for part in self.under_partition])
        if plot:
            self.pca(removed)

    def recombine_bins(self, mu: float = 0.0):
        recombined_final = [[] for _ in range(10)]
        sorted_final = sorted(self.under_partition, key=lambda x: len(x), reverse=True)
        not_added_bins = [bin for bin in self.under_partition]

        for i, bin in enumerate(sorted_final[:10]):
            recombined_final[i].extend(bin)
            not_added_bins.remove(bin)

        num_added = 0
        for bin in not_added_bins:
            print(num_added)
            cos_vals = []

            for i in range(len(recombined_final)):
                avg_cos = 0
                for pair1 in bin:
                    for pair2 in recombined_final[i]:
                        temp_cos = cosine_similarity(self.embedding_dict[self.vertices[pair1]],
                                                         self.embedding_dict[self.vertices[pair2]])
                        avg_cos += temp_cos

                avg_cos /= len(bin)*len(recombined_final[i])
                cos_vals.append(avg_cos)

            pos_spots = [(val, i) for i, val in zip(range(len(recombined_final)), cos_vals) if
                            len(bin) + len(recombined_final[i]) < self.num_pairs/10*(1+mu)]
            neg_spots = [(val, i) for i, val in zip(range(len(recombined_final)), cos_vals) if not
                            len(bin) + len(recombined_final[i]) < self.num_pairs/10*(1+mu)]

            if len(pos_spots) > 0:
                max_pos_test = max(pos_spots, key=lambda x: x[0])
            else:
                max_pos_test = None

            if len(neg_spots) > 0:
                max_neg_test = max(neg_spots, key=lambda x: x[0])
            else:
                max_neg_test = None

            if max_pos_test is not None:
                num_added += 1
                recombined_final[max_pos_test[1]].extend(bin)
            else:
                num_added += 1
                recombined_final[max_neg_test[1]].extend(bin)

        self.recombined_partition = recombined_final

        self.readable_partition = []
        for part in self.recombined_partition:
            self.readable_partition.append([tuple(self.vertices[id].split(",")) for id in part])
        self.recombined = True

    def save_clusters(self):
        if self.pathogenic:
            str_add = "_" + self.max_sim_str + "pathogenic_"
        else:
            str_add =  "_" + self.max_sim_str
            if self.num_neutral_holdout != 1:
                str_add += "num_"+str(self.num_neutral_holdout) + "_"

            str_add += "neutral_"


        with open(self.model.embedding_model.create_embedding_folder()+"/"+
                    str(self.model.embedding_transform.__name__) + "/"
                  + self.method_key + str_add + "clusterer.bin", "wb") as f:
            pck.dump(self, f)

        with open(self.model.embedding_model.create_embedding_folder()+"/"
                  + self.method_key + str_add + "holdout.txt", "w") as f:

            for pair_str in self.holdout_pairs:
                pair = pair_str.split(",")
                f.write(pair[0] + "\t" + pair[1] + "\n")

        if self.recombined:
            with open(self.model.embedding_model.create_embedding_folder()+"/"
                      + self.method_key + str_add + "partition.bin", "wb") as f:
                pck.dump(self.readable_partition, f)

class KMeansAlg(ClusteringAlg):
    def __init__(self, model: Model, pathogenic: bool = True,
                 num_clusters: int = 10, mini_batch: bool = False):

        super().__init__(model=model, pathogenic=pathogenic)
        if mini_batch:
            self.kmeans = MiniBatchKMeans(num_clusters, random_state=10)
        else:
            self.kmeans = KMeans(num_clusters, random_state=10)

        self.method_key = "kmeans"

    def normalise(self, embedding: ndarray):
        normal = (embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding))
        return normal

    def fit(self):
        embedding_array = []
        for pair in self.vertices:
            if pair in self.holdout_pairs:
                continue
            embedding_array.append(self.normalise(self.embedding_dict[pair]))

        self.kmeans.fit(asarray(embedding_array))
        labels = self.kmeans.labels_
        self.partition = []

        for lab, pair in zip(labels, self.vertices):
            if lab + 1 < len(self.partition):
                self.partition[lab + 1].append(self.label_to_id[pair])
            else:
                self.partition.append([self.label_to_id[pair]])




def compare_partitions(part1: List[List[str]], part2: List[List[str]]):
    for i, bin1 in enumerate(part1):
        for j, bin2 in enumerate(part2):
            total_num = 0
            for pair in bin1:
                if pair in bin2:
                    total_num += 1
            value = str(round(total_num/len(bin1)*100, 2))
            if len(value) < 4:
                value += "0"
            print(value + "\t", end="")

        print("\n", end="")

    print("\n"+str([len(part) for part in part1]))
