import sys

from stratified.clustering_alg import ClusteringAlg, KMeansAlg

from typing import List, Optional, Callable
from numpy import ndarray, asarray
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class ClusteringMetric():
    name: str
    metric_function: Callable[[ndarray, List[List[int]], Optional], float]
    metric_function_kwargs: dict
    default_val: float

    def compute(self, clusterer: ClusteringAlg):
        embedding_array = []
        labels = []
        for id in range(len(clusterer.label_to_id)):
            for i, part in enumerate(clusterer.partition):
                if id in part:
                    embedding_array.append(clusterer.embedding_dict[clusterer.vertices[id]])
                    labels.append(i)

        embedding_array = asarray(embedding_array)
        if len(clusterer.partition) < 2 or len(clusterer.partition) >= len(labels):
            return self.default_val
        return self.metric_function(embedding_array, labels,
                                    **self.metric_function_kwargs)

class SilhouetteCoefficient(ClusteringMetric):
    def __init__(self):
        self.name = "Silhouette Coefficient"
        self.metric_function = silhouette_score
        self.metric_function_kwargs = {"metric": "cosine"}
        self.default_val = 0

class DaviesBouldinIndex(ClusteringMetric):
    def __init__(self):
        self.name = "Davies-Bouldin Index"
        self.metric_function = davies_bouldin_score
        self.metric_function_kwargs = {}
        self.default_val = None

class CalinskiHarabaszIndex(ClusteringMetric):
    def __init__(self):
        self.name = "Calinski-Harabasz Index"
        self.metric_function = calinski_harabasz_score
        self.metric_function_kwargs = {}
        self.default_val = 0


def tune_num_clusters(clusterer_kwargs: dict,
                      num_clusters_list: Optional[List[int]] = None,
                      max_cluster_fact: float = 0.2,
                      max_cluster_num: int = 50,
                      early_stop: bool = False):

    if num_clusters_list is None:
        num_clusters_list = list(range(15, max_cluster_num+1, 5))

    metric_dict = {
        "silhouette": SilhouetteCoefficient(),
        "davies_bouldin": DaviesBouldinIndex(),
        "calinski_harabasz": CalinskiHarabaszIndex(),
    }
    results = {
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": [],
    }

    gamma_vals = {
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": []
    }

    max_cluster = []

    pars = clusterer_kwargs.copy()
    clusterer = KMeansAlg(**pars)
    if clusterer.valid:
        if early_stop:
            return 0
        for num_clusters in num_clusters_list:
            print("Num_clusters:", num_clusters)
            clusterer.kmeans.n_clusters = num_clusters
            clusterer.fit()
            if len(clusterer.partition) <= max_cluster_num and \
                    all([len(part) <= round(max_cluster_fact*len(clusterer.vertices)) for part in clusterer.partition]):
                for metric_name in metric_dict:
                    metric = metric_dict[metric_name]
                    res = metric.compute(clusterer)
                    if res is not None:
                        results[metric_name].append(res)
                        gamma_vals[metric_name].append(num_clusters)

            max_cluster.append(max([len(part) for part in clusterer.partition]))

        if clusterer_kwargs["pathogenic"]:
            fig_add = "_pathogenic_"
        else:
            fig_add = "_neutral_"

        for key in results:
            plt.plot(gamma_vals[key], results[key])
        plt.xticks(num_clusters_list)
        plt.xlabel("Number of clusters")
        plt.legend(list(results.keys()))
        plt.tight_layout()
        plt.savefig(clusterer_kwargs["model"].embedding_model.create_embedding_folder()+"/kmeans_num_clusters_tuning"+fig_add+".png")
        plt.clf()


        plt.plot(num_clusters_list, max_cluster)
        plt.xticks(num_clusters_list)
        plt.xlabel("Number of clusters")
        plt.ylabel("Max cluster size")
        plt.tight_layout()
        plt.savefig(clusterer_kwargs["model"].embedding_model.create_embedding_folder() + "/kmeans_num_clusters_max_cluster"+fig_add+".png")
        plt.clf()

        max_cluster_ch = None
        max_ch_val = -sys.maxsize
        for cluster_size, ch_val in zip(gamma_vals["calinski_harabasz"], results["calinski_harabasz"]):
            if ch_val > max_ch_val:
                max_ch_val = ch_val
                max_cluster_ch = cluster_size

        return max_cluster_ch

    else:
        return None