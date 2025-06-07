import datetime
import os
import pickle as pck
from math import nan
import pandas as pd
from typing import Optional, Union, List, Tuple

import networkx as nx
import numpy
import pandas
from numpy import asarray, ndarray
from sklearn.ensemble import RandomForestClassifier

from config.embedding_source_paths import DIGEPRED_PATH, DATASET_PATH
from node_model.cross_fold import CrossFold
from node_model.models.model import Model
from node_model.results import KFoldResults, SingleResults
from utils.operators import IdentityOperator

'''An implementation of DiGePred based on the github https://github.com/CapraLab/DiGePred,
the github associated to the article

Mukherjee, S., Cogan, J.D., Newman, J.H., Phillips, J.A., Hamid, R., Meiler, J., & Capra, J.A. (2021).
Identifying digenic disease genes via machine learning in the Undiagnosed Diseases Network. 
American journal of human genetics.'''


class DiGePred(Model):
    def __init__(self, running_mode: bool = True, holdout: bool = True):

        super().__init__(stratified=False, holdout=holdout)

        self.embedding_transform = IdentityOperator()
        if running_mode:
            now = datetime.datetime.now()
            self.month = str(now.strftime("%m"))
            self.day = str(now.strftime("%d"))
            self.year = str(now.strftime("%Y"))
            self.hour = str(now.strftime("%H"))
            self.minute = str(now.strftime("%M"))

            print("Time initiated")

            ## Load pathway data files
            self.reactome_gene_to_path_codes = pck.load(
                open(DIGEPRED_PATH+'/data/pathways/reactome/reactome_gene_to_path_codes.bin', 'rb'))
            self.kegg_gene_to_path_codes = pck.load(open(DIGEPRED_PATH+'/data/pathways/kegg/kegg_gene_to_path_codes.txt', 'rb'))
            print("Pathways initiated")

            ## Load phenotype data files
            self.hpo_gene_to_code = pck.load(open(DIGEPRED_PATH+'/data/phenotypes/hpo/hpo_gene_to_code.txt', 'rb'))
            print("Phenotype data initiated")

            ## Load co-expression data files
            self.coexpress_dict = pck.load(open(DIGEPRED_PATH+'/data/coex/mutual_co-expression_rank_dict.txt', 'rb'))
            print("Coexpression data initiated")

            ## Load network data files
            with open(DIGEPRED_PATH+"/data/networks/UCSC_ppi_network_new.bin","rb") as f:
                self.G_ppi = pck.load(f)
            print("PPI network initiated")
            with open(DIGEPRED_PATH+"/data/networks/UCSC_pwy_network_new.bin","rb") as f:
                self.G_pwy = pck.load(f)
            print("PWY network initiated")
            with open(DIGEPRED_PATH+"/data/networks/UCSC_txt_network_new.bin","rb") as f:
                self.G_txt = pck.load(f)
            print("TXT networks initiated")

            self.dists_ppi = pck.load(open(DIGEPRED_PATH+'/data/networks/PPI_network_all_pairs_shortest_paths_Feb21_19.pkl', 'rb'))
            print("PPI distance initiated")
            self.dists_pwy = pck.load(open(DIGEPRED_PATH+'/data/networks/PWY_network_all_pairs_shortest_paths_Feb21_19.pkl', 'rb'))
            print("PWY distance initiated")
            self.dists_txt = pck.load(open(DIGEPRED_PATH+'/data/networks/Txt_network_all_pairs_shortest_paths_Feb21_19.pkl', 'rb'))
            print("TXT distance initiated")

            ## Load evoltuonary biology and genomics feature data files
            self.lof_dict = pck.load(open(DIGEPRED_PATH+'/data/evolgen/lof_pli_dict.pickle', 'rb'))
            print("Evolution data initiated")
            self.hap_insuf_dict = pck.load(open(DIGEPRED_PATH+'/data/evolgen/happloinsufficiency_dict.pickle', 'rb'))
            print("Happloinsufficieny data initiated")
            self.protein_age_dict = pck.load(open(DIGEPRED_PATH+'/data/evolgen/protein_age_dict.pickle', 'rb'))
            print("Protein age data initiated")
            self.dNdS_avg_dict = pck.load(open(DIGEPRED_PATH+'/data/evolgen/dNdS_avg.pickle', 'rb'))
            print("dNdS data initiated")
            self.gene_ess_dict = pck.load(open(DIGEPRED_PATH+'/data/evolgen/Gene_Essentiality_dict.txt', 'rb'))
            print("Gene essentiality data initiated")

    def create_path(self) -> str:
        return ROOT + "results/DiGePred/"

    def results_file_path(self, gene_set:str = "easy", model_path: Optional[str] = None) -> str:
        if model_path is None:
            if "holdout" in gene_set:
                return self.create_path()+"holdout/results.json"
            return self.create_path()+"results.json"
        else:
            model_dir = os.path.dirname(model_path)
            basename = os.path.basename(model_path).split(".")[0]
            if "holdout" in gene_set:
                return model_dir + "holdout/results.json"
            else:
                return model_dir + "results" + basename + ".json"

    def get_features(self, input_pairs):
        pairs = []
        import tqdm
        for pair in input_pairs:
            if pair[0] != None and pair[1] != None:
                pairs.append(tuple(sorted([pair[0], pair[1]])))
            else:
                pairs.append((None, None))

        pairs = sorted(pairs)

        new_list_pairs = [p for p in list(pairs)]

        all_data = []

        for x in tqdm.tqdm(new_list_pairs):

            data = [None for _ in range(21)]

            #  Pathway
            path1 = []
            path2 = []

            if x[0] in self.kegg_gene_to_path_codes or x[0] in self.reactome_gene_to_path_codes:

                if x[0] in self.kegg_gene_to_path_codes:
                    path1 = self.kegg_gene_to_path_codes[x[0]]

                if x[0] in self.reactome_gene_to_path_codes:
                    path1.extend(self.reactome_gene_to_path_codes[x[0]])

            if x[1] in self.kegg_gene_to_path_codes or x[1] in self.reactome_gene_to_path_codes:

                if x[1] in self.kegg_gene_to_path_codes:
                    path2 = self.kegg_gene_to_path_codes[x[1]]

                if x[1] in self.reactome_gene_to_path_codes:
                    path2.extend(self.reactome_gene_to_path_codes[x[1]])

            total = list(set(path1).union(path2))
            common = list(set(path1).intersection(path2))

            vqm = numpy.sqrt((len(path1) ** 2 + len(path2) ** 2) / 2)
            data[0] = vqm

            if len(total) == 0:
                data[1] = 0.
            else:
                data[1] = float(len(common)) / len(total)

            # HPO
            hpo1 = []
            hpo2 = []
            if x[0] in self.hpo_gene_to_code:
                hpo1 = self.hpo_gene_to_code[x[0]]
            if x[1] in self.hpo_gene_to_code:
                hpo2 = self.hpo_gene_to_code[x[1]]
            total = list(set(hpo1).union(hpo2))
            common = list(set(hpo1).intersection(hpo2))
            vqm = numpy.sqrt((len(hpo1) ** 2 + len(hpo2) ** 2) / 2)

            data[2] = vqm

            if len(total) == 0:
                data[3] = 0.
            else:
                data[3] = float(len(common)) / len(total)

            # PPI Network
            dist = []
            neighbors1 = []
            neighbors2 = []
            if x[0] in self.dists_ppi:
                neighbors1 = [p for p in nx.all_neighbors(self.G_ppi, x[0]) if p != x[0]]
                if x[1] in self.dists_ppi[x[0]]:
                    dist.append(self.dists_ppi[x[0]][x[1]])
            if x[1] in self.dists_ppi:
                neighbors2 = [p for p in nx.all_neighbors(self.G_ppi, x[1]) if p != x[1]]
                if x[0] in self.dists_ppi[x[1]]:
                    dist.append(self.dists_ppi[x[1]][x[0]])
            if dist != [] and min(dist) > 0:
                ppi_dist = 1 / float(min(dist))
            else:
                ppi_dist = 0.

            total = list(set(neighbors1).union(neighbors2))
            common = list(set(neighbors1).intersection(neighbors2))
            vqm = numpy.sqrt((len(neighbors1) ** 2 + len(neighbors2) ** 2) / 2)

            data[4] = vqm

            if len(total) == 0:
                data[5] = 0.
            else:
                data[5] = float(len(common)) / len(total)

            # data[i][8] = len(common)
            data[6] = ppi_dist

            # PWY Network
            dist = []
            neighbors1 = []
            neighbors2 = []
            if x[0] in self.dists_pwy:
                try:
                    neighbors1 = [p for p in nx.all_neighbors(self.G_pwy, x[0]) if p is not x[0]]
                    if x[1] in self.dists_pwy[x[0]]:
                        dist.append(self.dists_pwy[x[0]][x[1]])
                except:
                    pass
            if x[1] in self.dists_pwy:
                try:
                    neighbors2 = [p for p in nx.all_neighbors(self.G_pwy, x[1]) if p is not x[1]]
                    if x[0] in self.dists_pwy[x[1]]:
                        dist.append(self.dists_pwy[x[1]][x[0]])
                except:
                    pass

            if dist != [] and min(dist) > 0:
                pwy_dist = 1 / float(min(dist))
            else:
                pwy_dist = 0.

            total = list(set(neighbors1).union(neighbors2))
            common = list(set(neighbors1).intersection(neighbors2))
            vqm = numpy.sqrt((len(neighbors1) ** 2 + len(neighbors2) ** 2) / 2)

            data[7] = vqm

            if len(total) == 0:
                data[8] = 0.
            else:
                data[8] = float(len(common)) / len(total)

            # data[i][12] = len(common)
            data[9] = pwy_dist

            # TXT Network
            dist = []
            neighbors1 = []
            neighbors2 = []
            if x[0] in self.dists_txt:
                neighbors1 = [p for p in nx.all_neighbors(self.G_txt, x[0]) if p is not x[0]]
                if x[1] in self.dists_txt[x[0]]:
                    dist.append(self.dists_txt[x[0]][x[1]])
            if x[1] in self.dists_txt:
                neighbors2 = [p for p in nx.all_neighbors(self.G_txt, x[1]) if p is not x[1]]
                if x[0] in self.dists_txt[x[1]]:
                    dist.append(self.dists_txt[x[1]][x[0]])

            if dist != [] and min(dist) > 0:
                txt_dist = 1 / float(min(dist))
            else:
                txt_dist = 0.

            total = list(set(neighbors1).union(neighbors2))
            common = list(set(neighbors1).intersection(neighbors2))
            vqm = numpy.sqrt((len(neighbors1) ** 2 + len(neighbors2) ** 2) / 2)

            data[10] = vqm

            if len(total) == 0:
                data[11] = 0.
            else:
                data[11] = float(len(common)) / len(total)

            # data[i][16] = len(common)
            data[12] = txt_dist

            # Co-expression

            rankcoex1 = []
            rankcoex2 = []
            coexvalue = 0.
            if x[0] in self.coexpress_dict:
                rankcoex1 = [c for c in self.coexpress_dict[x[0]] if self.coexpress_dict[x[0]][c] < 100]
                if x[1] in self.coexpress_dict[x[0]]:
                    coexvalue = 1 / self.coexpress_dict[x[0]][x[1]]
            if x[1] in self.coexpress_dict:
                rankcoex2 = [c for c in self.coexpress_dict[x[1]] if self.coexpress_dict[x[1]][c] < 100]
                if x[0] in self.coexpress_dict[x[1]]:
                    coexvalue = 1 / self.coexpress_dict[x[1]][x[0]]

            total = list(set(rankcoex1).union(rankcoex2))
            common = list(set(rankcoex1).intersection(rankcoex2))
            vqm = numpy.sqrt((len(rankcoex1) ** 2 + len(rankcoex2) ** 2) / 2)

            data[13] = vqm

            if len(total) == 0:
                data[14] = 0.
            else:
                data[14] = float(len(common)) / len(total)

            # data[i][20] = len(common)
            data[15] = coexvalue

            # Lof

            if x[0] in self.lof_dict:
                v1 = float(self.lof_dict[x[0]])
            else:
                v1 = 0

            if x[1] in self.lof_dict:
                v2 = float(self.lof_dict[x[1]])
            else:
                v2 = 0

            vqm = numpy.sqrt((v1 ** 2 + v2 ** 2) / 2)
            data[16] = vqm

            # Happloinsufficiency Analysis

            if x[0] in self.hap_insuf_dict:
                v1 = float(self.hap_insuf_dict[x[0]])
            else:
                v1 = 0

            if x[1] in self.hap_insuf_dict:
                v2 = float(self.hap_insuf_dict[x[1]])
            else:
                v2 = 0

            vqm = numpy.sqrt((v1 ** 2 + v2 ** 2) / 2)
            data[17] = vqm

            # Protein Age

            if x[0] in self.protein_age_dict:
                v1 = float(self.protein_age_dict[x[0]])
            else:
                v1 = 0

            if x[1] in self.protein_age_dict:
                v2 = float(self.protein_age_dict[x[1]])
            else:
                v2 = 0

            vqm = numpy.sqrt((v1 ** 2 + v2 ** 2) / 2)
            data[18] = vqm

            # dN/DS

            if x[0] in self.dNdS_avg_dict:
                v1 = float(self.dNdS_avg_dict[x[0]])
            else:
                v1 = 0

            if x[1] in self.dNdS_avg_dict:
                v2 = float(self.dNdS_avg_dict[x[1]])
            else:
                v2 = 0

            vqm = numpy.sqrt((v1 ** 2 + v2 ** 2) / 2)
            data[19] = vqm

            # Gene Essentiality

            if x[0] in self.gene_ess_dict:
                v1 = numpy.mean([float(x) for x in self.gene_ess_dict[x[0]]])
            else:
                v1 = 0.
            if x[1] in self.gene_ess_dict:
                v2 = numpy.mean([float(x) for x in self.gene_ess_dict[x[1]]])
            else:
                v2 = 0.

            vqm = numpy.sqrt((v1 ** 2 + v2 ** 2) / 2)
            data[20] = vqm

            all_data.append(data)

        all_data = numpy.asarray(all_data)

        df = pd.DataFrame(all_data, index=new_list_pairs, columns=[
            # Pathways
            '#ofpathways',  # 0
            'common_pathways',  # 1
            # Phenotypes
            '#ofphenotypes',  # 2
            'common_phenotypes',  # 3
            # PPI network
            '#ofNeighborsPPI',  # 4
            '#Common_PPI_Neighbors',  # 5
            'PPI_network_dist',  # 6
            # PWY network
            '#ofNeighborsPWY',  # 7
            '#common_PWY_neighbors',  # 8
            'PWY_network_dist',  # 9
            # Txt network
            '#ofNeighborsTxt',  # 10
            '#Common_Txt_Neighbors',  # 11
            'Txt_network_dist',  # 12
            # Co-expression
            '#ofHighlyCoexpressed',  # 13
            '#Common_coexpressed',  # 14
            'Co-expression_coefficient',  # 15
            # LoF
            'LoFintolerance',  # 16
            # Haploinsuffiency
            'Haploinsufficiency',  # 17
            # Protein Age
            'protein_Age',  # 18
            # dN/dS
            'dN/dS',  # 19
            # Gene Essentiality
            'Essentiality',  # 20

        ])

        return df

    def CV(self, gene_set: str = "easy", k: int = 10, random_seed: int = 10, model_path: Optional[str] = None):

        gene_data = self.read_gene_set(gene_set)
        input, output = self.gene_data_to_input_output(gene_data)

        if model_path is None:
            model_dir = self.create_path()
        else:
            model_dir = os.path.dirname(model_path)

        results = KFoldResults(result_folder= model_dir, beta=None)
        if results.exists:
            exit()

        for i, (train, test) in CrossFold(False, k, random_seed, model_dir).split(input, output):
            print("Fold", i)

            input_train = asarray(input)[train]
            output_train = output[train]
            input_test = asarray(input)[test]
            output_test = output[test]

            self.fit(input_train, output_train)

            pred_out = self.predict(input_test)

            results.update(output_test, pred_out)

        results.save_results()


    def fit(self, input, output):
        if not self.pre_trained:
            input_features = self.get_features(self.map_ensembl_id_to_gene_name(input))
            clf = RandomForestClassifier(n_jobs=1, n_estimators=500, max_depth=15,
                                         random_state=10)
            clf.fit(input_features, output)
            self.classifier = clf

    def full_training_from_array(self, gene_data: ndarray[str]):

        input, output = self.gene_data_to_input_output(gene_data)
        input_features = self.get_features(self.map_ensembl_id_to_gene_name(input))

        clf = RandomForestClassifier(n_jobs=1, n_estimators=500, max_depth=15,
                                         random_state=10)
        clf.fit(input_features, output)

        self.classifier = clf


    def predict(self, input: Union[ndarray[str],List[Tuple[str,str]]]) -> ndarray[float]:
        input_features = self.get_features(self.map_ensembl_id_to_gene_name(input))
        return self.classifier.predict_proba(input_features)[:, 1]



if __name__ == "__main__":
    model = DiGePred()
