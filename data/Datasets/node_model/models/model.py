import os.path
from random import seed, sample, shuffle
from math import nan

from typing import Dict
from abc import abstractmethod
import json
import pickle as pck
import inspect
from itertools import combinations

from config.embedding_source_paths import ROOT, DATASET_PATH, DIGEPRED_PATH
from node_model.parameters import *
from node_model.pipeline import *
from node_model.cross_fold import CrossFold
from node_model.results import KFoldResults, SingleResults
from node_model.utils.classifier_utils import classify
from utils.dataset_utils import sort_genes


class Model():

    @abstractmethod
    def __init__(self,*,dataset_layers: Union[str,List[str]] = "all",
                 stratified: bool = False,
                 beta: float = 1,
                 random_seed: int = 10,
                 holdout: bool = True,
                 fig_add: str = ""):

        seed(random_seed)
        self.classifier = None
        self.embedding_transform = None
        self.embedding_model = None
        self.classifier_folder = None
        self.stratified = stratified
        if self.stratified:
            holdout = False
        self.random_seed = random_seed
        self.dataset_layers = dataset_layers
        self.beta = beta
        self.holdout = holdout

        self.fig_add_given = fig_add
        self.compute_fig_add()

        self.has_embedding_transform = False

    def compute_fig_add(self):
        fig_add = ""

        if self.stratified:
            fig_add += "_strat"
            fig_add += "_kmeans"

        else:
            if self.holdout:
                fig_add += "_holdout"

        if self.beta != -1:
            fig_add += "_f" + str(self.beta) + "score"

        self.fig_add = fig_add

    def create_path(self) -> str:
        pass

    def model_variables_dict(self) -> Dict[str,Union[int,bool,Operator,Dict[str,Union[int,str]]]]:
        return_dict = {}
        for i in inspect.getmembers(self):
            if not i[0].startswith('_'):
                if not inspect.ismethod(i[1]):
                    return_dict[i[0]] = i[1]

        for i in inspect.getmembers(self.embedding_model):
            if not i[0].startswith('_'):
                if not inspect.ismethod(i[1]):
                    return_dict[i[0]] = i[1]

        del return_dict["classifier"]
        del return_dict["embedding_dim"]
        del return_dict["num_epochs"]
        del return_dict["embedding_folder"]
        del return_dict["classifier_folder"]
        del return_dict["embedding_model"]
        del return_dict["trained_model"]
        if "embeddings" in return_dict:
            del return_dict["embeddings"]
        if "label_to_id" in return_dict:
            del return_dict["label_to_id"]
        if "entity_id_to_label" in return_dict:
            del return_dict["entity_id_to_label"]
        if "num_walks" in return_dict:
            del return_dict["num_walks"]
        if "walk_length" in return_dict:
            del return_dict["walk_length"]
        if "relation_label_to_id" in return_dict:
            del return_dict["relation_label_to_id"]
        if "new_bock_txt_path" in return_dict:
            del return_dict["new_bock_txt_path"]
        if "triples" in return_dict:
            del return_dict["triples"]
        del return_dict["fig_add_given"]
        del return_dict["has_embedding_transform"]
        del return_dict["model_folder"]
        del return_dict["result_folder"]
        del return_dict["layers_str"]
        del return_dict["model_name"]
        del return_dict["inverse_str"]
        del return_dict["bock_name"]
        del return_dict["bock_layers"]
        del return_dict["bock_txt_path"]
        del return_dict["random_seed"]
        if "double" in return_dict:
            del return_dict["double"]
        if not self.has_embedding_transform:
            del return_dict["embedding_transform"]

        del return_dict["fig_add"]

        return return_dict

    def model_descriptor(self) -> str:
        pass

    def read_results(self, gene_set: str = "easy", model_path: Optional[str] = None) -> Dict[str,float]:
        with open(self.results_file_path(gene_set, model_path),"r") as f:
            results = json.load(f)

        return results

    def map_gene_name_to_ensembl_id(self, input: ndarray):
        hgnc_gene_name_to_ensembl, _ = hgnc_ensembl_mapper()

        pairs = []
        for pair in input:
            if pair[0] not in hgnc_gene_name_to_ensembl or pair[1] not in hgnc_gene_name_to_ensembl:
                pairs.append((None, None))
            else:
                g1 = hgnc_gene_name_to_ensembl[pair[0]][0]
                g2 = hgnc_gene_name_to_ensembl[pair[1]][0]
                pairs.append(tuple(sorted([g1, g2])))

        return asarray(pairs)

    def map_ensembl_id_to_gene_name(self, input: ndarray):
        _, ensembl_id_to_hgnc_id = hgnc_ensembl_mapper()

        pairs = []
        for pair in input:
            if pair[0] not in ensembl_id_to_hgnc_id or pair[1] not in ensembl_id_to_hgnc_id:
                pairs.append((None, None))
            else:
                g1 = ensembl_id_to_hgnc_id[pair[0]][0]
                g2 = ensembl_id_to_hgnc_id[pair[0]][1]
                pairs.append(tuple(sorted([g1, g2])))

        return asarray(pairs)

    def results_file_path(self, gene_set: str = "easy", model_path: Optional[str] = None) -> str:
        if model_dir is None:
            return self.create_path()+"/"+gene_set+"/results"+self.fig_add+".json"
        else:
            model_dir = os.path.dirname(model_path)
            basename = os.path.basename(model_path).split(".")[0]
            if "holdout" in gene_set:
                return self.create_path()+"/holdout/results"+basename+".json"
            return model_dir + "/results"+basename+".json"

    def write_results(self, dict_key: str, dict_val: float,
                      gene_set: str = "easy", model_path: Optional[str] = None) -> None:

        with open(self.results_file_path(gene_set, model_path), "r") as f:
            results = json.load(f)

        results[dict_key] = dict_val

        with open(self.results_file_path(gene_set, model_path), "w") as f:
            json.dump(results,f)

    def kfold_set(self, i: int) -> str:
        pass

    def gene_data_to_input_output(self, gene_data: ndarray[str]):
        input = []
        output = []
        for row in gene_data:
            input.append([row[0], row[1]])
            output.append(int(row[2]))

        input = asarray(input)
        output = asarray(output)

        return input, output


    def read_gene_set(self) -> ndarray:

        gene_file = "genes_data.bin"


        with open(DATASET_PATH + "Datasets/" + gene_file, "rb") as f:
            gene_data = pck.load(f)


        with open(DATASET_PATH+"Datasets/holdout_neutral_separate.tsv", "r") as f:
             neutral_holdout = []
             for row in f:
                 split_row = row.split("\t")
                 neutral_holdout.append([split_row[0], split_row[1][:-1]])

        new_gene_data = []
        for row in gene_data:
            if self.holdout:
                found = False
                for pair in neutral_holdout:
                    if row[0] in pair or row[1] in pair:
                        found = True
                        break
                if found:
                    continue

            shuffled_genes = [row[0], row[1]]
            shuffle(shuffled_genes)
            new_gene_data.append(shuffled_genes+list(row[2:]))
        gene_data = new_gene_data

        if self.stratified:

            file_path = self.embedding_model.create_embedding_folder() + "/kmeans_"

            with open(file_path+"pathogenic_partition.bin","rb") as f:
                positive_recombined = pck.load(f)
            with open(file_path+"neutral_partition.bin","rb") as f:
                neutral_recombined = pck.load(f)

            present_pairs = []
            for bin in positive_recombined:
                for pair in bin:
                    present_pairs.append(pair)
            for bin in neutral_recombined:
                for pair in bin:
                    present_pairs.append(pair)

            new_gene_data = []
            for row in gene_data:
                if (row[0], row[1]) in present_pairs or (row[1], row[0]) in present_pairs:
                    shuffled_genes = [row[0], row[1]]
                    shuffle(shuffled_genes)
                    new_gene_data.append(shuffled_genes + list(row[2:]))
            gene_data = new_gene_data

        return asarray(gene_data)

    def holdout_vs_mono(self, overwrite: bool = False, model_path: Optional[str] = None):
        initial_holdout = self.holdout

        if not initial_holdout and not self.stratified:
            print("holdout value overridden to True.")
            self.holdout = True

        self.compute_fig_add()

        if model_path is None:
            model_dir = self.create_path() + "/easy"
            model_base = self.fig_add
        else:
            model_dir = os.path.dirname(model_path)
            model_base = os.path.basename(model_path).split(".")[0]

        if overwrite or not os.path.exists(model_dir + "holdout_combined_dict" + model_base + ".bin"):
            path_file = open(DATASET_PATH + "Datasets/holdout_separate.tsv", "r")
            path_holdout = []
            for row in path_file:
                split_row = row.split("\t")
                split_row[1] = split_row[1][:-1]
                path_holdout.append(split_row)

            neut_file = open(DATASET_PATH + "Datasets/holdout_neutral_separate.tsv", "r")
            neut_holdout = []
            for row in neut_file:
                split_row = row.split("\t")
                split_row[1] = split_row[1][:-1]
                neut_holdout.append(split_row)

            mono_file = open(DATASET_PATH + "Datasets/holdout_monogenic.tsv", "r")
            mono_holdout = []
            for row in mono_file:
                split_row = row.split("\t")
                split_row[1] = split_row[1][:-1]
                mono_holdout.append(split_row)

            if self.classifier is None:
                self.full_training("easy")

            path_pred = list(self.predict(path_holdout))
            neut_pred = list(self.predict(neut_holdout))
            mono_pred = list(self.predict(mono_holdout))

            path_pred += [nan]*(len(mono_pred)-len(path_pred))
            neut_pred += [nan]*(len(mono_pred)-len(neut_pred))

            result_dict = {"Pathogenic": path_pred, "Combined": mono_pred, "Neutral": neut_pred}

            with open(model_dir + "holdout_combined_dict" + model_base + ".bin", "wb") as f:
                pck.dump(result_dict, f)

            self.holdout = initial_holdout
            self.compute_fig_add()
        else:
            self.holdout = initial_holdout
            self.compute_fig_add()
            result_dict = pck.load(open(self.create_path()+"/easy/holdout_combined_dict.bin", "rb"))

        combined_pred = result_dict["Combined"]
        if os.path.exists(self.results_file_path("easy", model_path)):
            if "threshold" in self.read_results("easy", model_path):
                trsh = self.read_results("easy", model_path)["threshold"]
            else:
                trsh = 0.5
        else:
            trsh = 0.5

        combined_path = len([p for p in combined_pred if p > trsh])/len(combined_pred)

        results = self.read_results("easy/holdout", model_path)
        results["Combined_pathogenic"] = combined_path
        with open(self.results_file_path("easy/holdout", model_path), "w") as f:
            json.dump(results, f)

        return result_dict

    def CV_from_array(self, gene_data: ndarray[str], k: int = 10,
                      random_seed: int = 10, model_path: Optional[str] = None) -> None:

        if model_path is None:
            model_dir = self.classifier_folder + "/easy"
            fig_add = self.fig_add
        else:
            model_dir = os.path.dirname(model_path)
            fig_add = os.path.basename(model_path).split(".")[0]


        if self.embedding_transform.rvis:
            gene_data = sort_genes(gene_data)

        input, output = self.gene_data_to_input_output(gene_data)

        results = KFoldResults(result_folder=model_dir, fig_add=fig_add, beta=self.beta)

        if results.exists:
            return

        if model_path is None:
            if hasattr(self, "embedding_model"):
                if self.embedding_model is not None:
                    file_path = self.embedding_model.create_embedding_folder()
                else:
                    file_path = None
            else:
                file_path = None
        else:
            file_path = model_dir

        for i, (train, test) in CrossFold(self.stratified, k, random_seed, file_path).split(input,output):
            print("Fold",i)

            input_train = input[train]
            output_train = output[train]
            input_test = input[test]

            if self.embedding_transform.double:
                output_test = []
                for test_row in test:
                    output_test.append(output[test_row])
                    output_test.append(output[test_row])
                output_test = asarray(output_test)
            else:
                output_test = output[test]

            gene_data_train = []
            for inp, out in zip(input_train, output_train):
                gene_data_train.append([inp[0], inp[1], out])

            if hasattr(self, "embedding_model"):
                if self.embedding_model is not None:
                    self.classifier.fit(input_train, output_train)
                else:
                    self.fit(input_train, output_train)
            else:
                self.fit(input_train, output_train)

            pred_out = self.predict(input_test)

            results.update(output_test,pred_out)

        results.save_results()

    def CV(self, model_path: Optional[str] = None, k: int = 10, random_seed: int = 10) -> None:

        gene_data = self.read_gene_set()

        self.CV_from_array(gene_data, k, random_seed, model_path=model_path)


    def holdout(self, full_overwrite: bool = False, model_path: Optional[str] = None):

        input = []
        output = []

        if model_path is None:
            model_dir = self.classifier_folder + "/easy/holdout"
            fig_add = self.fig_add
        else:
            model_dir = os.path.dirname(model_path) + "/holdout"
            fig_add = os.path.basename(model_path).split(".")[0]

        inital_holdout = self.holdout
        if not inital_holdout and not self.stratified:
            print("holdout overridden to be True")
            self.holdout = True


        path_file = open(
                DATASET_PATH+"Datasets/holdout_separate.tsv",
                "r")

        neut_file = open(
                DATASET_PATH+"Datasets/holdout_neutral_separate.tsv", "r")

        for row in path_file:
            g1, g2 = row.split("\t")
            input.append([g1, g2[:-1]])
            output.append(1)

        for row in neut_file:
            g1, g2 = row.split("\t")
            input.append([g1, g2[:-1]])
            output.append(0)

        results = SingleResults(results_folder=model_dir, fig_add=fig_add)

        self.full_training(full_overwrite=full_overwrite, model_path=model_path)

        pred_out = self.predict(input)

        trsh = 0.5
        if os.path.isfile(self.results_file_path("easy/holdout", model_path)):
            if "threshold" in self.read_results("easy/holdout", model_path):
                trsh = self.read_results("easy/holdout", model_path)["threshold"]


        results.update(output, pred_out, trsh)

        results.save_results(trsh)

        self.holdout = inital_holdout

        pred_dict = {}
        for pair, pred in zip(input, pred_out):
            pred_dict[tuple(sorted(pair))] = pred

        return pred_dict


    def predict(self, input: Union[ndarray[str],List[Tuple[str,str]]]) -> ndarray[float]:
        pred = self.classifier.predict(asarray(input))

        real_pred = []
        if self.embedding_transform.double:
            for i in range(pred.shape[0]//2):
                real_pred.append(min(pred[2*i],pred[2*i+1]))
        else:
            real_pred = pred

        return real_pred

    def classify(self, input: Union[ndarray[str],List[Tuple[str,str]]],
                 gene_set: str = "easy",
                 model_path: Optional[str] = None) -> ndarray[float]:
        try:
            trsh = self.read_results(gene_set, model_path)["threshold"]
        except:
            self.CV()
            trsh = self.read_results(gene_set, model_path)["threshold"]

        pred_round = classify(asarray(self.predict(input)), trsh)

        return pred_round

    def full_training(self, model_path: Optional[str] = None, full_overwrite: bool = False, overwrite: bool = False):

        fig_add = ""

        if self.stratified:
            fig_add = "_strat"

        else:
            if self.holdout:
                fig_add += "_holdout"

        if self.fig_add_given != "":
            fig_add += "_" + self.fig_add_given

        if model_path is None:
            model_path = self.create_path() + "/easy/trained_model" + fig_add + ".bin"

        if not overwrite and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.classifier = pck.load(f)
            return

        gene_data = self.read_gene_set()

        self.full_training_from_array(gene_data, full_overwrite)

        dirname = os.path.dirname(model_path)
        os.makedirs(dirname, exist_ok=True)
        if self.fig_add_given != "":
            with open(model_path, "wb") as f:
                pck.dump(self.classifier, f)

        self.model_path = model_path
        return gene_data

    def full_training_from_array(self, gene_data: ndarray[str], full_overwrite: bool = False,
                                 checkpoint_folder: Optional[str] = None):

        print("Training using " + str(gene_data.shape[0]) + " gene pairs.")

        input, output = self.gene_data_to_input_output(gene_data)

        if hasattr(self, "embedding_model"):
            if self.embedding_model is not None:
                print("Training")
                self.classifier.fit(input, output, self.create_path() + self.kfold_set(-1),
                                    checkpoint_folder=checkpoint_folder, full_overwrite=full_overwrite)
            else:
                self.fit(input, output)


    def male_infertility_test(self, model_path: Optional[str] = None):
        if model_path is None:
            model_dir = self.create_path() + "/easy"
            fig_add = self.fig_add
        else:
            model_dir = os.path.dirname(model_path)
            fig_add = os.path.basename(model_path).split(".")[0]

        if not os.path.exists(model_dir+"/male_infertility_predictions"+fig_add+".csv"):
            pairs = []
            with open(DATASET_PATH + "Datasets/male_infertility.tsv", "r") as f:
                for row in f:
                    split_row = row.split("\t")
                    split_row[1] = split_row[1][:-1]
                    pairs.append(list(split_row))


            self.full_training()


            predictions = self.predict(pairs)
            with open(model_dir+"/male_infertility_predictions"+fig_add+".csv", "w") as f:
                for pair, pred in zip(pairs, predictions):
                    f.write(pair[0]+"\t"+pair[1]+"\t"+str(pred)+"\n")

        return_dict = {}
        with open(model_dir+"/male_infertility_predictions"+fig_add+".csv", "r") as f:
            for row in f:
                if "gene" in row:
                    continue
                split_row = row.split("\t")
                if len(split_row) == 5:
                    pair = split_row[:2]
                else:
                    pair = sorted(self.map_ensembl_id_to_gene_name([split_row[:2]])[0])
                pair = sorted(pair)
                pred = float(split_row[-1])
                return_dict[(pair[0],pair[1])] = pred

        return return_dict

    def male_infertility_panel_predictions(self):
        if model_path is None:
            model_dir = self.create_path()
            fig_add = self.fig_add
        else:
            model_dir = os.path.dirname(model_path)
            fig_add = os.path.basename(model_path).split(".")[0]


        if not os.path.exists(model_dir+"/male_infertility_panel_predictions"+fig_add+".csv"):
            gene_panel = []
            with open(DATASET_PATH + "Datasets/hop_gene_panel.txt", "r") as f:
                for row in f:
                    gene_panel.append(row[:-1])

            gene_pairs = list(combinations(gene_panel, 2))

            gene_pairs = self.map_gene_name_to_ensembl_id(gene_pairs)


            self.full_training("easy")

            predictions = self.predict(gene_pairs)
            with open(model_dir+"/male_infertility_panel_predictions"+fig_add+".csv") as f:
                for pair, pred in zip(gene_pairs, predictions):
                    f.write(pair[0]+"\t"+pair[1]+"\t"+str(pred)+"\n")


        return_dict = {}
        with open(model_dir+"/male_infertility_panel_predictions"+fig_add+".csv", "r") as f:
            for row in f:
                split_row = row.split("\t")
                pair = split_row[:2]
                pred = float(split_row[-1])
                return_dict[(pair[0], pair[1])] = pred

        return return_dict