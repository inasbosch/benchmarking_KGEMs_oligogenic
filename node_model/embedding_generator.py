from typing import Type, List, Union, Optional, Dict
from numpy import ndarray, concatenate, asarray
import pickle as pck
import os
import json
import torch
import shutil
import time

from pykeen.models import ERModel
from pykeen.datasets import EagerDataset
from pykeen.triples import TriplesFactory
from pykeen.losses import BCEWithLogitsLoss, MarginRankingLoss
from pykeen.training import SLCWATrainingLoop
from pykeen.sampling import BasicNegativeSampler
from pykeen.training.training_loop import CheckpointMismatchError
from pykeen.evaluation import RankBasedEvaluator

from config.embedding_source_paths import DATASET_PATH, ROOT, EDGE2VEC_PATH
from utils.dataset_utils import Dataset
from node_model.parameters import total_layers
from utils.models import Edge2vec

'''Inverse has not yet been implemented for Edge2vec!!!!'''

class EmbeddingGenerator():
    def __init__(self,
                 KGE_model: Type[ERModel],
                 embedding_kwargs: Dict[str, int],
                 classifier_class: str,
                 dataset_layers: Union[List[str],str] = "all",
                 batch_size: int = 512,
                 learning_rate: float = 0.1,
                 num_negs_per_pos: int = 2,
                 random_seed: int = 10,
                 inverse: bool = False,
                 differentiate: bool = False):

        result_folder = ROOT + "results/"

        self.KGE_model = KGE_model
        self.model_name = KGE_model.__name__
        self.embedding_dim = embedding_kwargs["embedding_dim"]
        if self.KGE_model.__name__ == "Edge2vec":
            if "walk_length" in embedding_kwargs:
                self.walk_length = embedding_kwargs["walk_length"]
            else:
                self.walk_length = 3
            if "num_walks" in embedding_kwargs:
                self.num_walks = embedding_kwargs["num_walks"]
            else:
                self.num_walks = 2

        self.num_epochs = embedding_kwargs["num_epochs"]

        self.result_folder = result_folder
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_negs_per_pos = num_negs_per_pos
        self.random_seed = random_seed
        self.inverse = inverse
        self.differentiate = differentiate

        if dataset_layers == "all" or (type(dataset_layers) == list and len(dataset_layers) == len(total_layers)):
            self.bock_name = "all"
            self.bock_layers = sorted(total_layers)
        else:
            bock_name = ""
            dataset_layers.sort()
            for layer in dataset_layers:
                bock_name += layer + "_"
            self.bock_name = bock_name[:-1]
            self.bock_layers = sorted(dataset_layers)

        if self.differentiate:
            self.bock_name += "_diff"

        self.initialize_class()
        self.model_folder = self.create_model_folder(classifier_class)
        self.embedding_folder = self.create_embedding_folder()
        if self.KGE_model.__name__ == "Edge2vec":
            self.new_bock_txt_path = EDGE2VEC_PATH+self.layers_str+self.bock_name+".txt"
            self.bock_txt_path = self.result_folder+self.layers_str+self.bock_name+".txt"
        else:
            self.bock_txt_path = self.result_folder+self.layers_str+self.bock_name+".txt"
        self.trained_model = None

    def initialize_class(self) -> None:
        if self.inverse:
            inverse_str = "inverse"
        else:
            inverse_str = "not_inverse"

        if self.bock_name == "all":
            layers_str = "all/"
        else:
            if self.bock_name == "all_diff":
                layers_str = "all_diff/"
            else:
                layers_str = "Layer"+str(len(self.bock_layers))+"/"+self.bock_name+"/"

        self.inverse_str = inverse_str
        self.layers_str = layers_str

    def create_model_folder(self, classifier_class: str) -> str:
        return (self.result_folder + self.layers_str + classifier_class + "/" +
                self.model_name + "/" + self.inverse_str + "/")

    def create_embedding_folder(self) -> str:
        if self.KGE_model.__name__ == "Edge2vec":
            return (self.model_folder + "embedd" + str(self.embedding_dim) +
                    "/epochs" + str(self.num_epochs) +
                    "/length" + str(self.walk_length) +
                    "/num_walks" + str(self.num_walks))

        return self.model_folder + "embedd" + str(self.embedding_dim) + "/epochs" + str(self.num_epochs)

    def create_bock_txt(self, keep_all: bool = False) -> None:
        g = pck.load(open(DATASET_PATH + "Datasets/bock_pickled.bin", "rb"))

        os.makedirs(self.result_folder+self.layers_str,exist_ok=True)

        with open(self.bock_txt_path,"w") as f:
            for edge in g.edges():
                node_label_1 = g.vp.labels[edge.source()][1:]
                node_label_2 = g.vp.labels[edge.target()][1:]
                edge_label = g.ep.label[edge]
                if not keep_all:
                    if node_label_1 == "OligogenicCombination" or node_label_2 == "OligogenicCombination":
                        continue
                    if node_label_1 == "Disease" or node_label_2 == "Disease":
                        continue
                    if node_label_1 == "Drug" or node_label_2 == "Drug":
                        continue
                    if "phenotype" not in self.bock_layers:
                        if node_label_1 == "Phenotype" or node_label_2 == "Phenotype":
                            continue
                    if "proteindomain" not in self.bock_layers:
                        if node_label_1 == "ProteinDomain" or node_label_2 == "ProteinDomain":
                            continue
                    if "proteinfamily" not in self.bock_layers:
                        if node_label_1 == "ProteinFamily" or node_label_2 == "ProteinFamily":
                            continue
                    if "proteincomplex" not in self.bock_layers:
                        if node_label_1 == "ProteinComplex" or node_label_2 == "ProteinComplex":
                            continue
                    if "biologicalprocess" not in self.bock_layers:
                        if node_label_1 == "BiologicalProcess" or node_label_2 == "BiologicalProcess":
                            continue
                    if "cellularcomponent" not in self.bock_layers:
                        if node_label_1 == "CellularComponent" or node_label_2 == "CellularComponent":
                            continue
                    if "molecularfunction" not in self.bock_layers:
                        if node_label_1 == "MolecularFunction" or node_label_2 == "MolecularFunction":
                            continue
                    if "pathway" not in self.bock_layers:
                        if node_label_1 == "Pathway" or node_label_2 == "Pathway":
                            continue
                    if "tissue" not in self.bock_layers:
                        if node_label_1 == "Tissue" or node_label_2 == "Tissue":
                            continue
                    if "physinteracts" not in self.bock_layers:
                        if edge_label == "physInteracts":
                            continue
                    if "coexpresses" not in self.bock_layers:
                        if edge_label == "coexpresses":
                            continue
                    if "seqsimilar" not in self.bock_layers:
                        if edge_label == "seqSimilar":
                            continue
                node_entity_id_1 = g.vp.id[edge.source()]
                node_entity_id_2 = g.vp.id[edge.target()]

                if self.differentiate and edge_label in ["associated", "resembles"]:
                    node_abbrev = sorted([node_label_1[0], node_label_2[0]])
                    real_edge_label = node_abbrev[0] + edge_label + node_abbrev[1]
                else:
                    real_edge_label = edge_label

                f.write(node_entity_id_1+"\t"+real_edge_label+"\t"+node_entity_id_2+"\n")

                if edge_label not in ["coexpresses", "seqSimilar", "physInteracts", "resembles"]:
                    continue

                f.write(node_entity_id_2 + "\t" + real_edge_label + "\t" + node_entity_id_1 + "\n")

    def create_dataset(self, gene_pairs: Optional[ndarray[str]] = None) -> EagerDataset:

        gene_triples = []
        if gene_pairs is not None:
            for pair in gene_pairs:
                gene_triples.append((pair[0],"oligogenic",pair[1]))
                gene_triples.append((pair[1],"oligogenic",pair[0]))

            tf = TriplesFactory.from_labeled_triples(concatenate([self.triples, gene_triples]),
                                                     create_inverse_triples=self.inverse)
        else:
            tf = TriplesFactory.from_labeled_triples(self.triples, create_inverse_triples=self.inverse)

        training, testing = tf.split([0.9, 0.1], random_state=10)

        dataset = EagerDataset(training=training, testing=testing,
                               metadata=self.bock_name)

        return dataset

    def create_triples(self):
        if not os.path.isfile(self.bock_txt_path):
            self.create_bock_txt()

        triples = []
        with open(self.bock_txt_path,"r") as f:
            for row in f:
                split_row = row.split("\t")
                split_row[-1] = split_row[-1][:-1]

                triples.append(split_row)

        self.triples = asarray(triples)


    def entity_label_to_id(self, dataset: Optional[EagerDataset] = None,
                           saving_folder: str = "", overwrite: bool = True) -> None:

        if saving_folder == "":
            saving_folder = self.embedding_folder

        if dataset is not None:
            if dataset.__class__ == EagerDataset:
                entity_id_to_label = dataset.training.entity_id_to_label
            else:
                entity_id_to_label = dataset.entity_id_to_label

        if os.path.isfile(saving_folder + "/entity_label_to_id.bin") and (not overwrite):
            with open(saving_folder + "/entity_label_to_id.bin", "rb") as f:
                self.label_to_id = pck.load(f)
        else:
            if os.path.isfile(saving_folder + "/entity_id_to_label.bin") and (not overwrite):
                with open(saving_folder + "/entity_id_to_label.bin", "rb") as f:
                    entity_id_to_label = pck.load(f)
            else:
                if dataset is None:
                    raise ValueError("Model is not yet trained (entity_id_to_label.bin file missing) and "
                                         "dataset is not provided.")

            entity_label_to_id = {}
            for id in entity_id_to_label:
                entity_label_to_id[entity_id_to_label[id]] = id

            with open(saving_folder + "/entity_label_to_id.bin", "wb") as f:
                pck.dump(entity_label_to_id, f)

            self.label_to_id = entity_label_to_id


    def load_embeddings(self):
        if os.path.isfile(self.embedding_folder + "/entity_embedding_array.bin"):
            with open(self.embedding_folder + "/entity_embedding_array.bin", "rb") as f:
                return pck.load(f)

    def load_label_to_id(self):
        if os.path.isfile(self.embedding_folder + "/entity_label_to_id.bin"):
            with open(self.embedding_folder + "/entity_label_to_id.bin", "rb") as f:
                return pck.load(f)

    def generate_embeddings(self, KGE_evaluation: bool = False, dataset: Optional[EagerDataset] = None,
                           gene_pairs: Optional[ndarray[str]] = None, saving_folder: str = "",
                           full_overwrite: bool = False, checkpoint_folder: Optional[str] = None) -> None:

        if saving_folder == "" or saving_folder is None:
            saving_folder = self.embedding_folder

        os.makedirs(saving_folder, exist_ok=True)

        if self.KGE_model.__name__ == "Edge2vec":
            if not os.path.isfile(self.bock_txt_path):
                self.create_bock_txt()


            redone = False
            if (not all(os.path.isfile(saving_folder+file) for file in ["/entity_embedding_array.bin",
                                                                        "/entity_id_to_label.bin"])
                    or full_overwrite):

                redone = True

                model = Edge2vec(self.embedding_dim, self.num_epochs,
                             self.new_bock_txt_path, saving_folder,
                             len(dataset.relation_label_to_id),
                             self.walk_length, self.num_walks)
                model.train()
                with open(saving_folder+"/entity_id_to_label.bin", "wb") as f:
                    pck.dump(dataset.entity_id_to_label, f)

            self.trained_model = "trained"

            self.entity_label_to_id(dataset, overwrite=redone)
            self.embeddings = pck.load(open(saving_folder + "/entity_embedding_array.bin", "rb"))
            return

        self.create_triples()

        if not full_overwrite:
            present = True
            for file in ["/entity_embedding_array.bin","/entity_id_to_label.bin"]:
                if not os.path.isfile(saving_folder+file):
                    present = False
                    break
            if present:
                if not os.path.isfile(saving_folder+"/entity_label_to_id.bin"):
                    dataset = self.create_dataset(gene_pairs)
                    self.entity_label_to_id(dataset)
                else:
                    self.label_to_id = pck.load(open(saving_folder+"/entity_label_to_id.bin","rb"))
                self.embeddings = pck.load(open(saving_folder+"/entity_embedding_array.bin","rb"))
                return None

        self.create_triples()

        if dataset is None:
            dataset = self.create_dataset(gene_pairs)

        if dataset.__class__.__name__ == "EagerDataset":
            dataset_name = dataset.metadata
        else:
            dataset_name = dataset.__class__.__name__

        # Model initialization
        if self.KGE_model.__name__ in ["MuRE", "ConvE", "ConvKB", "QuatE", "MuRP", "ERMLP"]:
            model = self.KGE_model(triples_factory=dataset.training, random_seed=self.random_seed,
                                    embedding_dim=self.embedding_dim,
                                    loss=BCEWithLogitsLoss,
                                    loss_kwargs=dict(
                                        reduction="mean"
                                    ),
                                    )
            config = {
                "metadata": {},
                "pipeline": {
                    "dataset": dataset_name,
                    "model": self.model_name,
                    "model_kwargs": {
                        "embedding_dim": self.embedding_dim
                    },
                    "optimizer": "Adagrad",
                    "optimizer_kwargs": {
                        "lr": self.learning_rate
                    },
                    "loss": "MarginRankingLoss",
                    "loss_kwargs": {
                        "reduction": "mean"
                    },
                    "training_loop": "slcwa",
                    "negative_sampler": "basic",
                    "negative_sampler_kwargs": {
                        "num_neg_per_pos": self.num_negs_per_pos
                    },
                    "training_kwargs": {
                        "num_epochs": self.num_epochs,
                        "batch_size": self.batch_size,
                    },
                    "evaluator_kwargs": {
                        "filtered": True
                    }
                }
            }
        else:
            model = self.KGE_model(triples_factory=dataset.training, random_seed=self.random_seed,
                                    embedding_dim=self.embedding_dim, loss=MarginRankingLoss,
                                    loss_kwargs=dict(
                                        reduction="mean",
                                        margin=1),
                                    regularizer_kwargs=dict(
                                        apply_only_once=False,
                                        weight=0.0001,
                                        p=2,
                                        normalize=False
                                    ))
            config = {
                "metadata": {},
                "pipeline": {
                    "dataset": dataset_name,
                    "model": self.model_name,
                    "model_kwargs": {
                        "embedding_dim": self.embedding_dim
                    },
                    "regularizer": "Lp",
                    "regularizer_kwargs": {
                        "apply_only_once": False,
                        "weight": 0.0001,
                        "p": 2,
                        "normalize": False
                    },
                    "optimizer": "Adagrad",
                    "optimizer_kwargs": {
                        "lr": self.learning_rate
                    },
                    "loss": "MarginRankingLoss",
                    "loss_kwargs": {
                        "reduction": "mean",
                        "margin": 1
                    },
                    "training_loop": "slcwa",
                    "negative_sampler": "basic",
                    "negative_sampler_kwargs": {
                        "num_neg_per_pos": self.num_negs_per_pos
                    },
                    "training_kwargs": {
                        "num_epochs": self.num_epochs,
                        "batch_size": self.batch_size,
                    },
                    "evaluator_kwargs": {
                        "filtered": True
                    }
                }
            }

        # Save the configuration

        f = open(saving_folder + "/config.json", "w")
        json.dump(config, f, indent=2)
        f.close()

        # Optimizer
        optimizer = torch.optim.Adagrad(params=model.get_grad_params(), lr=self.learning_rate)

        # Training loop
        training_loop = SLCWATrainingLoop(negative_sampler=BasicNegativeSampler,
                                          negative_sampler_kwargs=dict(
                                              num_negs_per_pos=self.num_negs_per_pos
                                          ),
                                          model=model,
                                          triples_factory=dataset.training,
                                          optimizer=optimizer)

        # Training the model
        if full_overwrite or checkpoint_folder is not None or checkpoint_folder != "":
            if checkpoint_folder is None or checkpoint_folder == "":
                if os.path.isfile(saving_folder + "/checkpoint.pt"):
                    os.remove(saving_folder + "/checkpoint.pt")
            else:
                shutil.copy(os.path.join(self.model_folder, checkpoint_folder) + "/checkpoint.pt",
                            saving_folder + "/checkpoint.pt")

        check = False
        if os.path.isfile(saving_folder + "/checkpoint.pt"):
            if os.path.isfile(saving_folder + "/results.json"):
                f = open(saving_folder + "/results.json")
                results = json.load(f)
                training_time = results["times"]["training"]
                check = True
        try:
            start_time = time.time()
            losses = training_loop.train(triples_factory=dataset.training,
                                         num_epochs=self.num_epochs,
                                         batch_size=self.batch_size,
                                         checkpoint_directory=saving_folder,
                                         checkpoint_name="checkpoint.pt",
                                         checkpoint_frequency=5)
            end_time = time.time()
            if not check:
                training_time = end_time - start_time
        except CheckpointMismatchError:
            os.remove(saving_folder + "/checkpoint.pt")
            start_time = time.time()
            losses = training_loop.train(triples_factory=dataset.training,
                                         num_epochs=self.num_epochs,
                                         batch_size=self.batch_size,
                                         checkpoint_directory=saving_folder,
                                         checkpoint_name="checkpoint.pt",
                                         checkpoint_frequency=5)
            end_time = time.time()
            training_time = end_time - start_time

        # Save the embeddings
        entity_embedding_array = model.entity_representations[0](indices=None).detach().numpy()

        with open(saving_folder + "/entity_embedding_array.bin", "wb") as f:
            pck.dump(entity_embedding_array, f)

        # Save the entity and relations IDs
        with open(saving_folder + "/entity_id_to_label.bin", "wb") as f:
            pck.dump(dataset.training.entity_id_to_label, f)


        # Evaluation
        if KGE_evaluation:
            evaluator = RankBasedEvaluator()
            testing_triples = dataset.testing.mapped_triples
            start_time = time.time()
            metrics = evaluator.evaluate(model=model,
                                         mapped_triples=testing_triples,
                                         additional_filter_triples=[
                                             dataset.training.mapped_triples,
                                             dataset.validation.mapped_triples
                                         ])
            end_time = time.time()
            evaluation_time = end_time - start_time
            times = {"evaluation": evaluation_time, "training": training_time}

            # Save the results
            results = {"losses": losses, "metrics": metrics.to_dict(), "times": times}
            f = open(saving_folder + "/results.json", "w")
            json.dump(results, f, indent=2, sort_keys=True)
            f.close()
        else:
            # Save the results
            results = {"losses": losses, "times": {"training": training_time}}
            f = open(saving_folder + "/results.json", "w")
            json.dump(results, f, indent=2, sort_keys=True)
            f.close()

        self.entity_label_to_id(dataset,saving_folder,True)
        self.trained_model = model
        self.embeddings = entity_embedding_array