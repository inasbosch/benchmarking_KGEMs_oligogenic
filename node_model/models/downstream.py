import os
from typing import Type, Dict, Union, List, Optional

from pykeen.models import ERModel

from node_model.embedding_classifier import Classifier
from node_model.embedding_generator import EmbeddingGenerator
from node_model.models.model import Model
from node_model.parameters import default_parameters
from node_model.pipeline import DownstreamPipeline
from utils.operators import Operator, IdentityOperator


class DownstreamModel(Model):
    def __init__(self,
                 KGE_model: Type[ERModel],*,
                 embedding_kwargs: Dict[str, int],
                 dataset_layers: Union[str,List[str]] = "all",
                 classifier_type: str = "forest",
                 classifier_kwargs: Optional[dict] = None,
                 embedding_transform: Optional[Operator] = None,
                 random_seed: int = 10,
                 batch_size: int = 512,
                 learning_rate: float = 0.1,
                 num_negs_per_pos: int = 2,
                 inverse: bool = False,
                 differentiate: bool = False,
                 stratified: bool = False,
                 holdout: bool = True):

        super().__init__(dataset_layers=dataset_layers,
                         stratified=stratified,
                         random_seed=random_seed,
                         holdout=holdout)

        self.has_embedding_transform = True
        self.dataset_layers = dataset_layers
        self.embedding_kwargs = embedding_kwargs

        if classifier_type not in ["mlp","forest","svm"]:
            raise Exception("Accepted classifier type values are \"mlp\", \"forest\" and \"svm\".")

        self.embedding_model = EmbeddingGenerator(
            KGE_model,
            embedding_kwargs,
            "",
            dataset_layers,
            batch_size,
            learning_rate,
            num_negs_per_pos,
            self.random_seed,
            inverse,
            differentiate
        )

        self.classifier_type = classifier_type

        if classifier_kwargs is None:
            if classifier_type in ["mlp","forest", "svm"]:
                self.classifier_kwargs = default_parameters[classifier_type]
            else:
                self.classifier_kwargs = {}
        else:
            self.classifier_kwargs = classifier_kwargs

        if "random_state" not in self.classifier_kwargs and classifier_type in ["mlp","forest", "svm"]:
            self.classifier_kwargs["random_state"] = self.random_seed

        if embedding_transform is None:
            self.embedding_transform = default_parameters["embedding_transform"]
        else:
            self.embedding_transform = embedding_transform

        if classifier_type not in ["mlp", "forest", "svm"]:
            self.embedding_transform = IdentityOperator()

        self.classifier_folder = self.create_path()
        os.makedirs(self.classifier_folder, exist_ok=True)

        self.classifier = Classifier(
            self.embedding_model,
            DownstreamPipeline(self.classifier_type,self.classifier_kwargs),
            self.embedding_transform
        )

    def create_path(self) -> str:
        if self.classifier_type == "forest":
            parameter_str = "/trees"+str(self.classifier_kwargs["n_estimators"])
            embedding_transform_name = "/" + self.embedding_transform.__name__
        else:
            if self.classifier_type == "mlp":
                parameter_str = "/hidden"+str(self.classifier_kwargs["hidden_layer_sizes"])
                embedding_transform_name = "/" + self.embedding_transform.__name__
            else:
                if self.classifier_type == "svm":
                    parameter_str = "/kernel" + str(self.classifier_kwargs["kernel"])
                    embedding_transform_name = "/" + self.embedding_transform.__name__
                else:
                    parameter_str = ""
                    embedding_transform_name = ""

        return self.embedding_model.embedding_folder + embedding_transform_name + "/" + self.classifier_type + \
             parameter_str

    def kfold_set(self, i: int) -> str:
        return ""

    def model_descriptor(self) -> str:
        return "Downstream" + "\n" + self.embedding_model.KGE_model.__name__ + "\n" + self.classifier_type
