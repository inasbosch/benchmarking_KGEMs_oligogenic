from typing import Tuple, Optional, List, Union
from numpy import ndarray, asarray, ones
import pickle
import os

from node_model.embedding_generator import EmbeddingGenerator
from node_model.pipeline import Pipeline, LinkPredictionPipeline
from utils.operators import Operator


class Classifier():
    def __init__(self,
                 embedding_model: EmbeddingGenerator,
                 classification_pipeline: Pipeline,
                 embedding_transform: Operator
                 ):

        self.embedding_model = embedding_model
        self.classification_pipeline = classification_pipeline
        self.embedding_transform = embedding_transform


    def fit(self, input: ndarray[str], output: Optional[ndarray[int]] = None,
            kfold_set: Optional[str] = None, checkpoint_folder: Optional[str] = None,
            full_overwrite: bool = False) -> None:

        if type(self.classification_pipeline) == LinkPredictionPipeline:
            gene_pairs = []
            if output is not None:
                for pair, out in zip(input,output):
                    if out == 1:
                        gene_pairs.append(pair)
            else:
                gene_pairs = input
            gene_pairs = asarray(gene_pairs)

            self.embedding_model.generate_embeddings(gene_pairs=gene_pairs,saving_folder=kfold_set,
                                                     checkpoint_folder=checkpoint_folder,
                                                     full_overwrite=full_overwrite)

            if kfold_set is None or kfold_set == "":
                saving_folder = self.embedding_model.embedding_folder
            else:
                saving_folder = kfold_set

            with open(saving_folder + "/relation_embedding_array.bin", "rb") as f:
                relation_embedding_array = pickle.load(f)

            with open(saving_folder + "/relation_id_to_label.bin", "rb") as f:
                relation_id_to_label = pickle.load(f)

            for id in relation_id_to_label:
                if relation_id_to_label[id] == "oligogenic":
                    self.classification_pipeline.classifier.oligo_embedding = relation_embedding_array[id]

            self.classification_pipeline.classifier.interaction_function = (
                self.embedding_model.trained_model.interaction.func.__self__.score)

        else:
            if self.embedding_model.trained_model is None:
                self.embedding_model.generate_embeddings(checkpoint_folder=checkpoint_folder, full_overwrite=full_overwrite)

        emb_input, emb_output = self.input_output_transform(input,output)
        self.classification_pipeline.fit(emb_input,emb_output)


    def pair_transform(self, pair: Tuple[str,str]) -> ndarray[float]:
        if not hasattr(self.embedding_model, 'label_to_id'):
            self.embedding_model.label_to_id = self.embedding_model.load_label_to_id()

        if not hasattr(self.embedding_model, 'embeddings'):
            self.embedding_model.embeddings = self.embedding_model.load_label_to_id()

        if (pair[0] in self.embedding_model.label_to_id) and (pair[1] in self.embedding_model.label_to_id):
            id1 = self.embedding_model.label_to_id[pair[0]]
            id2 = self.embedding_model.label_to_id[pair[1]]
            emb1 = self.embedding_model.embeddings[id1]
            emb2 = self.embedding_model.embeddings[id2]
            return self.embedding_transform.compute((id1,emb1), (id2,emb2))
        else:
            emb1 = None
            emb2 = None
            return self.embedding_transform.compute(emb1,emb2)

    def input_output_transform(self, input: Union[ndarray[str],List[Tuple[str,str]]], output: Optional[ndarray[int]] = None) \
            -> Tuple[List[float], List[int]]:

        emb_input = []
        emb_output = []
        input = asarray(input)

        if output is None:
            output = ones(input.shape[0])

        for pair, out in zip(input, output):
            for emb in self.pair_transform(pair):
                emb_input.append(emb)
                emb_output.append(out)

        return emb_input, emb_output

    def predict(self, input: ndarray[str]) -> ndarray[float]:
        if (self.embedding_model.trained_model is None and
                not os.path.isfile(self.embedding_model.embedding_folder+"/entity_embedding_array.bin")):
            raise Exception("Model has to be trained before it can make predictions.")
        else:
            emb_input, _ = self.input_output_transform(input)
            return self.classification_pipeline.predict(emb_input)

    def classify(self,input: ndarray[str], max_t: float) -> ndarray[int]:
        return self.classification_pipeline.classify(input,max_t)