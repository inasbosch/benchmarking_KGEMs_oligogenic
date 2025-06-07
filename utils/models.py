from pykeen.models import *
from pykeen.constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from pykeen.regularizers import LpRegularizer, Regularizer
from pykeen.nn.init import xavier_normal_norm_, xavier_uniform_, xavier_uniform_norm_
from pykeen.typing import Hint, Initializer, Constrainer

from class_resolver import HintOrType, OptionalKwargs
from typing import Mapping, ClassVar, Type, Any, Optional, List
from torch.nn import functional
from torch.nn.init import normal_, zeros_, uniform_
import os
from numpy import asarray
import pickle as pck

from config.embedding_source_paths import DATASET_PATH, EDGE2VEC_PATH

bock_file = DATASET_PATH + "bock_new.graphml"

class Edge2vec():
    def __init__(self, embedding_dim: int, num_epochs: int, bock_file: str,
                 output_folder: str,
                 type_size: int, walk_length: int = 3, num_walks: int = 2):

        self.bock_file = bock_file
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.type_size = type_size
        self.output_folder = output_folder

    def train(self):
        print("Training Edge2vec...")
        os.chdir(EDGE2VEC_PATH)
        os.system("python3 transition.py --input " + self.bock_file + " --output " + self.output_folder + "/matrix.txt" +
                  " --type_size " + str(self.type_size) + " --walk-length " + str(self.walk_length) +
                  " --num-walks " + str(self.num_walks) + " --em_iteration 5 --e_step 3" +
                  " --dimension " + str(self.embedding_dim) + " --iter " + str(self.num_epochs) +
                  " --directed")
        os.system("python3 edge2vec.py --input " + self.bock_file + " --matrix " + self.output_folder+"/matrix.txt " +
                  "--output " + self.output_folder + "/entity_embedding_array.txt --dimensions " + str(self.embedding_dim) +
                  " --walk-length " + str(self.walk_length) + " --num-walks " + str(self.num_walks) +
                  " --iter " + str(self.num_epochs) +
                  " --directed")

        with open(self.output_folder + "/entity_embedding_array.txt", "r") as f:
            first = True
            for row in f:
                if first:
                    num = int(row.split(" ")[0])
                    entity_embedding_array = [None for _ in range(num)]
                    first = False
                    continue

                id = int(row.split(" ")[0])
                vector = row.split(" ")[1:]
                entity_embedding_array[id] = [float(v) for v in vector]

        entity_embedding_array = asarray(entity_embedding_array)

        with open(self.output_folder + "/entity_embedding_array.bin", "wb") as f:
            pck.dump(entity_embedding_array, f)

        os.remove(self.output_folder+"/entity_embedding_array.txt")


map_to_KGEM = {
    KGE_model.__name__: KGE_model for KGE_model in [TransE, MuRE, RotatE, RESCAL, DistMult, ComplEx, QuatE, ConvE, ERMLP, Edge2vec]
}