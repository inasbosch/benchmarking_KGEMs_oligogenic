import graph_tool
import pickle
from config.embedding_source_paths import DATASET_PATH

g = graph_tool.load_graph(DATASET_PATH + "Datasets/bock_new.graphml")
with open(DATASET_PATH + "Datasets/bock_pickled.bin", "wb") as f:
    pickle.dump(g, f)