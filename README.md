# benchmarking_KGEMs_oligogenic

## Setting up

The usage of this repository requires first the download of two separate github repositories and the datasetset of predictions of the DIEP model. 
- The datasetset of predictions from DIEP can be downloaded as described on the github repository https://github.com/pmglab/DIEP.
- The DiGePred repository can be downloaded from https://github.com/CapraLab/DiGePred. Several files containing the necessary biological data need to be downloaded as described in the repository.
- Additionally, the Edge2vec Knowledge Graph Embedding Model also has an associated repository https://github.com/RoyZhengGao/edge2vec, which needs to be downloaded.

The file `config/embedding_source_paths.py` should be edited so that the `ROOT`, `EDGE2VEC_PATH`, `DIGEPRED_PATH` and `DIEP_PATH` point to the respective repositories or datasets. 

The required Python packages can be installed using `pip install -r requirements.txt` or `pip3 install -r requirements.txt`.

