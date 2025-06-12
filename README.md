# benchmarking_KGEMs_oligogenic

## Setting up

The usage of this repository requires first the download of two separate github repositories and the datasetset of predictions of the DIEP model. 
- The datasetset of predictions from DIEP can be downloaded as described on the github repository https://github.com/pmglab/DIEP.
- The DiGePred repository can be downloaded from https://github.com/CapraLab/DiGePred. Several files containing the necessary biological data need to be downloaded as described in the repository.
- Additionally, the Edge2vec Knowledge Graph Embedding Model also has an associated repository https://github.com/RoyZhengGao/edge2vec, which needs to be downloaded.

The file `config/embedding_source_paths.py` should be edited so that the `ROOT`, `EDGE2VEC_PATH`, `DIGEPRED_PATH` and `DIEP_PATH` point to the respective repositories or datasets. 

The required Python packages can be installed using `pip install -r requirements.txt` or `pip3 install -r requirements.txt`.

## Additional files

The datasets of the predictions obtained for each of the top six Knowledge Graph Embedding Models (KGEMs) for pairs classified as pathogenic can be found in the dropbox https://www.dropbox.com/scl/fo/hpv6dv5zwmdpnsdpnmu6u/AJAOxJwTKdK6XSvLm-0hoS8?rlkey=0rukv282233tnfwvv2n5y6ws9&st=tms6onou&dl=0 .

Additionally, the results generated and the files necessary to train, predict and evaluate the top pipelines for these six KGEMs are found at https://1drv.ms/f/c/b0194f3bfe765170/Et_hj72j2DVFlYKxWn7Eb1IBac6Z3cGH1b35xOy8HWnx2Q . In order to integrate these files in the repository, add them in the folder `benchmarking_KGEMs_oligogenic/results/top_models`. The file `bock_pickled.bin` - a binary file containing the BOCK knowledge graph - can also be found with this link and should be added in the folder `benchmarking_KGEMs_oligogenic/data/Datasets`.
