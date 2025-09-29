# benchmarking_KGEMs_oligogenic

## Setting up

The usage of this repository requires first the download of two separate github repositories and the datasetset of predictions of the DIEP model. 
- The datasetset of predictions from DIEP can be downloaded as described on the github repository https://github.com/pmglab/DIEP.
- The DiGePred repository can be downloaded from https://github.com/CapraLab/DiGePred. Several files containing the necessary biological data need to be downloaded as described in the repository.
- Additionally, the Edge2vec Knowledge Graph Embedding Model also has an associated repository https://github.com/RoyZhengGao/edge2vec, which needs to be downloaded.

The file `config/embedding_source_paths.py` should be edited so that the `ROOT`, `EDGE2VEC_PATH`, `DIGEPRED_PATH` and `DIEP_PATH` point to the respective repositories or datasets. 

The required Python packages can be installed using `pip install -r requirements.txt` or `pip3 install -r requirements.txt`.

## Additional files

The datasets of the predictions obtained for each of the top six Knowledge Graph Embedding Models (KGEMs), the generated results and the files necessary to train, predict and evaluate the top pipelines for these six KGEMs are found at:
<li> TransE: https://doi.org/10.5281/zenodo.17170197 </li>
<li> MuRE: https://doi.org/10.5281/zenodo.17187155 </li>
<li> RotatE: https://doi.org/10.5281/zenodo.17187391 </li>
<li> DistMult: https://doi.org/10.5281/zenodo.17181522 </li>
<li> QuatE: https://doi.org/10.5281/zenodo.17187536 </li>
<li> ERMLP: https://doi.org/10.5281/zenodo.17192335 </li>



In order to integrate these files in the repository, add them in the folder `benchmarking_KGEMs_oligogenic/results/top_models`. 

The file `bock_new.graphml` - a GraphML file containing the BOCK knowledge graph - can also be found on Zenodo (https://doi.org/10.5281/zenodo.14979916) and should be added in the folder `benchmarking_KGEMs_oligogenic/data/Datasets`. Then run `python3 generate_bock.py` to generate the binary file `bock_pickled.bin`.

## Tutorial

Once all the necessary files have been downloaded, five different commands can be used.
- `python3 main.py train` allows to train a new model with the indicated parameters.
- `python3 main.py predict` uses a previously trained model to generate predictions on a provided tab-delimited file of gene pairs.
- `python3 main.py evaluate` uses cross-validation on a provided model or for the indicated parameters. Add `-stratified True` to use a stratified cross-validation approach.
- `python3 main.py stratify` generates stratified folds for the provided model.
- `python3 main.py independent` generates the results shown in the article on the holdout set for a provided model.
