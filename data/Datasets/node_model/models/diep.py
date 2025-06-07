from typing import Optional, List, Union, Tuple

from numpy import ndarray
import tqdm
import pickle

from node_model.models.model import Model
from config.embedding_source_paths import DIEP_PATH
from utils.operators import IdentityOperator

'''An implementation of DIEP using the prediction database as provided on https://github.com/pmglab/DIEP, the
github associated to the article

Yuan, Yangyang et al. “An accurate prediction model of digenic interaction for estimating pathogenic gene pairs 
of human diseases.” Computational and Structural Biotechnology Journal 20 (2022): 3639 - 3652.

This implementation does not allow to fit the model to new data.'''

class DIEP(Model):
    def __init__(self, fig_add: str = ""):

        super().__init__(fig_add=fig_add)

        self.embedding_transform = IdentityOperator()
        self.classifier_folder = self.create_path()
        self.fig_add = ""

    def create_path(self):
        return DIEP_PATH + "/results"

    def results_file_path(self, gene_set:str = "easy") -> str:
        return self.create_path()+"/"+gene_set+"/results.json"


    def full_training(self, gene_set: str, full_overwrite: bool = False, checkpoint_folder: Optional[str] = None,
                      exclude: Optional[List[List[str]]] = None):

        print("Training is not implemented for DIEP model.")


    def full_training_from_array(self, gene_data: ndarray[str], gene_set: str,
                                 full_overwrite: bool = False, checkpoint_folder: Optional[str] = None,
                                 fig_add: bool = False):

        print("Training is not implemented for DIEP model.")


    def fit(self, input_data: ndarray, output_data: ndarray):
        print("Training is not implemented for DIEP model.")


    def pred_file_to_dict(self):
        pred_dict = {}
        with open(DIEP_PATH + "/Dataset_S7-CodingDIScores/Coding_predict_fixed.txt", "r") as f:
            first = True
            for row in tqdm.tqdm(f):
                if first:
                    first = False
                    continue

                split_row = row.split("\t")
                genes = sorted(split_row[:-1])
                pred_dict[genes[0]+","+genes[1]] = float(split_row[-1][:-1])

        with open(self.create_path() + "/pred_dict.bin", "wb") as f:
            pickle.dump(pred_dict, f)

        return pred_dict

    def predict(self, input: Union[ndarray[str],List[Tuple[str,str]]], gene_set: str) -> ndarray[float]:
        with open(self.create_path() + "/pred_dict.bin", "rb") as f:
            pred_dict = pickle.load(f)

        pred_out = []
        for pair in input:
            sorted_pair = sorted(pair)
            pair_str = sorted_pair[0] + "," + sorted_pair[1]
            if pair_str in pred_dict:
                pred_out.append(pred_dict[pair_str])
            else:
                pred_out.append(0)
        return pred_out


if __name__ == "__main__":
    model = DIEP()

