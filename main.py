import sys
import os

from node_model.parameters import tuned_parameters, top_models
from node_model.models.downstream import DownstreamModel
from node_model.models.digepred import DiGePred
from node_model.models.diep import DIEP
from utils.models import map_to_KGEM
from utils.operators import map_name_to_transform, Operator
from config.embedding_source_paths import ROOT
from stratified.clustering_utils import tune_num_clusters
from stratified.clustering_alg import KMeansAlg

model_type_dict = {
    "KGEM": DownstreamModel,
    "DIEP": DIEP,
    "DiGePred": DiGePred
}

training_parameters = [
    "KGE_model",
    "model_type",
    "classifier_type",
    "embedding_dim",
    "num_epochs",
    "classifier_kwargs",
    "embedding_transform",
    "model_path",
    "stratified"
]

evaluation_parameters = [
    "KGE_model",
    "model_type",
    "classifier_type",
    "embedding_dim",
    "num_epochs",
    "classifier_kwargs",
    "embedding_transform",
    "stratified",
    "model_path"
]

prediction_parameters = [
    "model_path",
    "gene_format",
    "input_file",
    "input_header"
]

independent_parameters = [
    "model_path"
]

clustering_parameters = [
    "model_path"
]


training_error = ("Command:\n"
                  "\t python3 main.py train\n"
                  "\t\t --model_type <str>\n"
                  "\t\t --KGE_model <str>\n"
                  "\t\t --model_path <str>\n"
                  "\t\t --embedding_dim <int>\n"
                  "\t\t --num_epochs <int>\n"
                  "\t\t --classifier_type <str>\n"
                  "\t\t --classifier_kwargs <int/tuple/str>\n"
                  "\t\t --embedding_transform <str>\n"
                  "Parameters:\n"
                  "\t model_type: The type of model. Can be \"KGEM\", \"DIEP\" or \"DiGePred\".\n"
                  "\t KGE_model: The name of the KGEM.\n"
                  "\t model_path: The full path to the binary file used to save the model. (Optional)\n"
                  "\t classifier_type: The type of the classifier. (Optional) Can be \"svm\" (Support Vector Machine), "
                  "\"forest\" (Balanced Random Forest) or \"mlp\" (Multilayer Perceptron).\n"
                  "\t embedding_dim: The embedding dimension. (Optional)\n"
                  "\t num_epochs: The number of epochs used to train the KGEM. (Optional)\n"
                  "\t classifier_kwargs: The parameter values of the classifier. (Optional) "
                  "For a Balanced Random Forest, provide an integer corresponding to the number of trees. "
                  "For a Multilayer Perceptron, provide a sequence of integers corresponding to the number of nodes in each hidden layer. "
                  "For a Support Vector Machine, provide the name of the kernel (\"poly\", \"linear\", \"rbf\", \"sigmoid\").\n"
                  "\t embedding_transform: The embedding transform operator to be used. (Optional) "
                  "Can be \"average\", \"hadamard\", \"weightedl1\", \"weightedl2\", \"rvis-concatenate\" or \"rand-concatenate\".\n\n"
                  "If not given, optional parameters are set to the values of the "
                  "top-performing pipelines of the KGEM or of the KGEM and "
                  "classifier type (if given)."
                  )

prediction_error = ("Command:\n"
                  "\t python3 main.py predict\n"
                  "\t\t --model_path <str>\n"
                  "\t\t --input_file <str>\n"
                  "\t\t --input_header <bool>\n"
                  "\t\t --gene_format <str>\n"
                  "Parameters:\n"
                  "\t model_path: \t\t The full path to the binary file used to save the model.\n"
                  "\t input_file: \t A tab-delimited file providing the gene pairs.\n"
                  "\t input_header: \t A boolean describing whether the input file contains a header.\n\t\t\t\t Default is set to False.\n"
                  "\t gene_format: \t\t The naming convention used for the genes. Can either be \"HGNC\" or \"Ensembl\".\n"
                    "\t\t\t\t Default is \"Ensembl\".\n")

evaluation_error = ("")

independent_error = ("")

clustering_error = ("")

def transform_parameter_values(args_dict, training):
    if training:
        if args_dict["model_type"] == DownstreamModel:
            if args_dict["KGE_model"] not in map_to_KGEM:
                return True
            else:
                args_dict["KGE_model"] = map_to_KGEM[args_dict["KGE_model"]]

            if args_dict["classifier_type"] not in ["forest", "svm", "mlp"]:
                return True

            if type(args_dict["embedding_transform"]) == str:
                if args_dict["embedding_transform"] not in map_name_to_transform:
                    return True
                else:
                    args_dict["embedding_transform"] = map_name_to_transform[args_dict["embedding_transform"]]
            else:
                if Operator not in type(args_dict["embedding_transform"]).__mro__:
                    return True


            if args_dict["classifier_type"] == "svm":
                if args_dict["classifier_kwargs"]["kernel"] not in ["linear", "poly", "rbf", "sigmoid"]:
                    return True

            if args_dict["classifier_type"] == "mlp":
                if type(args_dict["classifier_kwargs"]["hidden_layer_sizes"]) == list:
                    if any([type(val) != int for val in args_dict["classifier_kwargs"]["hidden_layer_sizes"]]):
                        if any([not val.isdigit() for val in args_dict["classifier_kwargs"]["hidden_layer_sizes"]]):
                            return True

                    args_dict["classifier_kwargs"]["hidden_layer_sizes"] = (int(val) for val in args_dict["classifier_kwargs"]["hidden_layer_sizes"])
                else:
                    if type(args_dict["classifier_kwargs"]["hidden_layer_sizes"]) != int:
                        if not args_dict["classifier_kwargs"]["hidden_layer_sizes"].isdigit():
                            return True

                    args_dict["classifier_kwargs"]["hidden_layer_sizes"] = (int(args_dict["classifier_kwargs"]["hidden_layer_sizes"]),)

            if args_dict["classifier_type"] == "forest":
                if type(args_dict["classifier_kwargs"]["n_estimators"]) != int:
                    if not args_dict["classifier_kwargs"]["n_estimators"].isdigit():
                        return True

                args_dict["classifier_kwargs"]["n_estimators"] = int(args_dict["classifier_kwargs"]["n_estimators"])

            for subpar in args_dict["embedding_kwargs"]:
                if type(args_dict["embedding_kwargs"][subpar]) != int:
                    if not args_dict["embedding_kwargs"][subpar].isdigit():
                        return True

                args_dict["embedding_kwargs"][subpar] = int(args_dict["embedding_kwargs"][subpar])

            if "stratified" in args_dict:
                if args_dict["stratified"] not in ["True", "False"]:
                    return True
                else:
                    bool_dict = {"True": True, "False": False}
                    args_dict["stratified"] = bool_dict[args_dict["stratified"]]

    if type(args_dict["model_path"]) != str:
        if args_dict["model_path"] != None:
            return True

    if "input_file" in args_dict:
        if type(args_dict["input_file"]) != str:
            return True

    if "input_header" in args_dict:
        if args_dict["input_header"] not in ["True", "False"]:
            return True
        else:
            bool_dict = {"True": True, "False": False}
            args_dict["input_header"] = bool_dict[args_dict["input_header"]]

    if "gene_format" in args_dict:
        if args_dict["gene_format"] not in ["Ensembl", "HGNC"]:
            return True


def resolve_model_parameters(args_dict, training):
    if training:
        if "model_type" not in args_dict:
            args_dict["model_type"] = DownstreamModel
        else:
            args_dict["model_type"] = model_type_dict[args_dict["model_type"]]

    if training and args_dict["model_type"] == DownstreamModel:
        if "classifier_type" not in args_dict:
            pars_dict = top_models[args_dict["KGE_model"]]
        else:
            pars_dict = tuned_parameters[args_dict["classifier_type"]][args_dict["KGE_model"]]

        for par in pars_dict:
            if par == "embedding_kwargs":
                args_dict[par] = {}
                for sub_par in pars_dict[par]:
                    if sub_par not in args_dict:
                        args_dict[par][sub_par] = pars_dict[par][sub_par]
                    else:
                        args_dict[par][sub_par] = args_dict[sub_par]
                        args_dict.pop(sub_par)

            if par not in args_dict:
                args_dict[par] = pars_dict[par]
            else:
                if par == "classifier_kwargs":
                    temp_dict = {}
                    for subpar in pars_dict[par]:
                        temp_dict[subpar] = args_dict[par]
                    args_dict[par] = temp_dict

    if "model_path" not in args_dict:
        args_dict["model_path"] = None


def write_config(args_dict, model_path):
    with open(os.path.dirname(model_path) + "/config.txt", "w") as f:
        for par in args_dict:
            if type(args_dict[par]) == dict:
                for subpar in args_dict[par]:
                    f.write(subpar + "\t" + str(args_dict[par][subpar]) + "\n")
            else:
                if type(args_dict[par]) not in [tuple, int, str]:
                    f.write(par + "\t" + args_dict[par].__name__ + "\n")
                else:
                    f.write(par + "\t" + str(args_dict[par]) + "\n")

def main():
    args = sys.argv

    args_dict = {}

    current_argument = None
    current_values = []
    if len(args) > 2:
        for i in range(2,len(args)):
            if args[i][:2] == "--":
                if current_argument is not None:
                    if len(current_values) == 1:
                        args_dict[current_argument] = current_values[0]
                    else:
                        args_dict[current_argument] = current_values

                current_argument = args[i][2:]
                current_values = []
            else:
                current_values.append(args[i])
        if current_argument is not None:
            if len(current_values) == 1:
                args_dict[current_argument] = current_values[0]
            else:
                args_dict[current_argument] = current_values


    if args[1] == "train":
        if any([par not in training_parameters for par in args_dict]) \
                or "KGE_model" not in args_dict:
            print("Incorrect parameters.")
            print(training_error)
            sys.exit()

        resolve_model_parameters(args_dict, True)
        error = transform_parameter_values(args_dict, True)

        if error:
            print("Incorrect parameters.")
            print(training_error)
            sys.exit()

        train(args_dict)

    if args[1] == "evaluate":
        if any([par not in evaluation_parameters for par in args_dict]) \
                or "model_path" not in args_dict:
            print("Incorrect parameters.")
            print(evaluation_error)
            sys.exit()

        resolve_model_parameters(args_dict, False)
        error = transform_parameter_values(args_dict, False)

        if error:
            print("Incorrect parameters.")
            print(evaluation_error)
            sys.exit()

        evaluate(args_dict)

    if args[1] == "predict":
        if any([par not in prediction_parameters for par in args_dict]) \
                or "model_path" not in args_dict:
            print("Incorrect parameters.")
            print(prediction_error)
            sys.exit()

        resolve_model_parameters(args_dict, False)
        error = transform_parameter_values(args_dict, False)

        if error:
            print("Incorrect parameters.")
            print(evaluation_error)
            sys.exit()

        predict(args_dict)

    if args[1] == "independent":
        if any([par not in independent_parameters for par in args_dict]) \
                or "model_path" not in args_dict:
            print("Incorrect parameters.")
            print(independent_error)
            sys.exit()

        resolve_model_parameters(args_dict, False)
        error = transform_parameter_values(args_dict, False)

        if error:
            print("Incorrect parameters.")
            print(independent_error)
            sys.exit()


def train(args_dict):
    model_path = args_dict.pop("model_path")
    model_type = args_dict.pop("model_type")

    model = model_type(**args_dict)
    model.full_training(True, model_path)

    write_config(args_dict, model.model_path)


def predict(args_dict):
    model_path = args_dict.pop("model_path")
    input_file = args_dict.pop("input_file")
    if "input_header" not in args_dict:
        args_dict["input_header"] = False
    header = args_dict.pop("input_header")
    if "gene_format" not in args_dict:
        args_dict["gene_format"] = "Ensembl"
    gene_format = args_dict.pop("gene_format")

    input = []
    with open(input_file, "r") as f:
        first = True
        for row in f:
            if header and first:
                first = False
                continue

            row_split = row.split("\t")
            input.append([row_split[0], row_split[1][:-1]])

    model = pickle.load(open(model_path, "rb"))
    model.full_training(model_path)

    if gene_format == "HGNC":
        new_input = model.map_gene_name_to_ensembl_id(input)

    pred_out = model.predict(new_input)

    dirname = os.path.dirname(model.model_path)
    basename = os.path.basename(input_file)
    basename = basename.split(".")[0]
    with open(dirname + "/" + basename + "_output.txt", "w") as f:
        for pair, pred in zip(input, pred_out):
            f.write(pair[0] + "\t" + pair[1] + "\t" + str(pred_out))

def evaluate(args_dict):
    model_path = args_dict["model_path"]
    args_dict.pop("model_path")

    model = pickle.load(open(model_path, "rb"))
    model.full_training(model_path=model_path)

    model.CV(model.model_path)


def independent(args_dict):
    model_path = args_dict.pop("model_path")

    model = pickle.load(open(model_path, "rb"))
    model.full_training(model_path=model_path)

    model.holdout_CV(True, model_path=model_path)
    model.holdout_vs_mono(model_path=model_path)



def generate_folds(args_dict):
    if "stratified" not in args_dict:
        print("Setting value of stratified to True.")
    else:
        if not args_dict["stratified"]:
            print("Setting value of stratified to True.")

    model_path = args_dict.pop("model_path")

    model = pickle.load(open(model_path, "rb"))

    clusterer_kwargs = {"model": model, "pathogenic": True}
    path_num = tune_num_clusters(clusterer_kwargs)

    clusterer = KMeansAlg(**clusterer_kwargs, num_clusters=path_num)
    clusterer.fit()
    clusterer.undersampling()
    clusterer.recombine_bins()
    clusterer.save_clusters()

    clusterer_kwargs = {"model": model, "pathogenic": False}
    neut_num = tune_num_clusters(clusterer_kwargs)

    clusterer = KMeansAlg(**clusterer_kwargs, num_clusters=neut_num)
    clusterer.fit()
    clusterer.undersampling()
    clusterer.recombine_bins()
    clusterer.save_clusters()



if __name__ == "__main__":
    main()