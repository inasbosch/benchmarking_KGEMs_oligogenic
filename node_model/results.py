import matplotlib.pyplot as plt
import numpy as np
from statistics import stdev
import json
import os
from typing import Optional, Any
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import pickle


from node_model.utils.classifier_utils import scores, classify


class Results():
    def __init__(self,result_folder: str, fig_add: str = "", beta: float = -1,
                 overwrite: bool = False):

        self.result_folder = result_folder
        self.beta = beta
        self.fig_add = fig_add
        os.makedirs(self.result_folder, exist_ok=True)
        self.results_dict = {}
        self.exists = False

        if not overwrite and os.path.exists(self.result_folder + "/results_object" + self.fig_add + ".bin"):
            self.load()
            self.exists = True

    def update(self,output_test: np.ndarray[int], pred_out: np.ndarray[float], max_t: Optional[int] = None) -> None:
        pass

    def plot_CV(self):
        pass

    def _plot_CV(self,PR: bool):
        pass

    def save_results(self):
        with open(self.result_folder + "/results_object" + self.fig_add + ".bin", "wb") as f:
            pickle.dump(self, f)

    def load(self):
        print("Loading results from " + self.result_folder + "/results_object" + self.fig_add + ".bin")
        with open(self.result_folder + "/results_object" + self.fig_add + ".bin", "rb") as f:
            new_results = pickle.load(f)
            self.__dict__.update(new_results.__dict__)

class KFoldResultsRegressor(Results):
    def __init__(self, result_folder: str, fig_add: str = "", k: int = 10):
        super().__init__(result_folder, fig_add)
        self.k = k

        self.results_dict = {
            "MAE": [],
            "MSE": [],
            "R2": []
        }

        self.fig_add = fig_add

    def update(self,output_test: np.ndarray[int], pred_out: np.ndarray[float], max_t: Optional[int] = None) -> None:
        self.results_dict["MAE"].append(mean_absolute_error(output_test, pred_out))
        self.results_dict["MSE"].append(mean_squared_error(output_test, pred_out))
        self.results_dict["R2"].append(r2_score(output_test, pred_out))

    def save_results(self):
        to_add = []
        for key in self.results_dict:
            avg_val = np.mean(self.results_dict[key])
            to_add.append(("avg"+key, avg_val))

        for tup in to_add:
            self.results_dict[tup[0]] = tup[1]

        with open(self.result_folder+"/results"+self.fig_add+".json","w") as f:
            json.dump(self.results_dict,f)

        super().save_results()

class KFoldResults(Results):
    def __init__(self, result_folder: str, fig_add: str = "", beta: Optional[float] = -1, k: int = 10):
        super().__init__(result_folder, fig_add, beta)
        self.k = k

        if not self.exists:

            self.results_dict ={
                "TPR": [],
                "FPR": [],
                "ROCavg": 0,
                "ROCthresholds": [],

                "Precision": [],
                "Recall": [],
                "PRavg": 0,
                "PRthresholds": [],
            }

            self.PR_list = []
            self.ROC_list = []

        self.fig_add = fig_add

    def compute_std(self):
        self.ROC_list = []
        for tpr, fpr in zip(self.results_dict["TPR"], self.results_dict["FPR"]):
            self.ROC_list.append(auc(fpr, tpr))

        self.PR_list = []
        for precision, recall in zip(self.results_dict["Precision"], self.results_dict["Recall"]):
            self.PR_list.append(auc(np.flip(recall), np.flip(precision)))

    def update(self,output_test: np.ndarray[int], pred_out: np.ndarray[float],
               holdout_out: Optional[np.ndarray[str]] = None,
               max_t: Optional[int] = None) -> None:

        fpr, tpr, trsh = roc_curve(output_test, pred_out)
        self.results_dict["TPR"].append(tpr)
        self.results_dict["FPR"].append(fpr)
        self.results_dict["ROCthresholds"].append(trsh)
        ROC = auc(fpr, tpr)
        self.ROC_list.append(ROC)
        self.results_dict["ROCavg"] += ROC/self.k

        if holdout_out is not None:
            if self.beta == -1:
                best_trsh, _, _, _ = self.compute_best_threshold(tpr, fpr, trsh)

        precision, recall, trsh = precision_recall_curve(output_test, pred_out)
        self.results_dict["Precision"].append(np.flip(precision))
        self.results_dict["Recall"].append(np.flip(recall))
        self.results_dict["PRthresholds"].append(np.flip(trsh))
        PR = auc(recall, precision)
        self.PR_list.append(PR)
        self.results_dict["PRavg"] += PR/self.k


        if holdout_out is not None:
            if self.beta != -1:
                best_trsh, _, _, _ = self.compute_best_threshold(np.flip(recall), np.flip(precision), np.flip(trsh))

        if holdout_out is not None:
            if "holdout_avg" not in self.results_dict:
                self.results_dict["holdout_avg"] = 0
            round_out = classify(holdout_out, best_trsh)
            self.results_dict["holdout_avg"] += np.mean(round_out)/self.k

    def plot_CV(self):
        if self.beta == -1:
            self.results_dict["max_t"], self.results_dict["FPR_max"], self.results_dict["TPR_max"], best_ix = \
                self.get_best_threshold_interpolation()
            std_TPR, self.results_dict["mean_FPR"], self.results_dict["mean_TPR"], roc_threshold = \
                self.get_mean_interpolated_curve()
            std_Precision, self.results_dict["mean_Recall"], self.results_dict["mean_Precision"], pr_threshold = \
                self.get_mean_interpolated_curve(True)

            trsh_P = np.interp(np.flip(roc_threshold), np.flip(pr_threshold), np.flip(self.results_dict["mean_Precision"]))
            trsh_P = np.flip(trsh_P)
            #trsh_R = np.interp(np.flip(roc_threshold), np.flip(pr_threshold), np.flip(mean_Recall))
            self.results_dict["Precision_max"] = trsh_P[best_ix]
            self.results_dict["Recall_max"] = self.results_dict["TPR_max"]
        else:
            if self.beta is None:
                std_Precision, self.results_dict["mean_Recall"], self.results_dict["mean_Precision"], pr_threshold = self.get_mean_interpolated_curve(True)

                self.results_dict["max_t"] = 0.5
                self.results_dict["Precision_max"] = np.interp(0.5, pr_threshold, self.results_dict["mean_Precision"])
                self.results_dict["Recall_max"] = np.interp(0.5, pr_threshold, self.results_dict["mean_Recall"])

                std_TPR, self.results_dict["mean_FPR"], self.results_dict["mean_TPR"], roc_threshold = self.get_mean_interpolated_curve()
                self.results_dict["FPR_max"] = np.interp(0.5, roc_threshold, self.results_dict["mean_FPR"])
                self.results_dict["TPR_max"] = self.results_dict["Recall_max"]

            else:
                self.results_dict["max_t"], self.results_dict["Precision_max"], self.results_dict["Recall_max"], best_ix = \
                    self.get_best_threshold_interpolation_pr()
                std_TPR, self.results_dict["mean_FPR"], self.results_dict["mean_TPR"], roc_threshold = \
                    self.get_mean_interpolated_curve()
                std_Precision, self.results_dict["mean_Recall"], self.results_dict["mean_Precision"], pr_threshold = \
                    self.get_mean_interpolated_curve(True)

                trsh_FPR = np.interp(np.flip(pr_threshold), np.flip(roc_threshold), np.flip(self.results_dict["mean_FPR"]))
                #trsh_FPR = np.flip(trsh_FPR)
                self.results_dict["FPR_max"] = trsh_FPR[best_ix]
                self.results_dict["TPR_max"] = self.results_dict["Recall_max"]

        self.results_dict["low_std_TPR"] = self.results_dict["mean_TPR"] - std_TPR
        self.results_dict["high_std_TPR"] = self.results_dict["mean_TPR"] + std_TPR

        self.results_dict["low_std_Precision"] = self.results_dict["mean_Precision"] - std_Precision
        self.results_dict["high_std_Precision"] = self.results_dict["mean_Precision"] + std_Precision

        self.results_dict["TNR_max"] = 1 - self.results_dict["FPR_max"]
        self.results_dict["FNR_max"] = 1 - self.results_dict["TPR_max"]
        self.results_dict["NPV"] = self.results_dict["TNR_max"] / (
                self.results_dict["TNR_max"] + self.results_dict["FNR_max"])

        self.results_dict["f1score"] = ((2 * self.results_dict["Precision_max"] * self.results_dict["Recall_max"]) /
                                        (self.results_dict["Precision_max"] + self.results_dict["Recall_max"]))
        self.results_dict["gmean"] = sqrt(self.results_dict["TPR_max"] * self.results_dict["TNR_max"])

        self.results_dict["ROC_std"] = stdev(self.ROC_list)
        self.results_dict["PR_std"] = stdev(self.PR_list)

        self._plot_CV(False)
        self._plot_CV(True)

    def _plot_CV(self, PR: bool):
        if PR:
            plot_name = "/PR_threshold" + self.fig_add + ".png"
            xlabel = "Recall"
            ylabel = "Precision"
            legend = "AUC = " + str(round(self.results_dict["PRavg"], 3)) + ")"
            title = str(self.k) + "-fold CV mean PR curve"

            max_X = round(self.results_dict["Recall_max"],2)
            max_Y = round(self.results_dict["Precision_max"],2)
            mean_X = self.results_dict["mean_Recall"]
            mean_Y = self.results_dict["mean_Precision"]
            low_std = self.results_dict["low_std_Precision"]
            high_std = self.results_dict["high_std_Precision"]

            X = self.results_dict["Recall"]
            Y = self.results_dict["Precision"]
        else:
            plot_name = "/ROC_threshold" + self.fig_add + ".png"
            xlabel = "False Positive Rate"
            ylabel = "True Positive Rate"
            legend = "AUC = " + str(round(self.results_dict["ROCavg"], 3)) + ")"
            title = str(self.k) + "-fold CV mean ROC curve"

            max_X = round(self.results_dict["FPR_max"], 2)
            max_Y = round(self.results_dict["TPR_max"], 2)
            mean_X = self.results_dict["mean_FPR"]
            mean_Y = self.results_dict["mean_TPR"]
            low_std = self.results_dict["low_std_TPR"]
            high_std = self.results_dict["high_std_TPR"]

            X = self.results_dict["FPR"]
            Y = self.results_dict["TPR"]

        max_t = self.results_dict["max_t"]
        plt.figure(figsize=(6.4, 4.8))
        ax = plt.gca()
        i = 1
        for x, y in zip(X, Y):
            ax.plot(x, y, alpha=0.5, linewidth=0.75, label="CV " + str(i))
            i += 1

        ax.plot([max_X, max_X], [0, max_Y], color="grey", linestyle="--", linewidth=0.75)
        ax.plot([0, max_X], [max_Y, max_Y], color="grey", linestyle="--", linewidth=0.75)
        ax.plot(max_X, max_Y, "ko", label="Best threshold (" + str(round(max_t, 3)) + ")")
        ax.plot(mean_X, mean_Y, "b", label=legend)
        ax.plot([0, 0], [1, 1], "r", linestyle="--", linewidth=0.5)
        plt.xticks(list(plt.xticks()[0]) + [max_X])
        plt.yticks(list(plt.yticks()[0]) + [max_Y])
        xtick = ax.get_xticklabels()[-1]
        xtick.set_color("grey")
        xtick.set_y(xtick.get_position()[1] - 0.055)
        xlabels = [tick.get_text() for tick in ax.get_xticklabels()]
        for i in range(len(xlabels)):
            if len(xlabels[i]) > 4:
                xlabels[i] = xlabels[i][:-1]
            while len(xlabels[i]) <= 3:
                xlabels[i] += "0"
        ax.set_xticklabels(xlabels)
        ytick = ax.get_yticklabels()[-1]
        ytick.set_color("grey")
        ytick.set_x(ytick.get_position()[0] - 0.11)
        ylabels = [tick.get_text() for tick in ax.get_yticklabels()]
        for i in range(len(ylabels)):
            if len(ylabels[i]) > 4:
                ylabels[i] = ylabels[i][:-1]
            while len(ylabels[i]) <= 3:
                ylabels[i] += "0"
        ax.set_yticklabels(ylabels)
        ax.fill_between(mean_X, low_std, high_std, color="grey", alpha=0.5)
        if PR:
            ax.legend(loc=2, fontsize=8)
        else:
            ax.legend(fontsize=8)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.result_folder + plot_name)
        plt.clf()

    def get_mean_interpolated_curve(self, PR: bool = False):
        if PR:
            all_var1 = self.results_dict["Recall"]
            all_var2 = self.results_dict["Precision"]
            all_thresholds = self.results_dict["PRthresholds"]
        else:
            all_var1 = self.results_dict["FPR"]
            all_var2 = self.results_dict["TPR"]
            all_thresholds = self.results_dict["ROCthresholds"]

        mean_var1 = np.linspace(0, 1, 1000)
        interp_var2 = []
        interp_thresholds = []

        for var1, var2, thresholds in zip(all_var1, all_var2, all_thresholds):
            if len(var1) != len(thresholds):
                interp_temp = np.interp(mean_var1, var1[1:], var2[1:])
                interp_var2.append(interp_temp)
                interp_threshold = np.interp(mean_var1, var1[1:], thresholds)
            else:
                interp_temp = np.interp(mean_var1, var1, var2)
                interp_var2.append(interp_temp)
                interp_threshold = np.interp(mean_var1, var1, thresholds)
            interp_thresholds.append(interp_threshold)

        mean_var2 = np.mean(interp_var2, axis=0)
        mean_threshold = np.mean(interp_thresholds, axis=0)
        std_var2 = np.std(interp_var2, axis=0)
        return std_var2, mean_var1, mean_var2, mean_threshold

    def get_best_threshold_interpolation(self):
        _, mean_fpr, mean_tpr, mean_thresholds = self.get_mean_interpolated_curve()
        return self.compute_best_threshold(mean_tpr, mean_fpr, mean_thresholds)

    def compute_best_threshold(self, tpr, fpr, thresholds):
        tpr_fpr_product = tpr * (1 - fpr)
        best_ix = np.argmax(tpr_fpr_product)
        best_thr = thresholds[best_ix]
        best_tpr = tpr[best_ix]
        best_fpr = fpr[best_ix]
        return best_thr, best_fpr, best_tpr, best_ix

    def get_best_threshold_interpolation_pr(self):
        _, mean_r, mean_p, mean_thresholds = self.get_mean_interpolated_curve(True)
        return self.compute_best_threshold_pr(mean_r, mean_p, mean_thresholds)

    def compute_best_threshold_pr(self, r, p, thresholds):
        f1 = (1 + self.beta ** 2) * r * p / (self.beta ** 2 * p + r)
        best_ix = np.argmax(f1)
        best_thr = thresholds[best_ix]
        best_p = p[best_ix]
        best_r = r[best_ix]
        return best_thr, best_p, best_r, best_ix

    def save_results(self):
        self.plot_CV()

        result_dict = {
            "threshold": self.results_dict["max_t"],
            "recall/TPR": self.results_dict["TPR_max"],
            "TNR": self.results_dict["TNR_max"],
            "precision": self.results_dict["Precision_max"],
            "FPR": self.results_dict["FPR_max"],
            "FNR": self.results_dict["FNR_max"],
            "ROC": self.results_dict["ROCavg"],
            "PR": self.results_dict["PRavg"],
            "NPV": self.results_dict["NPV"],
            "f1score": self.results_dict["f1score"],
            "gmean": self.results_dict["gmean"],
            "ROC_std": self.results_dict["ROC_std"],
            "PR_std": self.results_dict["PR_std"]
        }

        if "holdout_avg" in self.results_dict:
            result_dict["holdout_avg"] = self.results_dict["holdout_avg"]

        with open(self.result_folder+"/results"+self.fig_add+".json","w") as f:
            json.dump(result_dict,f)

        super().save_results()

class SingleResults(Results):
    def __init__(self, results_folder: str, fig_add: str = "") -> None:
        super().__init__(results_folder, fig_add)
        self.fig_add = fig_add

    def update(self,output_test: np.ndarray[Any,np.dtype[Any]], pred_out: np.ndarray[Any,np.dtype[Any]], max_t: Optional[int] = None) -> None:
        pred_round = classify(pred_out,max_t)
        (self.results_dict["maxTPR"], self.results_dict["maxFPR"],
         self.results_dict["maxRecall"], self.results_dict["maxPrecision"]) = scores(output_test,pred_round)

        self.results_dict["FPR"], self.results_dict["TPR"], _ = roc_curve(output_test, pred_out)
        self.results_dict["ROC"] = auc(self.results_dict["FPR"], self.results_dict["TPR"])

        self.results_dict["Precision"], self.results_dict["Recall"], _ = precision_recall_curve(output_test, pred_out)
        self.results_dict["PR"] = auc(self.results_dict["Recall"], self.results_dict["Precision"])

        self.results_dict["maxTNR"] = 1 - self.results_dict["maxFPR"]
        self.results_dict["maxFNR"] = 1 - self.results_dict["maxTPR"]
        self.results_dict["NPV"] = self.results_dict["maxTNR"] / (
                    self.results_dict["maxTNR"] + self.results_dict["maxFNR"])

        if self.results_dict["maxPrecision"] == 0 and self.results_dict["maxRecall"] == 0:
            self.results_dict["f1score"] = 0
        else:
            self.results_dict["f1score"] = ((2 * self.results_dict["maxPrecision"]*self.results_dict["maxRecall"])/
                                        (self.results_dict["maxPrecision"]+self.results_dict["maxRecall"]))
        self.results_dict["gmean"] = sqrt(self.results_dict["maxTPR"]*self.results_dict["maxTNR"])

    def save_results(self, max_t: Optional[int] = None):
        self.plot_CV()

        result_dict = {
            "threshold": max_t,
            "recall/TPR": self.results_dict["maxTPR"],
            "TNR": self.results_dict["maxTNR"],
            "FNR": self.results_dict["maxFNR"],
            "precision": self.results_dict["maxPrecision"],
            "f1score": self.results_dict["f1score"],
            "gmean": self.results_dict["gmean"],
            "NPV": self.results_dict["NPV"],
            "FPR": self.results_dict["maxFPR"],
            "ROC": self.results_dict["ROC"],
            "PR": self.results_dict["PR"]
        }
        print("Saving results at " + self.result_folder+"/results"+self.fig_add+".json")
        with open(self.result_folder+"/results"+self.fig_add+".json","w") as f:
            json.dump(result_dict,f)

        super().save_results()

    def plot_CV(self):
        self._plot_CV(True)
        self._plot_CV(False)

    def _plot_CV(self, PR: bool):
        if PR:
            xlabel = "Recall"
            ylabel = "Precision"
            val = self.results_dict["PR"]
            title = "Area under the PR curve: " + str(val)

            X = self.results_dict["Recall"]
            Y = self.results_dict["Precision"]
            max_X = self.results_dict["maxRecall"]
            max_Y = self.results_dict["maxPrecision"]
            figname = "/pr_curve"+self.fig_add+".png"
        else:
            xlabel = "False Positive Rate"
            ylabel = "True Positive Rate"
            val = self.results_dict["ROC"]
            title = "Area under the ROC curve: " + str(val)

            X = self.results_dict["FPR"]
            Y = self.results_dict["TPR"]
            max_X = self.results_dict["maxFPR"]
            max_Y = self.results_dict["maxTPR"]
            figname = "/roc_curve"+self.fig_add+".png"



        plt.figure(figsize=(6.4, 4.8))
        plt.plot(X, Y)
        plt.plot([0, max_X], [max_Y, max_Y], color="k", linestyle="--")
        plt.plot([max_X, max_X], [0, max_Y], color="k", linestyle="--")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axline((0, 0), slope=1, color="r", linestyle="--", linewidth="0.5")
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.savefig(self.result_folder + figname)
        plt.clf()
        plt.close("all")
