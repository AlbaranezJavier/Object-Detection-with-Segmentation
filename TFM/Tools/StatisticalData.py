from tabulate import tabulate
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import f

"""
This script contains a code that analyzes a method showing a table of statistics.
"""


class Statistics:

    def __init__(self, p):
        self.confidence_interval_p = p
        self.basic_stats = {"basic":{"tp": 0, "tn": 0, "fn": 0, "fp":0},
                            "cumulative":{"tp": 0, "tn": 0, "fn": 0, "fp":0}}
        self.stats = {
            "population": 0, "pp": 0, "bias": 0, "pn": 0, "ib": 0, "rp": 0, "tp": 0, "recall": 0, "fn": 0, "fnr": 0,
            "lr+": 0, "prevalence": 0, "precision": 0, "performance": 0, "fna": 0, "irr": 0, "lr-": 0, "rn": 0, "fp": 0,
            "fpr": 0, "tn": 0, "specifity": 0, "dor": 0, "ner": 0, "fdr": 0, "der": 0, "ip": 0, "crr": 0,
            "informedness": 0, "chi": 0, "correlation": 0, "pra": 0, "markedness": 0, "accuracy": 0, "mcc": 0, "iou": 0,
            "ck": 0, "mr": 0, "f1": 0,
            "i_recall": [0, 0], "i_fnr": 0, "i_precision": 0, "i_performance": 0, "i_fna": 0,
            "i_irr": 0, "i_bias": 0, "i_ib": 0, "i_prevalence": 0, "i_fpr": 0, "i_tnr": 0, "i_ner": 0, "i_fdr": 0,
            "i_der": 0, "i_ip": 0, "i_crr": 0, "i_correlation": 0, "i_accuracy": 0, "i_iou": 0, "i_mr": 0, "i_f1": 0,
            "i_specifity": 0, "p": 0
        }

    def cal_basic_stats(self, predicted, gt):
        """
        Calculates tp, fn, fp and tn for a predicted mask and its ground thruth
        :param predicted: mask
        :param gt: mask
        :return: tp, fn, fp, tn
        """
        # variables
        _ones = np.ones_like(predicted)
        _predicted_zeros = np.logical_not(predicted)
        _gt_zeros = np.logical_not(gt)

        # TP, FN, FP, TN
        self.basic_stats["basic"]["tp"] = np.sum(np.logical_and(predicted, gt))
        self.basic_stats["basic"]["fn"] = np.sum(np.logical_and(_predicted_zeros, gt))
        self.basic_stats["basic"]["fp"] = np.sum(np.logical_and(predicted, _gt_zeros))
        self.basic_stats["basic"]["tn"] = np.sum(np.logical_and(_predicted_zeros, _gt_zeros))

        return self.basic_stats["basic"]["tp"], self.basic_stats["basic"]["fn"], self.basic_stats["basic"]["fp"], self.basic_stats["basic"]["tn"]

    def update_cumulative_stats(self):
        """
        Add current tp, tn, fp, fn to cumulative stats
        :return: None
        """
        self.basic_stats["cumulative"]["tp"] += self.basic_stats["basic"]["tp"]
        self.basic_stats["cumulative"]["fn"] += self.basic_stats["basic"]["fn"]
        self.basic_stats["cumulative"]["fp"] += self.basic_stats["basic"]["fp"]
        self.basic_stats["cumulative"]["tn"] += self.basic_stats["basic"]["tn"]

    def test(self, tp, fn, fp, tn):
        """
        Initializes some values for tp, fn, fp, tn
        :param tp: true positive
        :param fn: false negative
        :param fp: false positive
        :param tn: true negative
        :return:
        """
        # TP, FN, FP, TN
        self.basic_stats["basic"]["tp"] = tp
        self.basic_stats["basic"]["fn"] = fn
        self.basic_stats["basic"]["fp"] = fp
        self.basic_stats["basic"]["tn"] = tn

    def cal_complex_stats(self, basic_type):
        """
        Calculates the metrics of:
        Population, Predicted Positive, Predicted Negative, Real Positive, Real Negative
        Bias, Inverse Bias, Prevalence, Null Error Rate
        Recall, FNR, FPR, FNR,
        Performance, Incorrect Rejection rate, Delivered Error rate, Correct Rejection Rate
        Precision, False Negative Rate, False Discovery Rate, Inverse Precision
        Accuracy
        Informedness, Markedness
        LR+, LR-, Diagnostic Odds Ratio
        Correlation
        Probability of random agreement, Matthews corr. coeff., IoU, Cohen's Kappa, Misclasification rate, f1 score
        :return: None
        """
        # Population, Predicted Positive, Predicted Negative, Real Positive, Real Negative
        self.stats["population"] = self.basic_stats[basic_type]["tp"] + self.basic_stats[basic_type]["tn"] + self.basic_stats[basic_type]["fp"] + self.basic_stats[basic_type]["fn"]
        self.stats["pp"] = self.basic_stats[basic_type]["tp"] + self.basic_stats[basic_type]["fp"]
        self.stats["pn"] = self.basic_stats[basic_type]["fn"] + self.basic_stats[basic_type]["tn"]
        self.stats["rp"] = self.basic_stats[basic_type]["tp"] + self.basic_stats[basic_type]["fn"]
        self.stats["rn"] = self.basic_stats[basic_type]["fp"] + self.basic_stats[basic_type]["tn"]

        # Bias, Inverse Bias, Prevalence, Null Error Rate
        self.stats["bias"] = np.around((self.stats["pp"] / self.stats["population"]) * 100, 2)
        _temp = proportion_confint(count=self.stats["pp"], nobs=self.stats["population"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_bias"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        self.stats["ib"] = np.around((self.stats["pn"] / self.stats["population"]) * 100, 2)
        _temp = proportion_confint(count=self.stats["pn"], nobs=self.stats["population"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_ib"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        self.stats["prevalence"] = np.around((self.stats["rp"] / self.stats["population"]) * 100, 2)
        _temp = proportion_confint(count=self.stats["rp"], nobs=self.stats["population"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_prevalence"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        self.stats["ner"] = np.around((self.stats["rn"] / self.stats["population"]) * 100, 2)
        _temp = proportion_confint(count=self.stats["rn"], nobs=self.stats["population"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_ner"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'

        # Recall, FNR, FPR, FNR,
        self.stats["recall"] = np.around((self.basic_stats[basic_type]["tp"] / self.stats["rp"]) * 100, 2)
        _temp = proportion_confint(count=self.basic_stats[basic_type]["tp"], nobs=self.stats["rp"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_recall"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        self.stats["fnr"] = np.around((self.basic_stats[basic_type]["fn"] / self.stats["rp"]) * 100, 2)
        _temp = proportion_confint(count=self.basic_stats[basic_type]["fn"], nobs=self.stats["rp"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_fnr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        self.stats["fpr"] = np.around((self.basic_stats[basic_type]["fp"] / self.stats["rn"]) * 100, 2)
        _temp = proportion_confint(count=self.basic_stats[basic_type]["fp"], nobs=self.stats["rn"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_fpr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        self.stats["specifity"] = np.around((self.basic_stats[basic_type]["tn"] / self.stats["rn"]) * 100, 2)
        _temp = proportion_confint(count=self.basic_stats[basic_type]["tn"], nobs=self.stats["rn"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_specifity"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'

        # Performance, Incorrect Rejection rate, Delivered Error rate, Correct Rejection Rate
        self.stats["performance"] = np.around((self.basic_stats[basic_type]["tp"] / self.stats["population"]) * 100, 2)
        _temp = proportion_confint(count=self.basic_stats[basic_type]["tp"], nobs=self.stats["population"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_performance"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        self.stats["irr"] = np.around((self.basic_stats[basic_type]["fn"] / self.stats["population"]) * 100, 2)
        _temp = proportion_confint(count=self.basic_stats[basic_type]["fn"], nobs=self.stats["population"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_irr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        self.stats["der"] = np.around((self.basic_stats[basic_type]["fp"] / self.stats["population"]) * 100, 2)
        _temp = proportion_confint(count=self.basic_stats[basic_type]["fp"], nobs=self.stats["population"], alpha=self.confidence_interval_p, method="beta")
        self.stats["i_der"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        self.stats["crr"] = np.around((self.basic_stats[basic_type]["tn"] / self.stats["population"]) * 100, 2)
        _temp = proportion_confint(count=self.basic_stats[basic_type]["tn"], nobs=self.stats["population"], alpha=self.confidence_interval_p, method="beta")
        self.stats["i_crr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'

        #  Precision, False Negative Rate, False Discovery Rate, Inverse Precision
        self.stats["precision"] = np.around((self.basic_stats[basic_type]["tp"] / self.stats["pp"]) * 100, 2)
        _temp = proportion_confint(count=self.basic_stats[basic_type]["tp"], nobs=self.stats["pp"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_precision"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        self.stats["fna"] = np.around((self.basic_stats[basic_type]["fn"] / self.stats["pn"]) * 100, 2)
        _temp = proportion_confint(count=self.basic_stats[basic_type]["fn"], nobs=self.stats["pn"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_fna"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        self.stats["fdr"] = np.around((self.basic_stats[basic_type]["fp"] / self.stats["pp"]) * 100, 2)
        _temp = proportion_confint(count=self.basic_stats[basic_type]["fp"], nobs=self.stats["pp"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_fdr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        self.stats["ip"] = np.around((self.basic_stats[basic_type]["tn"] / self.stats["pn"]) * 100, 2)
        _temp = proportion_confint(count=self.basic_stats[basic_type]["tn"], nobs=self.stats["pn"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_ip"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'

        # Accuracy
        self.stats["accuracy"] = np.around(
            ((self.basic_stats[basic_type]["tp"] + self.basic_stats[basic_type]["tn"]) / self.stats["population"]) * 100, 2)
        _temp = proportion_confint(count=(self.basic_stats[basic_type]["tp"] + self.basic_stats[basic_type]["tn"]),
                                   nobs=self.stats["population"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_accuracy"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'

        # Informedness, Markedness
        self.stats["informedness"] = np.around((self.stats["recall"] - self.stats["fpr"]), 2)
        self.stats["markedness"] = np.around((self.stats["precision"] - self.stats["fna"]), 2)

        # LR+, LR-, Diagnostic Odds Ratio
        self.stats["lr+"] = np.around((self.stats["recall"] / self.stats["fpr"]), 2)
        self.stats["lr-"] = np.around((self.stats["fnr"] / self.stats["specifity"]), 2)
        self.stats["dor"] = np.around((self.stats["lr+"] / self.stats["lr-"]), 2)

        # Chi square
        _temp11 = (self.stats["rp"] * self.stats["pp"]) / self.stats["population"]
        _temp12 = (self.stats["rp"] * self.stats["pn"]) / self.stats["population"]
        _temp21 = (self.stats["rn"] * self.stats["pp"]) / self.stats["population"]
        _temp22 = (self.stats["rn"] * self.stats["pn"]) / self.stats["population"]

        _temp11 = (self.basic_stats[basic_type]["tp"] - _temp11) ** 2 / _temp11
        _temp12 = (self.basic_stats[basic_type]["fp"] - _temp12) ** 2 / _temp12
        _temp21 = (self.basic_stats[basic_type]["fn"] - _temp21) ** 2 / _temp21
        _temp22 = (self.basic_stats[basic_type]["tn"] - _temp22) ** 2 / _temp22

        self.stats["chi"] = np.around(_temp11 + _temp12 + _temp21 + _temp22, 2)
        self.stats["p"] = np.around(1 - f.cdf(self.stats["chi"], 1, 1000000000), 2)

        # Correlation
        self.stats["correlation"] = np.around(
            (self.basic_stats[basic_type]["tp"] * self.basic_stats[basic_type]["tn"] + self.basic_stats[basic_type]["fp"] * self.basic_stats[basic_type]["fn"]) / (
                    self.stats["pp"] * self.stats["rp"] * self.stats["rn"] * self.stats[
                "pn"]) ** .5, 2)
        _temp = proportion_confint(
                count=(self.basic_stats[basic_type]["tp"] * self.basic_stats[basic_type]["tn"] + self.basic_stats[basic_type]["fp"] * self.basic_stats[basic_type]["fn"]),
                nobs=(self.stats["pp"] * self.stats["rp"] * self.stats["rn"] * self.stats[
                    "pn"]) ** .5,
                alpha=self.confidence_interval_p, method="beta")
        self.stats["i_correlation"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'

        # Probability of random agreement, Matthews corr. coeff., IoU, Cohen's Kappa, Misclasification rate, f1 score
        self.stats["pra"] = np.around(
            (self.stats["pp"] * self.stats["rp"] + self.stats["pn"] * self.stats["rn"]) /
            self.stats["population"] ** 2, 2) * 100
        self.stats["mcc"] = np.around((self.stats["chi"] / self.stats["population"]) ** 0.5, 2)
        self.stats["ck"] = np.around(
            (self.stats["accuracy"] - self.stats["pra"]) / (100 - self.stats["pra"]), 2)
        self.stats["mr"] = np.around(
            (self.basic_stats[basic_type]["fp"] + self.basic_stats[basic_type]["fn"]) / self.stats["population"], 2) * 100
        _temp = proportion_confint(count=self.basic_stats[basic_type]["fp"] + self.basic_stats[basic_type]["fn"], nobs=self.stats["population"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_mr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        self.stats["iou"] = np.around(
            self.basic_stats[basic_type]["tp"] / (self.stats["population"] - self.basic_stats[basic_type]["tn"]), 2) * 100
        _temp = proportion_confint(count=self.basic_stats[basic_type]["tp"], nobs=self.stats["population"] - self.basic_stats[basic_type]["tn"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_iou"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        self.stats["f1"] = np.around(2 * self.basic_stats[basic_type]["tp"] / (self.stats["rp"] + self.stats["pp"]),
                                     2)
        _temp = proportion_confint(count=2 * self.basic_stats[basic_type]["tp"], nobs=self.stats["rp"] + self.stats["pp"],
                                   alpha=self.confidence_interval_p, method="beta")
        self.stats["i_f1"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'

    def print_table(self, basic_type):
        """
        Draws a table with all the metrics
        :return:
        """
        # Creating table
        table = [[f'Population\nN = TP+TN+FP+FN\n{self.stats["population"]}', "", f'Predicted Positive\nPP = TP+FP\n{self.stats["pp"]}',
                  f'Bias\npp = PP/N\n{self.stats["bias"]} {self.stats["i_bias"]}%', f'Predicted Negative\nPN = FN+TN\n{self.stats["pn"]}',
                  f'Inverse Bias\npn = PN/N\n{self.stats["ib"]} {self.stats["i_ib"]}%', "", ""],
                 [],
                 [f'Real Positive\nRP = TP+FN\n{self.stats["rp"]}', "", f'TP\n\n{self.basic_stats[basic_type]["tp"]}',
                  f'Recall\ntpr = TP/RP\n{self.stats["recall"]} {self.stats["i_recall"]}%',
                  f'FN\n\n{self.basic_stats[basic_type]["fn"]}', f'FNR\nfnr = FN/RP\n{self.stats["fnr"]} {self.stats["i_fnr"]}%', "",
                  f'LR+\nLR+ = tpr/fpr\n{self.stats["lr+"]}'],
                 [f'Prevalence\nrp = RP/N\n{self.stats["prevalence"]} {self.stats["i_prevalence"]}%', "",
                  f'Precision\ntpa = TP/PP\n{self.stats["precision"]} {self.stats["i_precision"]}%',
                  f'Performance\ntp = TP/N\n{self.stats["performance"]} {self.stats["i_performance"]}%',
                  f'FN Accuracy\nfna = FN/PN\n{self.stats["fna"]} {self.stats["i_fna"]}%',
                  f'Incorrect Rejection Rate\nfn = FN/N\n{self.stats["irr"]} {self.stats["i_irr"]}%', "",
                  f'LR-\nLR- = fnr/tnr\n{self.stats["lr-"]}'],
                 [],
                 [f'Real Negative\nRN = FP+TN\n{self.stats["rn"]}', "", f'FP\n\n{self.basic_stats[basic_type]["fp"]}',
                  f'FPR\nfpr = FP/RN\n{self.stats["fpr"]} {self.stats["i_fpr"]}%',
                  f'TN\n\n{self.basic_stats["basic"]["tn"]}', f'Specbasic_typetnr = TN/RN\n{self.stats["specifity"]} {self.stats["i_specifity"]}%',
                  "",
                  f'Odds ratio\ndor = LR+/LR-\n{self.stats["dor"]}'],
                 [f'Null Error Rate\nrn = RN/N\n{self.stats["ner"]} {self.stats["i_ner"]}%', "",
                  f'False Discovery Rate\nfdr = FP/PP\n{self.stats["fdr"]} {self.stats["i_fdr"]}%',
                  f'Delivered Error Rate\nfp = FP/N\n{self.stats["der"]} {self.stats["i_der"]}%',
                  f'Inverse Precision\ntna = TN/PN\n{self.stats["ip"]} {self.stats["i_ip"]}%',
                  f'Correct Rejection Rate\ntn = TN/N\n{self.stats["crr"]} {self.stats["i_crr"]}%', "",
                  f'Informedness\n = tpr - fpr\n{self.stats["informedness"]}%'],
                 [],
                 ["", "", f'Chi square\n{self.stats["chi"]}\np={self.stats["p"]}',
                  f'Correlation\n(TP*TN - FP*FN)/(PP*RP*RN*PN)**.5\n{self.stats["correlation"]} {self.stats["i_correlation"]}',
                  f'Prob. Random Agreement\npra = (PP*RP + PN*RN)/(N)**2\n{self.stats["pra"]}%', f'Markedness\n= tpa - fna\n{self.stats["markedness"]}%', "",
                  f'Accuracy\nacc = (TP + TN) / N\n{self.stats["accuracy"]} {self.stats["i_accuracy"]}%'],
                 ["", "", f'Matthews Corr. Coeff.\nmcc = (chi / N)**.5\n{self.stats["mcc"]}',
                  f'IoU\n= TP/(N-TN)\n{self.stats["iou"]} {self.stats["i_iou"]}%',
                  f'Cohen Kappa\n= (acc-pra)/(1-pra)\n{self.stats["ck"]}',
                  f'Misclassification Rate\nerr = (PF+FN)/N\n{self.stats["mr"]} {self.stats["i_mr"]}%', "",
                  f'F1 score\n= 2*TP/(RP+PP)\n{self.stats["f1"]} {self.stats["i_f1"]}%']]

        # Displaying table
        print(tabulate(table, tablefmt='grid',
                       colalign=("center", "center", "center", "center", "center", "center", "center", "center")))


if __name__ == '__main__':
    s = Statistics(p=0.01)
    s.test(50, 20, 10, 200)
    s.cal_complex_stats()
    s.print_table()
