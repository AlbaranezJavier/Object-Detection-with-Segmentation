from tabulate import tabulate
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import f
import matplotlib.pyplot as plt
from pprint import pprint

"""
This script contains a code that analyzes a method showing a table of statistics.
"""


class Metrics:
    def __init__(self, p: float, label: list):
        self.confidence_interval_p = p
        self.label = label

        self.all_iou_correspondencies4det = []
        self.basic_stats = {"tp": 0, "tn": 0, "fn": 0, "fp": 0}
        self.complex_stats = {
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

    def cal_probseg_stats(self, predicted, gt, threshold):
        predicted = predicted > threshold
        gt = gt > threshold
        # variables
        _ones = np.ones_like(predicted)
        _predicted_zeros = np.logical_not(predicted)
        _gt_zeros = np.logical_not(gt)

        # TP, FN, FP, TN
        self.basic_stats["tp"] += int(np.sum(np.logical_and(predicted, gt)))
        self.basic_stats["fn"] += int(np.sum(np.logical_and(_predicted_zeros, gt)))
        self.basic_stats["fp"] += int(np.sum(np.logical_and(predicted, _gt_zeros)))
        self.basic_stats["tn"] += int(np.sum(np.logical_and(_predicted_zeros, _gt_zeros)))

        return self.basic_stats["tp"], self.basic_stats["fn"], self.basic_stats["fp"], self.basic_stats["tn"]

    def cal_seg_stats(self, predicted, gt):
        # variables
        _ones = np.ones_like(predicted)
        _predicted_zeros = np.logical_not(predicted)
        _gt_zeros = np.logical_not(gt)

        # TP, FN, FP, TN
        self.basic_stats["tp"] += int(np.sum(np.logical_and(predicted, gt)))
        self.basic_stats["fn"] += int(np.sum(np.logical_and(_predicted_zeros, gt)))
        self.basic_stats["fp"] += int(np.sum(np.logical_and(predicted, _gt_zeros)))
        self.basic_stats["tn"] += int(np.sum(np.logical_and(_predicted_zeros, _gt_zeros)))

        return self.basic_stats["tp"], self.basic_stats["fn"], self.basic_stats["fp"], self.basic_stats["tn"]

    def _get_labels4det(self, y):
        counter = 0
        while counter < len(y):
            if y[counter][0] != self.label[1]:
                y.pop(counter)
                counter -= 1
            counter += 1
        return y

    def cal_det_stats(self, predicted_raw, gt_raw):
        # variables
        tp = {}
        corresponds = []

        predicted = self._get_labels4det(predicted_raw)
        gt = self._get_labels4det(gt_raw)

        for g in range(len(gt)):
            corresponds.append([])
            for p in range(len(predicted)):
                if predicted[p][0] is None or predicted[p][0] == gt[g][0]:
                    iou = self._iou4det(predicted[p][1], predicted[p][2], gt[g][1], gt[g][2])
                    if iou > 0:
                        corresponds[g].append([p, iou])
        corresponds = self._sort_corresponds(corresponds)
        f = 0
        while f < len(corresponds):
            if len(corresponds[f]) > 0:
                if tp.get(corresponds[f][0][0]) is None:
                    tp[corresponds[f][0][0]] = [f, corresponds[f][0][1]]
                elif tp[corresponds[f][0][0]][1] < corresponds[f][0][1]:
                    corresponds[tp[corresponds[f][0][0]][0]].pop(0)
                    f, tp = -1, {}
                else:
                    corresponds[f].pop(0)
                    f -= 1
            f += 1
        for key in tp.keys():
            self.all_iou_correspondencies4det.append(tp[key][1])

        # TP, FN, FP, TN, IoU
        self.basic_stats["tp"] += len(tp)
        self.basic_stats["fn"] += len(gt) - len(tp)
        self.basic_stats["fp"] += len(predicted) - len(tp)
        self.basic_stats["tn"] = -1

        return self.basic_stats["tp"], self.basic_stats["fn"], self.basic_stats["fp"], self.all_iou_correspondencies4det

    def cal_cls_stats(self, predicted_idx, gt_idx):
        assert type(predicted_idx) == int
        assert type(gt_idx) == int

        if gt_idx != self.label[1]:
            if predicted_idx == self.label[1]:
                self.basic_stats["fn"] += 1
            elif predicted_idx != self.label[1]:
                self.basic_stats["tn"] += 1
            else:
                raise NameError("Caso no contemplado: "
                                f"predicted={predicted_idx}, gt={gt_idx} and label={self.label[1]}")
        else:
            if gt_idx == predicted_idx or predicted_idx == self.label[1]:
                self.basic_stats["tp"] += 1
            elif predicted_idx != gt_idx or predicted_idx != self.label[1]:
                self.basic_stats["fp"] += 1
            else:
                raise NameError(f"Caso no contemplado: "
                                f"predicted={predicted_idx}, gt={gt_idx} and label={self.label[1]}")


        return self.basic_stats["tp"], self.basic_stats["fn"], self.basic_stats["fp"], self.basic_stats["tn"]

    def _sort_corresponds(self, corresponds):
        for correspond in corresponds:
            correspond.sort(key=lambda x: x[1])
        return corresponds

    def _iou4det(self, pred_p1, pred_p2, gt_p1, gt_p2):
        p1_max = [max(pred_p1[0], gt_p1[0]), max(pred_p1[1], gt_p1[1])]
        p2_min = [min(pred_p2[0], gt_p2[0]), min(pred_p2[1], gt_p2[1])]
        inter_area = max(0, p2_min[0] - p1_max[0] + 1) * max(0, p2_min[1] - p1_max[1] + 1)
        pred_area = (pred_p2[0] - pred_p1[0] + 1) * (pred_p2[1] - pred_p1[1] + 1)
        gt_area = (gt_p2[0] - gt_p1[0] + 1) * (gt_p2[1] - gt_p1[1] + 1)
        iou = inter_area / float(pred_area + gt_area - inter_area)
        return iou

    def add4average(self, basic_stats):
        self.basic_stats["tp"] += basic_stats["tp"]
        self.basic_stats["fp"] += basic_stats["fp"]
        self.basic_stats["tn"] += basic_stats["tn"]
        self.basic_stats["fn"] += basic_stats["fn"]


    def cal_complex_stats(self):
        self.population()
        self.predicted_positive()
        self.predicted_negative()
        self.real_positive()
        self.real_negative()

        self.bias()
        self.inverse_bias()
        self.prevalence()
        self.null_error_rate()

        self.recall()
        self.false_negative_rate()
        self.false_positive_rate()
        self.specifity()
        self.hit_rate()
        self.incorrect_rejection_rate()
        self.delivered_error_rate()
        self.correct_rejection_rate()

        self.precision()
        self.false_negative_accuracy()
        self.false_discovery_rate()
        self.inverse_precision()

        self.accuracy()

        self.informedness()
        self.markedness()

        self.positive_likelihood_ratio()
        self.negative_likelihood_ratio()
        self.diagnostic_odds_ratio()

        self.chi_square()
        self.correlation()

        self.probability_random_agreement()
        self.matthews_correlation_coefficent()
        self.cohens_kappa()
        self.missclassification_rate()
        self.iou()
        self.f1_score()

    def print_resume(self):
        print(f"{self.label[0] : <10} => "
              f"Population:\033[92m{self.complex_stats['population'] : <6}\033[00m "
              f"Accuracy:\033[92m{self.complex_stats['accuracy'] : <6}\033[00m "
              f"IoU:\033[92m{self.complex_stats['iou'] : <6}\033[00m "
              f"Precision:\033[92m{self.complex_stats['precision'] : <6}\033[00m "
              f"Recall:\033[92m{self.complex_stats['recall'] : <6}\033[00m "
              f"F1 score:\033[92m{self.complex_stats['f1'] : <6}\033[00m")

    def print_table(self, tablefmt="grid"):
        print(f"==============>{self.label[0]}<==============")
        # Creating table
        table = [[f'Population\nN = TP+TN+FP+FN\n{self.complex_stats["population"]}', "",
                  f'Predicted Positive\nPP = TP+FP\n{self.complex_stats["pp"]}',
                  f'Bias\npp = PP/N\n{self.complex_stats["bias"]} {self.complex_stats["i_bias"]}%',
                  f'Predicted Negative\nPN = FN+TN\n{self.complex_stats["pn"]}',
                  f'Inverse Bias\npn = PN/N\n{self.complex_stats["ib"]} {self.complex_stats["i_ib"]}%', "", ""],
                 [],
                 [f'Real Positive\nRP = TP+FN\n{self.complex_stats["rp"]}', "", f'TP\n\n{self.basic_stats["tp"]}',
                  f'Recall/Sensitivity\ntpr = TP/RP\n{self.complex_stats["recall"]} {self.complex_stats["i_recall"]}%',
                  f'FN\n\n{self.basic_stats["fn"]}',
                  f'FNR\nfnr = FN/RP\n{self.complex_stats["fnr"]} {self.complex_stats["i_fnr"]}%', "",
                  f'LR+\nLR+ = tpr/fpr\n{self.complex_stats["lr+"]}'],
                 [f'Prevalence\nrp = RP/N\n{self.complex_stats["prevalence"]} {self.complex_stats["i_prevalence"]}%', "",
                  f'Precision\ntpa = TP/PP\n{self.complex_stats["precision"]} {self.complex_stats["i_precision"]}%',
                  f'Performance\ntp = TP/N\n{self.complex_stats["performance"]} {self.complex_stats["i_performance"]}%',
                  f'FN Accuracy\nfna = FN/PN\n{self.complex_stats["fna"]} {self.complex_stats["i_fna"]}%',
                  f'Incorrect Rejection Rate\n fn = FN/N \n{self.complex_stats["irr"]} {self.complex_stats["i_irr"]}%', "",
                  f'LR-\nLR- = fnr/tnr\n{self.complex_stats["lr-"]}'],
                 [],
                 [f'Real Negative\nRN = FP+TN\n{self.complex_stats["rn"]}', "", f'FP\n\n{self.basic_stats["fp"]}',
                  f'FPR\nfpr = FP/RN\n{self.complex_stats["fpr"]} {self.complex_stats["i_fpr"]}%',
                  f'TN\n\n{self.basic_stats["tn"]}',
                  f'Specifity\n = TN/RN \n{self.complex_stats["specifity"]} {self.complex_stats["i_specifity"]}%',
                  "",
                  f'Odds ratio\ndor = LR+/LR-\n{self.complex_stats["dor"]}'],
                 [f'Null Error Rate\nrn = RN/N\n{self.complex_stats["ner"]} {self.complex_stats["i_ner"]}%', "",
                  f'False Discovery Rate\nfdr = FP/PP\n{self.complex_stats["fdr"]} {self.complex_stats["i_fdr"]}%',
                  f'Delivered Error Rate\nfp = FP/N\n{self.complex_stats["der"]} {self.complex_stats["i_der"]}%',
                  f'Inverse Precision\ntna = TN/PN\n{self.complex_stats["ip"]} {self.complex_stats["i_ip"]}%',
                  f'Correct Rejection Rate\ntn = TN/N\n{self.complex_stats["crr"]} {self.complex_stats["i_crr"]}%', "",
                  f'Informedness\n = tpr - fpr\n{self.complex_stats["informedness"]}%'],
                 [],
                 ["", "", f'Chi square\n{self.complex_stats["chi"]}\np={self.complex_stats["p"]}',
                  f'Correlation\n(TP*TN - FP*FN)/(PP*RP*RN*PN)**.5\n{self.complex_stats["correlation"]} {self.complex_stats["i_correlation"]}',
                  f'Prob. Random Agreement\npra = (PP*RP + PN*RN)/(N)**2\n{self.complex_stats["pra"]}%',
                  f'Markedness\n= tpa - fna\n{self.complex_stats["markedness"]}%', "",
                  f'Accuracy\nacc = (TP + TN) / N\n{self.complex_stats["accuracy"]} {self.complex_stats["i_accuracy"]}%'],
                 ["", "", f'Matthews Corr. Coeff.\nmcc = (chi / N)**.5\n{self.complex_stats["mcc"]}',
                  f'IoU\n= TP/(N-TN)\n{self.complex_stats["iou"]} {self.complex_stats["i_iou"]}%',
                  f'Cohen Kappa\n= (acc-pra)/(1-pra)\n{self.complex_stats["ck"]}',
                  f'Misclassification Rate\nerr = (PF+FN)/N\n{self.complex_stats["mr"]} {self.complex_stats["i_mr"]}%', "",
                  f'F1 score\n= 2*TP/(RP+PP)\n{self.complex_stats["f1"]} {self.complex_stats["i_f1"]}%']]

        # Displaying table
        print(tabulate(table, tablefmt=tablefmt,
                       colalign=("center", "center", "center", "center", "center", "center", "center", "center")))
        print("")

    def population(self):
        if self.basic_stats["tn"] == -1:
            self.complex_stats["population"] = self.basic_stats["tp"] + self.basic_stats["fp"] + self.basic_stats["fn"]
        else:
            self.complex_stats["population"] = self.basic_stats["tp"] + self.basic_stats["tn"] + \
                                       self.basic_stats["fp"] + self.basic_stats["fn"]

    def predicted_positive(self):
        self.complex_stats["pp"] = self.basic_stats["tp"] + self.basic_stats["fp"]

    def bias(self):
        try:
            self.complex_stats["bias"] = round((self.complex_stats["pp"] / self.complex_stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.complex_stats["pp"], nobs=self.complex_stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_bias"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.complex_stats["bias"] = 0
            self.complex_stats["i_bias"] = f'[{0} - {0}]'

    def predicted_negative(self):
        if self.basic_stats["tn"] == -1:
            self.complex_stats["pn"] = -1
        else:
            self.complex_stats["pn"] = self.basic_stats["fn"] + self.basic_stats["tn"]

    def inverse_bias(self):
        try:
            self.complex_stats["ib"] = round((self.complex_stats["pn"] / self.complex_stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.complex_stats["pn"], nobs=self.complex_stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_ib"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.complex_stats["ib"] = 0
            self.complex_stats["i_ib"] = f'[{0} - {0}]'

    def real_positive(self):
        self.complex_stats["rp"] = self.basic_stats["tp"] + self.basic_stats["fn"]

    def recall(self):
        try:
            self.complex_stats["recall"] = round((self.basic_stats["tp"] / self.complex_stats["rp"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats["tp"], nobs=self.complex_stats["rp"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_recall"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.complex_stats["recall"] = 0
            self.complex_stats["i_recall"] = f'[{0} - {0}]'

    def false_negative_rate(self):
        try:
            self.complex_stats["fnr"] = round((self.basic_stats["fn"] / self.complex_stats["rp"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats["fn"], nobs=self.complex_stats["rp"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_fnr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.complex_stats["fnr"] = 0
            self.complex_stats["i_fnr"] = f'[{0} - {0}]'

    def positive_likelihood_ratio(self):
        try:
            self.complex_stats["lr+"] = round((self.complex_stats["recall"] / self.complex_stats["fpr"]), 2)
        except ZeroDivisionError:
            self.complex_stats["lr+"] = 0

    def prevalence(self):
        try:
            self.complex_stats["prevalence"] = round((self.complex_stats["rp"] / self.complex_stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.complex_stats["rp"], nobs=self.complex_stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_prevalence"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.complex_stats["prevalence"] = 0
            self.complex_stats["i_prevalence"] = f'[{0} - {0}]'

    def precision(self):
        try:
            self.complex_stats["precision"] = round((self.basic_stats["tp"] / self.complex_stats["pp"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats["tp"], nobs=self.complex_stats["pp"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_precision"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.complex_stats["precision"] = 0
            self.complex_stats["i_precision"] = f'[{0} - {0}]'

    def hit_rate(self):
        try:
            self.complex_stats["performance"] = round((self.basic_stats["tp"] / self.complex_stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats["tp"], nobs=self.complex_stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_performance"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.complex_stats["performance"] = 0
            self.complex_stats["i_performance"] = f'[{0} - {0}]'

    def false_negative_accuracy(self):
        try:
            self.complex_stats["fna"] = round((self.basic_stats["fn"] / self.complex_stats["pn"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats["fn"], nobs=self.complex_stats["pn"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_fna"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.complex_stats["fna"] = 0
            self.complex_stats["i_fna"] = f'[{0} - {0}]'

    def incorrect_rejection_rate(self):
        try:
            self.complex_stats["irr"] = round((self.basic_stats["fn"] / self.complex_stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats["fn"], nobs=self.complex_stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_irr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.complex_stats["irr"] = 0
            self.complex_stats["i_irr"] = f'[{0} - {0}]'

    def negative_likelihood_ratio(self):
        try:
            self.complex_stats["lr-"] = round((self.complex_stats["fnr"] / self.complex_stats["specifity"]), 2)
        except ZeroDivisionError:
            self.complex_stats["lr-"] = 0

    def real_negative(self):
        if self.basic_stats["tn"] == -1:
            self.complex_stats["rn"] = -1
        else:
            self.complex_stats["rn"] = self.basic_stats["fp"] + self.basic_stats["tn"]

    def false_positive_rate(self):
        try:
            self.complex_stats["fpr"] = round((self.basic_stats["fp"] / self.complex_stats["rn"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats["fp"], nobs=self.complex_stats["rn"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_fpr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.complex_stats["fpr"] = 0
            self.complex_stats["i_fpr"] = f'[{0} - {0}]'

    def specifity(self):
        if self.basic_stats["tn"] == -1:
            self.complex_stats["specifity"] = -1
            self.complex_stats["i_specifity"] = "det"
        else:
            try:
                self.complex_stats["specifity"] = round((self.basic_stats["tn"] / self.complex_stats["rn"]) * 100, 2)
                _temp = proportion_confint(count=self.basic_stats["tn"], nobs=self.complex_stats["rn"],
                                           alpha=self.confidence_interval_p, method="beta")
                self.complex_stats["i_specifity"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
            except ZeroDivisionError:
                self.complex_stats["specifity"] = 0
                self.complex_stats["i_specifity"] = f'ZeroDivision'

    def diagnostic_odds_ratio(self):
        try:
            self.complex_stats["dor"] = round((self.complex_stats["lr+"] / self.complex_stats["lr-"]), 2)
        except:
            self.complex_stats["dor"] = 0

    def null_error_rate(self):
        try:
            self.complex_stats["ner"] = round((self.complex_stats["rn"] / self.complex_stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.complex_stats["rn"], nobs=self.complex_stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_ner"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.complex_stats["ner"] = 0
            self.complex_stats["i_ner"] = f'[{0} - {0}]'

    def false_discovery_rate(self):
        try:
            self.complex_stats["fdr"] = round((self.basic_stats["fp"] / self.complex_stats["pp"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats["fp"], nobs=self.complex_stats["pp"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_fdr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except:
            self.complex_stats["fdr"] = 0
            self.complex_stats["i_fdr"] = f'[{0} - {0}]'

    def delivered_error_rate(self):
        try:
            self.complex_stats["der"] = round((self.basic_stats["fp"] / self.complex_stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats["fp"], nobs=self.complex_stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_der"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.complex_stats["der"] = 0
            self.complex_stats["i_der"] = f'[{0} - {0}]'

    def inverse_precision(self):
        if self.basic_stats["tn"] == -1:
            self.complex_stats["ip"] = -1
            self.complex_stats["i_ip"] = "det"
        else:
            try:
                self.complex_stats["ip"] = round((self.basic_stats["tn"] / self.complex_stats["pn"]) * 100, 2)
                _temp = proportion_confint(count=self.basic_stats["tn"], nobs=self.complex_stats["pn"],
                                           alpha=self.confidence_interval_p, method="beta")
                self.complex_stats["i_ip"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
            except:
                self.complex_stats["ip"] = 0
                self.complex_stats["i_ip"] = f'ZeroDivision'

    def correct_rejection_rate(self):
        if self.basic_stats["tn"] == -1:
            self.complex_stats["crr"] = -1
            self.complex_stats["i_crr"] = "det"
        else:
            self.complex_stats["crr"] = round((self.basic_stats["tn"] / self.complex_stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats["tn"], nobs=self.complex_stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_crr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'

    def informedness(self):
        self.complex_stats["informedness"] = round((self.complex_stats["recall"] - self.complex_stats["fpr"]), 2)

    def chi_square(self):
        if self.basic_stats["tn"] == -1:
            self.complex_stats["chi"] = -1
            self.complex_stats["p"] = "det"
        else:
            _temp11 = (self.complex_stats["rp"] * self.complex_stats["pp"]) / self.complex_stats["population"]
            _temp12 = (self.complex_stats["rp"] * self.complex_stats["pn"]) / self.complex_stats["population"]
            _temp21 = (self.complex_stats["rn"] * self.complex_stats["pp"]) / self.complex_stats["population"]
            _temp22 = (self.complex_stats["rn"] * self.complex_stats["pn"]) / self.complex_stats["population"]

            _temp11 = 0 if _temp11 == 0 else (self.basic_stats["tp"] - _temp11) ** 2 / _temp11
            _temp12 = 0 if _temp12 == 0 else (self.basic_stats["fp"] - _temp12) ** 2 / _temp12
            _temp21 = 0 if _temp21 == 0 else (self.basic_stats["fn"] - _temp21) ** 2 / _temp21
            _temp22 = 0 if _temp22 == 0 else (self.basic_stats["tn"] - _temp22) ** 2 / _temp22

            self.complex_stats["chi"] = round(_temp11 + _temp12 + _temp21 + _temp22, 2)
            self.complex_stats["p"] = round(1 - f.cdf(self.complex_stats["chi"], 1, 1000000000), 2)

    def correlation(self):
        if self.basic_stats["tn"] == -1:
            self.complex_stats["correlation"] = -1
            self.complex_stats["i_correlation"] = "det"
        else:
            try:
                self.complex_stats["correlation"] = round(((self.basic_stats["tp"] * self.basic_stats[
                    "tn"] + self.basic_stats["fp"] * self.basic_stats["fn"]) / (
                                                               self.complex_stats["pp"] * self.complex_stats["rp"] * self.complex_stats["rn"] *
                                                               self.complex_stats["pn"]) ** .5) * 100, 2)
                _temp = proportion_confint(
                    count=(self.basic_stats["tp"] * self.basic_stats["tn"] +
                           self.basic_stats["fp"] * self.basic_stats["fn"]),
                    nobs=(self.complex_stats["pp"] * self.complex_stats["rp"] * self.complex_stats["rn"] * self.complex_stats[
                        "pn"]) ** .5,
                    alpha=self.confidence_interval_p, method="beta")
                self.complex_stats["i_correlation"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
            except:
                self.complex_stats["correlation"] = 0
                self.complex_stats["i_correlation"] = f'ZeroDivision'

    def probability_random_agreement(self):
        try:
            self.complex_stats["pra"] = round(
                ((self.complex_stats["pp"] * self.complex_stats["rp"] + self.complex_stats["pn"] * self.complex_stats["rn"]) /
                 self.complex_stats["population"] ** 2) * 100, 2)
        except ZeroDivisionError:
            self.complex_stats["pra"] = 0
            self.complex_stats["i_pra"] = f'[{0} - {0}]'

    def markedness(self):
        self.complex_stats["markedness"] = round((self.complex_stats["precision"] - self.complex_stats["fna"]), 2)

    def accuracy(self):
        if self.basic_stats["tn"] == -1:
            self.complex_stats["accuracy"] = -1
            self.complex_stats["i_accuracy"] = "det"
        else:
            self.complex_stats["accuracy"] = round(
                ((self.basic_stats["tp"] + self.basic_stats["tn"]) / self.complex_stats[
                    "population"]) * 100, 2)
            _temp = proportion_confint(count=(self.basic_stats["tp"] + self.basic_stats["tn"]),
                                       nobs=self.complex_stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_accuracy"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'

    def matthews_correlation_coefficent(self):
        if self.complex_stats["chi"] == -1:
            self.complex_stats["mcc"] = -1
        else:
            self.complex_stats["mcc"] = round((self.complex_stats["chi"] / self.complex_stats["population"]) ** 0.5, 2)

    def iou(self):
        if self.basic_stats["tn"] == -1:
            self.complex_stats["iou"] = round(np.mean(self.all_iou_correspondencies4det) * 100, 2)
            self.complex_stats["i_iou"] = f'det'
        else:
            try:
                self.complex_stats["iou"] = round((self.basic_stats["tp"] / (
                            self.complex_stats["population"] - self.basic_stats["tn"])) * 100, 2)
                _temp = proportion_confint(count=self.basic_stats["tp"],
                                           nobs=self.complex_stats["population"] - self.basic_stats["tn"],
                                           alpha=self.confidence_interval_p, method="beta")
                self.complex_stats["i_iou"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
            except:
                self.complex_stats["iou"] = 0
                self.complex_stats["i_iou"] = f'ZeroDivision'

    def cohens_kappa(self):
        try:
            self.complex_stats["ck"] = round((self.complex_stats["accuracy"] - self.complex_stats["pra"]) / (100 - self.complex_stats["pra"]), 2)
        except ZeroDivisionError:
            self.complex_stats["ck"] = 0

    def missclassification_rate(self):
        try:
            self.complex_stats["mr"] = round((
                                             (self.basic_stats["fp"] + self.basic_stats["fn"]) /
                                             self.complex_stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats["fp"] + self.basic_stats["fn"],
                                       nobs=self.complex_stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_mr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.complex_stats["mr"] = 0
            self.complex_stats["i_mr"] = f'[{0} - {0}]'

    def f1_score(self):
        try:
            self.complex_stats["f1"] = round(
                (2 * self.basic_stats["tp"] / (self.complex_stats["rp"] + self.complex_stats["pp"])) * 100,
                2)
            _temp = proportion_confint(count=2 * self.basic_stats["tp"],
                                       nobs=self.complex_stats["rp"] + self.complex_stats["pp"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.complex_stats["i_f1"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except:
            self.complex_stats["f1"] = 0
            self.complex_stats["i_f1"] = f'[{0} - {0}]'

