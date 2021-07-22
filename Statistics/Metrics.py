from tabulate import tabulate
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import f

"""
This script contains a code that analyzes a method showing a table of statistics.
"""


class Metrics:

    def __init__(self, p, stats_type):
        self.confidence_interval_p = p
        self.all_iou_correspondencies4det = []
        self.basic_stats = {"basic": {"tp": 0, "tn": 0, "fn": 0, "fp": 0},
                            "cumulative": {"tp": 0, "tn": 0, "fn": 0, "fp": 0}}
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
        self.stats_type = stats_type
        assert self.stats_type == "det" or self.stats_type == "seg", \
            "Metrics, init: stats_type must be 'det' or 'seg'"

    def _cal_seg_stats(self, predicted, gt):
        """
        Calculates tp, fn, fp and tn for a predicted mask and its ground truth
        :param predicted: mask
        :param gt: mask
        :return: tp, fn, fp, tn
        """
        # variables
        _ones = np.ones_like(predicted)
        _predicted_zeros = np.logical_not(predicted)
        _gt_zeros = np.logical_not(gt)

        # TP, FN, FP, TN
        self.basic_stats["basic"]["tp"] = int(np.sum(np.logical_and(predicted, gt)))
        self.basic_stats["basic"]["fn"] = int(np.sum(np.logical_and(_predicted_zeros, gt)))
        self.basic_stats["basic"]["fp"] = int(np.sum(np.logical_and(predicted, _gt_zeros)))
        self.basic_stats["basic"]["tn"] = int(np.sum(np.logical_and(_predicted_zeros, _gt_zeros)))

        return self.basic_stats["basic"]["tp"], self.basic_stats["basic"]["fn"], self.basic_stats["basic"]["fp"], \
               self.basic_stats["basic"]["tn"], []

    def _cal_det_stats(self, predicted, gt):
        """
        Calculates tp, fn, fp and tn for a predicted detections and its ground truth
        :param predicted: detections
        :param gt: detections
        :return: tp, fn, fp, tn
        """
        # variables
        tp = {}
        corresponds = []

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
        self.basic_stats["basic"]["tp"] = len(tp)
        self.basic_stats["basic"]["fn"] = len(gt) - len(tp)
        self.basic_stats["basic"]["fp"] = len(predicted) - len(tp)

        return self.basic_stats["basic"]["tp"], self.basic_stats["basic"]["fn"], self.basic_stats["basic"]["fp"], \
               self.basic_stats["basic"]["tn"], self.all_iou_correspondencies4det

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

    def cal_stats(self, predicted, gt):
        if self.stats_type == "det":
            return self._cal_det_stats(predicted, gt)
        elif self.stats_type == "seg":
            return self._cal_seg_stats(predicted, gt)

    def update_cumulative_stats(self):
        """
        Add current tp, tn, fp, fn to cumulative stats
        :return: None
        """
        self.basic_stats["cumulative"]["tp"] += self.basic_stats["basic"]["tp"]
        self.basic_stats["cumulative"]["fn"] += self.basic_stats["basic"]["fn"]
        self.basic_stats["cumulative"]["fp"] += self.basic_stats["basic"]["fp"]
        self.basic_stats["cumulative"]["tn"] += self.basic_stats["basic"]["tn"]

    def add_cumulative_stats(self, tp, fn, fp, tn, correspondecies=[]):
        """
        Add tp, fn, fp and tn to actual count
        :param tp: true positive
        :param fn: false negative
        :param fp: false positive
        :param tn: true negative
        :return: None
        """
        self.basic_stats["cumulative"]["tp"] += tp
        self.basic_stats["cumulative"]["fn"] += fn
        self.basic_stats["cumulative"]["fp"] += fp
        self.basic_stats["cumulative"]["tn"] += tn
        for cor in correspondecies:
            self.all_iou_correspondencies4det.append(cor)

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
        self.basic_stats["basic"]["tp"] = int(tp)
        self.basic_stats["basic"]["fn"] = int(fn)
        self.basic_stats["basic"]["fp"] = int(fp)
        self.basic_stats["basic"]["tn"] = int(tn)

    def cal_complex_stats(self, type):
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
        self.population(type)
        self.predicted_positive(type)
        self.predicted_negative(type)
        self.real_positive(type)
        self.real_negative(type)

        self.bias(type)
        self.inverse_bias(type)
        self.prevalence(type)
        self.null_error_rate(type)

        self.recall(type)
        self.false_negative_rate(type)
        self.false_positive_rate(type)
        self.specifity(type)
        self.hit_rate(type)
        self.incorrect_rejection_rate(type)
        self.delivered_error_rate(type)
        self.correct_rejection_rate(type)

        #  Precision, False Negative Rate, False Discovery Rate, Inverse Precision
        self.precision(type)
        self.false_negative_accuracy(type)
        self.false_discovery_rate(type)
        self.inverse_precision(type)

        self.accuracy(type)

        self.informedness(type)
        self.markedness(type)

        self.positive_likelihood_ratio(type)
        self.negative_likelihood_ratio(type)
        self.diagnostic_odds_ratio(type)


        self.chi_square(type)
        self.correlation(type)

        self.probability_random_agreement(type)
        self.matthews_correlation_coefficent(type)
        self.cohens_kappa(type)
        self.missclassification_rate(type)
        self.iou(type)
        self.f1_score(type)

    def print_table(self, basic_type, tablefmt="grid"):
        """
        Draws a table with all the metrics
        :return:
        """
        # Creating table
        table = [[f'Population\nN = TP+TN+FP+FN\n{self.stats["population"]}', "",
                  f'Predicted Positive\nPP = TP+FP\n{self.stats["pp"]}',
                  f'Bias\npp = PP/N\n{self.stats["bias"]} {self.stats["i_bias"]}%',
                  f'Predicted Negative\nPN = FN+TN\n{self.stats["pn"]}',
                  f'Inverse Bias\npn = PN/N\n{self.stats["ib"]} {self.stats["i_ib"]}%', "", ""],
                 [],
                 [f'Real Positive\nRP = TP+FN\n{self.stats["rp"]}', "", f'TP\n\n{self.basic_stats[basic_type]["tp"]}',
                  f'Recall\ntpr = TP/RP\n{self.stats["recall"]} {self.stats["i_recall"]}%',
                  f'FN\n\n{self.basic_stats[basic_type]["fn"]}',
                  f'FNR\nfnr = FN/RP\n{self.stats["fnr"]} {self.stats["i_fnr"]}%', "",
                  f'LR+\nLR+ = tpr/fpr\n{self.stats["lr+"]}'],
                 [f'Prevalence\nrp = RP/N\n{self.stats["prevalence"]} {self.stats["i_prevalence"]}%', "",
                  f'Precision\ntpa = TP/PP\n{self.stats["precision"]} {self.stats["i_precision"]}%',
                  f'Performance\ntp = TP/N\n{self.stats["performance"]} {self.stats["i_performance"]}%',
                  f'FN Accuracy\nfna = FN/PN\n{self.stats["fna"]} {self.stats["i_fna"]}%',
                  f'Incorrect Rejection Rate\n fn = FN/N \n{self.stats["irr"]} {self.stats["i_irr"]}%', "",
                  f'LR-\nLR- = fnr/tnr\n{self.stats["lr-"]}'],
                 [],
                 [f'Real Negative\nRN = FP+TN\n{self.stats["rn"]}', "", f'FP\n\n{self.basic_stats[basic_type]["fp"]}',
                  f'FPR\nfpr = FP/RN\n{self.stats["fpr"]} {self.stats["i_fpr"]}%',
                  f'TN\n\n{self.basic_stats[basic_type]["tn"]}',
                  f'Specifity\n = TN/RN \n{self.stats["specifity"]} {self.stats["i_specifity"]}%',
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
                  f'Prob. Random Agreement\npra = (PP*RP + PN*RN)/(N)**2\n{self.stats["pra"]}%',
                  f'Markedness\n= tpa - fna\n{self.stats["markedness"]}%', "",
                  f'Accuracy\nacc = (TP + TN) / N\n{self.stats["accuracy"]} {self.stats["i_accuracy"]}%'],
                 ["", "", f'Matthews Corr. Coeff.\nmcc = (chi / N)**.5\n{self.stats["mcc"]}',
                  f'IoU\n= TP/(N-TN)\n{self.stats["iou"]} {self.stats["i_iou"]}%',
                  f'Cohen Kappa\n= (acc-pra)/(1-pra)\n{self.stats["ck"]}',
                  f'Misclassification Rate\nerr = (PF+FN)/N\n{self.stats["mr"]} {self.stats["i_mr"]}%', "",
                  f'F1 score\n= 2*TP/(RP+PP)\n{self.stats["f1"]} {self.stats["i_f1"]}%']]

        # Displaying table
        print(tabulate(table, tablefmt=tablefmt,
                       colalign=("center", "center", "center", "center", "center", "center", "center", "center")))

    def population(self, basic_type):
        self.stats["population"] = self.basic_stats[basic_type]["tp"] + self.basic_stats[basic_type]["tn"] + \
                                   self.basic_stats[basic_type]["fp"] + self.basic_stats[basic_type]["fn"]

    def predicted_positive(self, basic_type):
        self.stats["pp"] = self.basic_stats[basic_type]["tp"] + self.basic_stats[basic_type]["fp"]

    def bias(self, basic_type):
        try:
            self.stats["bias"] = round((self.stats["pp"] / self.stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.stats["pp"], nobs=self.stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_bias"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.stats["bias"] = 0
            self.stats["i_bias"] = f'[{0} - {0}]'

    def predicted_negative(self, basic_type):
        if self.stats_type == "det":
            self.stats["pn"] = -1
        else:
            self.stats["pn"] = self.basic_stats[basic_type]["fn"] + self.basic_stats[basic_type]["tn"]

    def inverse_bias(self, basic_type):
        try:
            self.stats["ib"] = round((self.stats["pn"] / self.stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.stats["pn"], nobs=self.stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_ib"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.stats["ib"] = 0
            self.stats["i_ib"] = f'[{0} - {0}]'

    def real_positive(self, basic_type):
        self.stats["rp"] = self.basic_stats[basic_type]["tp"] + self.basic_stats[basic_type]["fn"]

    def recall(self, basic_type):
        try:
            self.stats["recall"] = round((self.basic_stats[basic_type]["tp"] / self.stats["rp"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats[basic_type]["tp"], nobs=self.stats["rp"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_recall"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.stats["recall"] = 0
            self.stats["i_recall"] = f'[{0} - {0}]'

    def false_negative_rate(self, basic_type):
        try:
            self.stats["fnr"] = round((self.basic_stats[basic_type]["fn"] / self.stats["rp"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats[basic_type]["fn"], nobs=self.stats["rp"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_fnr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.stats["fnr"] = 0
            self.stats["i_fnr"] = f'[{0} - {0}]'

    def positive_likelihood_ratio(self, basic_type):
        try:
            self.stats["lr+"] = round((self.stats["recall"] / self.stats["fpr"]), 2)
        except ZeroDivisionError:
            self.stats["lr+"] = 0

    def prevalence(self, basic_type):
        try:
            self.stats["prevalence"] = round((self.stats["rp"] / self.stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.stats["rp"], nobs=self.stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_prevalence"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.stats["prevalence"] = 0
            self.stats["i_prevalence"] = f'[{0} - {0}]'

    def precision(self, basic_type):
        try:
            self.stats["precision"] = round((self.basic_stats[basic_type]["tp"] / self.stats["pp"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats[basic_type]["tp"], nobs=self.stats["pp"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_precision"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.stats["precision"] = 0
            self.stats["i_precision"] = f'[{0} - {0}]'

    def hit_rate(self, basic_type):
        try:
            self.stats["performance"] = round((self.basic_stats[basic_type]["tp"] / self.stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats[basic_type]["tp"], nobs=self.stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_performance"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.stats["performance"] = 0
            self.stats["i_performance"] = f'[{0} - {0}]'

    def false_negative_accuracy(self, basic_type):
        try:
            self.stats["fna"] = round((self.basic_stats[basic_type]["fn"] / self.stats["pn"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats[basic_type]["fn"], nobs=self.stats["pn"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_fna"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.stats["fna"] = 0
            self.stats["i_fna"] = f'[{0} - {0}]'

    def incorrect_rejection_rate(self, basic_type):
        try:
            self.stats["irr"] = round((self.basic_stats[basic_type]["fn"] / self.stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats[basic_type]["fn"], nobs=self.stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_irr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.stats["irr"] = 0
            self.stats["i_irr"] = f'[{0} - {0}]'

    def negative_likelihood_ratio(self, basic_type):
        try:
            self.stats["lr-"] = round((self.stats["fnr"] / self.stats["specifity"]), 2)
        except ZeroDivisionError:
            self.stats["lr-"] = 0

    def real_negative(self, basic_type):
        if self.stats_type == "det":
            self.stats["rn"] = -1
        else:
            self.stats["rn"] = self.basic_stats[basic_type]["fp"] + self.basic_stats[basic_type]["tn"]

    def false_positive_rate(self, basic_type):
        try:
            self.stats["fpr"] = round((self.basic_stats[basic_type]["fp"] / self.stats["rn"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats[basic_type]["fp"], nobs=self.stats["rn"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_fpr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.stats["fpr"] = 0
            self.stats["i_fpr"] = f'[{0} - {0}]'

    def specifity(self, basic_type):
        if self.stats_type == "det":
            self.stats["specifity"] = -1
            self.stats["i_specifity"] = "det"
        else:
            try:
                self.stats["specifity"] = round((self.basic_stats[basic_type]["tn"] / self.stats["rn"]) * 100, 2)
                _temp = proportion_confint(count=self.basic_stats[basic_type]["tn"], nobs=self.stats["rn"],
                                           alpha=self.confidence_interval_p, method="beta")
                self.stats["i_specifity"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
            except ZeroDivisionError:
                self.stats["specifity"] = 0
                self.stats["i_specifity"] = f'[{0} - {0}]'

    def diagnostic_odds_ratio(self, basic_type):
        try:
            self.stats["dor"] = round((self.stats["lr+"] / self.stats["lr-"]), 2)
        except:
            self.stats["dor"] = 0

    def null_error_rate(self, basic_type):
        try:
            self.stats["ner"] = round((self.stats["rn"] / self.stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.stats["rn"], nobs=self.stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_ner"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.stats["ner"] = 0
            self.stats["i_ner"] = f'[{0} - {0}]'

    def false_discovery_rate(self, basic_type):
        try:
            self.stats["fdr"] = round((self.basic_stats[basic_type]["fp"] / self.stats["pp"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats[basic_type]["fp"], nobs=self.stats["pp"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_fdr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except:
            self.stats["fdr"] = 0
            self.stats["i_fdr"] = f'[{0} - {0}]'

    def delivered_error_rate(self, basic_type):
        try:
            self.stats["der"] = round((self.basic_stats[basic_type]["fp"] / self.stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats[basic_type]["fp"], nobs=self.stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_der"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.stats["der"] = 0
            self.stats["i_der"] = f'[{0} - {0}]'

    def inverse_precision(self, basic_type):
        if self.stats_type == "det":
            self.stats["ip"] = -1
            self.stats["i_ip"] = "det"
        else:
            try:
                self.stats["ip"] = round((self.basic_stats[basic_type]["tn"] / self.stats["pn"]) * 100, 2)
                _temp = proportion_confint(count=self.basic_stats[basic_type]["tn"], nobs=self.stats["pn"],
                                           alpha=self.confidence_interval_p, method="beta")
                self.stats["i_ip"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
            except:
                self.stats["ip"] = 0
                self.stats["i_ip"] = f'[{0} - {0}]'

    def correct_rejection_rate(self, basic_type):
        if self.stats_type == "det":
            self.stats["crr"] = -1
            self.stats["i_crr"] = "det"
        else:
            self.stats["crr"] = round((self.basic_stats[basic_type]["tn"] / self.stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats[basic_type]["tn"], nobs=self.stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_crr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'

    def informedness(self, basic_type):
        self.stats["informedness"] = round((self.stats["recall"] - self.stats["fpr"]), 2)

    def chi_square(self, basic_type):
        if self.stats_type == "det":
            self.stats["chi"] = -1
            self.stats["p"] = "det"
        else:
            _temp11 = (self.stats["rp"] * self.stats["pp"]) / self.stats["population"]
            _temp12 = (self.stats["rp"] * self.stats["pn"]) / self.stats["population"]
            _temp21 = (self.stats["rn"] * self.stats["pp"]) / self.stats["population"]
            _temp22 = (self.stats["rn"] * self.stats["pn"]) / self.stats["population"]

            _temp11 = 0 if _temp11 == 0 else (self.basic_stats[basic_type]["tp"] - _temp11) ** 2 / _temp11
            _temp12 = 0 if _temp12 == 0 else (self.basic_stats[basic_type]["fp"] - _temp12) ** 2 / _temp12
            _temp21 = 0 if _temp21 == 0 else (self.basic_stats[basic_type]["fn"] - _temp21) ** 2 / _temp21
            _temp22 = 0 if _temp22 == 0 else (self.basic_stats[basic_type]["tn"] - _temp22) ** 2 / _temp22

            self.stats["chi"] = round(_temp11 + _temp12 + _temp21 + _temp22, 2)
            self.stats["p"] = round(1 - f.cdf(self.stats["chi"], 1, 1000000000), 2)

    def correlation(self, basic_type):
        if self.stats_type == "det":
            self.stats["correlation"] = -1
            self.stats["i_correlation"] = "det"
        else:
            try:
                self.stats["correlation"] = round(((self.basic_stats[basic_type]["tp"] * self.basic_stats[basic_type][
                    "tn"] + self.basic_stats[basic_type]["fp"] * self.basic_stats[basic_type]["fn"]) / (
                                                               self.stats["pp"] * self.stats["rp"] * self.stats["rn"] *
                                                               self.stats["pn"]) ** .5) * 100, 2)
                _temp = proportion_confint(
                    count=(self.basic_stats[basic_type]["tp"] * self.basic_stats[basic_type]["tn"] +
                           self.basic_stats[basic_type]["fp"] * self.basic_stats[basic_type]["fn"]),
                    nobs=(self.stats["pp"] * self.stats["rp"] * self.stats["rn"] * self.stats[
                        "pn"]) ** .5,
                    alpha=self.confidence_interval_p, method="beta")
                self.stats["i_correlation"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
            except:
                self.stats["correlation"] = 0
                self.stats["i_correlation"] = f'[{0} - {0}]'

    def probability_random_agreement(self, basic_type):
        try:
            self.stats["pra"] = round(
                ((self.stats["pp"] * self.stats["rp"] + self.stats["pn"] * self.stats["rn"]) /
                 self.stats["population"] ** 2) * 100, 2)
        except ZeroDivisionError:
            self.stats["pra"] = 0
            self.stats["i_pra"] = f'[{0} - {0}]'

    def markedness(self, basic_type):
        self.stats["markedness"] = round((self.stats["precision"] - self.stats["fna"]), 2)

    def accuracy(self, basic_type):
        if self.stats_type == "det":
            self.stats["accuracy"] = -1
            self.stats["i_accuracy"] = "det"
        else:
            self.stats["accuracy"] = round(
                ((self.basic_stats[basic_type]["tp"] + self.basic_stats[basic_type]["tn"]) / self.stats[
                    "population"]) * 100, 2)
            _temp = proportion_confint(count=(self.basic_stats[basic_type]["tp"] + self.basic_stats[basic_type]["tn"]),
                                       nobs=self.stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_accuracy"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'

    def matthews_correlation_coefficent(self, basic_type):
        if self.stats_type == "det":
            self.stats["mcc"] = -1
        else:
            self.stats["mcc"] = round((self.stats["chi"] / self.stats["population"]) ** 0.5, 2)

    def iou(self, basic_type):
        if self.stats_type == "det":
            self.stats["iou"] = round(np.mean(self.all_iou_correspondencies4det) * 100, 2)
            self.stats["i_iou"] = f'det'
        else:
            try:
                self.stats["iou"] = round((self.basic_stats[basic_type]["tp"] / (
                            self.stats["population"] - self.basic_stats[basic_type]["tn"])) * 100, 2)
                _temp = proportion_confint(count=self.basic_stats[basic_type]["tp"],
                                           nobs=self.stats["population"] - self.basic_stats[basic_type]["tn"],
                                           alpha=self.confidence_interval_p, method="beta")
                self.stats["i_iou"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
            except:
                self.stats["iou"] = 0
                self.stats["i_iou"] = f'[{0} - {0}]'

    def cohens_kappa(self, basic_type):
        try:
            self.stats["ck"] = round((self.stats["accuracy"] - self.stats["pra"]) / (100 - self.stats["pra"]), 2)
        except ZeroDivisionError:
            self.stats["ck"] = 0

    def missclassification_rate(self, basic_type):
        try:
            self.stats["mr"] = round((
                                             (self.basic_stats[basic_type]["fp"] + self.basic_stats[basic_type]["fn"]) /
                                             self.stats["population"]) * 100, 2)
            _temp = proportion_confint(count=self.basic_stats[basic_type]["fp"] + self.basic_stats[basic_type]["fn"],
                                       nobs=self.stats["population"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_mr"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except ZeroDivisionError:
            self.stats["mr"] = 0
            self.stats["i_mr"] = f'[{0} - {0}]'

    def f1_score(self, basic_type):
        try:
            self.stats["f1"] = round(
                (2 * self.basic_stats[basic_type]["tp"] / (self.stats["rp"] + self.stats["pp"])) * 100,
                2)
            _temp = proportion_confint(count=2 * self.basic_stats[basic_type]["tp"],
                                       nobs=self.stats["rp"] + self.stats["pp"],
                                       alpha=self.confidence_interval_p, method="beta")
            self.stats["i_f1"] = f'[{round(_temp[0] * 100, 2)} - {round(_temp[1] * 100, 2)}]'
        except:
            self.stats["f1"] = 0
            self.stats["i_f1"] = f'[{0} - {0}]'


if __name__ == '__main__':
    s = Metrics(p=0.01)
    type = "basic"
    s.test(50, 20, 10, 200)
    s.cal_complex_stats(type)
    s.print_table(type)
