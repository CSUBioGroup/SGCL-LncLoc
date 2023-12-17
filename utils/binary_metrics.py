import numpy as np
from sklearn import metrics as skmetrics
import warnings

warnings.filterwarnings("ignore")


class Metrictor:
    def __init__(self):
        self._reporter_ = {"F1": self.F1, "ACC": self.ACC, "AUC": self.AUC, "AUPR": self.AUPR, "MCC": self.MCC, "Pre": self.Pre,
                           "Rec": self.Rec}

    def __call__(self, report, end='\n'):
        res = {}
        for mtc in report:
            v = self._reporter_[mtc]()
            print(f" {mtc}={v:6.3f}", end=';')
            res[mtc] = v
        print(end=end)
        return res

    def set_data(self, Y_prob_pre, Y, threshold=0.5):
        # Y:(n_samples), Y_prob_pre:(n_samples), ndarray
        self.Y_prob_pre, self.Y = Y_prob_pre, Y
        self.Y_pre = np.where(Y_prob_pre > threshold, 1, 0)
        self.N = len(self.Y)

    @staticmethod
    def table_show(resList, report, rowName='CV'):
        lineLen = len(report) * 8 + 6
        print("=" * (lineLen // 2 - 6) + "FINAL RESULT" + "=" * (lineLen // 2 - 6))
        print(f"{'-':^6}" + "".join([f"{i:>8}" for i in report]))
        for i, res in enumerate(resList):
            print(f"{rowName + '_' + str(i + 1):^6}" + "".join([f"{res[j]:>8.3f}" for j in report]))
        print(f"{'MEAN':^6}" + "".join([f"{np.mean([res[i] for res in resList]):>8.3f}" for i in report]))
        print("======" + "========" * len(report))

    def each_class_indictor_show(self, id2lab):
        id2lab = np.array(id2lab)
        Yarr = np.zeros((self.N, 2), dtype='int32')
        Yarr[list(range(self.N)), self.Y] = 1
        TPi, FPi, TNi, FNi = _TPiFPiTNiFNi(2, self.Y_pre, self.Y)
        MCCi = fill_inf((TPi * TNi - FPi * FNi) / np.sqrt((TPi + FPi) * (TPi + FNi) * (TNi + FPi) * (TNi + FNi)),
                        np.nan)
        Pi = fill_inf(TPi / (TPi + FPi))
        Ri = fill_inf(TPi / (TPi + FNi))
        Fi = fill_inf(2 * Pi * Ri / (Pi + Ri))
        sortedIndex = np.argsort(id2lab)
        classRate = Yarr.sum(axis=0)[sortedIndex] / self.N
        id2lab, MCCi, Pi, Ri, Fi = id2lab[sortedIndex], MCCi[sortedIndex], Pi[sortedIndex], Ri[sortedIndex], Fi[
            sortedIndex]
        print("-" * 28 + "MACRO INDICTOR" + "-" * 28)
        print(f"{'':30}{'rate':<8}{'MCCi':<8}{'Pi':<8}{'Ri':<8}{'Fi':<8}")
        for i, c in enumerate(id2lab):
            print(f"{c:30}{classRate[i]:<8.2f}{MCCi[i]:<8.3f}{Pi[i]:<8.3f}{Ri[i]:<8.3f}{Fi[i]:<8.3f}")
        print("-" * 70)

    def F1(self):
        return skmetrics.f1_score(self.Y, self.Y_pre)

    def ACC(self):
        return skmetrics.accuracy_score(self.Y, self.Y_pre)

    def MCC(self):
        return skmetrics.matthews_corrcoef(self.Y, self.Y_pre)

    def AUPR(self):
        return skmetrics.average_precision_score(self.Y, self.Y_prob_pre)
    
    def AUC(self):
        return skmetrics.roc_auc_score(self.Y, self.Y_prob_pre)

    def Pre(self):
        return skmetrics.precision_score(self.Y, self.Y_pre)

    def Rec(self):
        return skmetrics.recall_score(self.Y, self.Y_pre)


def _TPiFPiTNiFNi(classNum, Y_pre, Y):
    Yarr, Yarr_pre = np.zeros((len(Y), classNum), dtype='int32'), np.zeros((len(Y), classNum), dtype='int32')
    Yarr[list(range(len(Y))), Y] = 1
    Yarr_pre[list(range(len(Y))), Y_pre] = 1
    isValid = (Yarr.sum(axis=0) + Yarr_pre.sum(axis=0)) > 0
    Yarr, Yarr_pre = Yarr[:, isValid], Yarr_pre[:, isValid]
    TPi = np.array([Yarr_pre[:, i][Yarr[:, i] == 1].sum() for i in range(Yarr.shape[1])], dtype='float32')
    FPi = Yarr_pre.sum(axis=0) - TPi
    TNi = (1 ^ Yarr).sum(axis=0) - FPi
    FNi = Yarr.sum(axis=0) - TPi
    return TPi, FPi, TNi, FNi


from collections.abc import Iterable


def fill_inf(x, v=0.0):
    if isinstance(x, Iterable):
        x[x == np.inf] = v
        x[np.isnan(x)] = v
    else:
        x = v if x == np.inf else x
        x = v if np.isnan(x) else x
    return x
