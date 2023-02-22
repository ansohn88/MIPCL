import pandas as pd
import torch
from torchmetrics.classification import (MulticlassAccuracy, MulticlassAUROC,
                                         MulticlassAveragePrecision,
                                         MulticlassF1Score,
                                         MulticlassPrecision, MulticlassRecall,
                                         MulticlassSpecificity)


def get_preds_tgts_per_fold(results_fp: str,
                            fold_num: int) -> dict:
    fold_dict = (pd.read_pickle(results_fp))[f'fold_{fold_num}']
    fold_preds = fold_dict['predictions'].cpu()
    fold_probs = fold_dict['probabilities'].cpu()
    fold_tgts = fold_dict['targets'].cpu()

    return {
        'fold_preds': fold_preds,
        'fold_probs': fold_probs,
        'fold_tgts': fold_tgts
    }


def all_folds_preds_tgts(results_fp: str):
    probs = []
    preds = []
    tgts = []
    for i in range(1, 11):
        fold_results = get_preds_tgts_per_fold(results_fp, fold_num=i)
        probs.append(fold_results['fold_probs'])
        preds.append(fold_results['fold_preds'])
        tgts.append(fold_results['fold_tgts'])

    probs = torch.cat(probs)
    preds = torch.cat(preds)
    tgts = torch.cat(tgts)

    return probs, preds, tgts


def get_metric(preds_tgts: dict,
               metric: str,
               average: str):
    # preds = preds_tgts['fold_preds']
    # probs = preds_tgts['fold_probs']
    # tgts = preds_tgts['fold_tgts']

    probs = preds_tgts[0]
    preds = preds_tgts[1]
    tgts = preds_tgts[2]

    if metric == 'sensitivity':
        metric = MulticlassRecall(num_classes=2, average=average)
    elif metric == 'specificity':
        metric = MulticlassSpecificity(num_classes=2, average=average)
    elif metric == 'auroc':
        metric = MulticlassAUROC(num_classes=2, average=average)
    elif metric == 'precision':
        metric = MulticlassPrecision(num_classes=2, average=average)
    elif metric == 'f1_score':
        metric = MulticlassF1Score(num_classes=2, average=average)
    elif metric == 'auprc':
        metric = MulticlassAveragePrecision(num_classes=2, average=average)
    elif metric == 'accuracy':
        metric = MulticlassAccuracy(num_classes=2, average=average)

    if metric == 'auroc' or metric == 'auprc':
        m_val = metric(probs, tgts)
    else:
        m_val = metric(preds, tgts)

    return m_val


def avg_metric_across_folds(results_fp: str,
                            metric: str,
                            average: str) -> float:
    # total = 0
    # for i in range(1, 11):
    #     fold_results = get_preds_tgts_per_fold(results_fp, fold_num=i)
    #     metric_val = get_metric(fold_results, metric, average)
    #     total += metric_val
    # return (total / 10) * 100.

    fold_results = all_folds_preds_tgts(results_fp)
    metric_val = get_metric(fold_results, metric, average)
    return metric_val


def get_confusion_mtx(results_fp: str):
    preds = []
    tgts = []
    for i in range(1, 11):
        fold_results = get_preds_tgts_per_fold(results_fp, fold_num=i)
        preds.append(fold_results['fold_preds'])
        tgts.append(fold_results['fold_tgts'])

    preds = torch.cat(preds)
    tgts = torch.cat(tgts)

    # metric = MulticlassConfusionMatrix(num_classes=2, normalize=None)
    # return metric(preds, tgts)
    return preds, tgts


abmil = "/home/andy/baraslab/projects/panc_cyto/abmil_lit/results/abmil_dim384/convnext_abmil.pkl"
clam = "/home/andy/baraslab/projects/panc_cyto/clam_lit/results/stain_seed/clam_trial_001.pkl"

fgbg_1dfg = "/home/andy/baraslab/projects/panc_cyto/mipcl_lit/results/fgbg_1dfg_dim384_thr0.85_temp0.07_aNone_bwNone/convnext_bagwt_None.pkl"
fg_fc = "/home/andy/baraslab/projects/panc_cyto/mipcl_lit/results/fg_fc_dim384_thr0.85_temp0.07_aNone_bwNone/convnext_bagwt_None.pkl"


if __name__ == '__main__':
    metrics = ['sensitivity', 'specificity', 'auroc',
               'precision', 'f1_score', 'auprc']

    average = "weighted"

    for m in metrics:
        amv = avg_metric_across_folds(abmil, m, average)
        cmv = avg_metric_across_folds(clam, m, average)

        mipcl = avg_metric_across_folds(fgbg_1dfg, m, average)
        mipcl_fgfc = avg_metric_across_folds(fg_fc, m, average)

        print(f'ABMIL {m} avg across 10-folds: {amv}')
        print(f'CLAM {m} avg across 10-folds: {cmv}')

        print(f'FGBG 1DFG {m} avg across 10-folds: {mipcl}')
        print(f'FG FC {m} avg across 10-folds: {mipcl_fgfc}')

        print('\n')


# if __name__ == '__main__':
#     cm = get_confusion_mtx(t85_infonce)
