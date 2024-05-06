import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.impute import SimpleImputer

##############################################
#Chosen parameters
##############################################
temperatures = {"fitness": 1, "surfacc": 1, "exchangability": 2}
flu_thresh = 0.054
hiv_thresh = 0.138
rbd_thresh = 0.57
rbd_xie_thresh = 0.90

#####################################################
#Read processed experiments and components scores
#####################################################
# 提取flu数据
flu = pd.read_csv("../results/summaries/h1_experiments_and_scores.csv")
flu_ablist = [col for col in flu.columns.values if "mutfracsurvive" in col]
flu_all_ab = [col for col in flu.columns.values if "mutfracsurvive" in col]

# 提取hiv数据
hiv = pd.read_csv("../results/summaries/bg505_experiments_and_scores.csv")
hiv_ablist = [
    col for col in hiv.columns.values
    if "mutfracsurvive" in col and "VRC34" not in col
]
hiv_all_ab = [col for col in hiv.columns.values if "mutfracsurvive" in col]

# 提取rbd数据
rbd = pd.read_csv("../results/summaries/rbd_experiments_and_scores.csv")
rbd_ablist = [
    col for col in rbd.columns.values if "escape_" in col and "_Bloom" in col
]

rbd_xie_ablist = [
    col for col in rbd.columns.values if "escape_" in col and "_Xie" in col
]

# 提取spike数据
spike = pd.read_csv("../results/summaries/spike_scores.csv")
rbd_all_ab = [col for col in rbd.columns.values if "escape_" in col]

# 提取lassa数据
lassa = pd.read_csv('../results/summaries/lassa_glycoprotein_scores.csv')

# 提取nipahg数据
nipahg = pd.read_csv('../results/summaries/nipah_glycoprotein_scores.csv')

# 提取nipahf数据
nipahf = pd.read_csv('../results/summaries/nipah_fusion_scores.csv')

###################################################################
#Functions to calculate EVEscape and aggregate/binarize experiments
###################################################################

# 定义了一个名为 logistic 的函数，它实现了逻辑斯蒂回归中的逻辑函数（Logistic function），也称为 Sigmoid 函数
def logistic(x):
    return 1 / (1 + np.exp(-x))

# 执行标准化处理（或称为 Z-score 标准化），主要目的是将数据转换为具有平均值为 0 和标准差为 1 的形式。
def standardization(x):
    """Assumes input is numpy array or pandas series"""
    return (x - x.mean()) / x.std()

## 做预测
def make_predictors(summary_init, thresh, ablist, scores=True):

    summary = summary_init.copy()

    #Drop extraneous WCN columns
    summary = summary.drop(
        columns=[col for col in summary.columns if "wcn_fill_" in col])
    summary = summary.drop(
        columns=[col for col in summary.columns if "wcn_sc" in col])
    summary = summary.drop(
        columns=[col for col in summary.columns if "diff" in col])

    #Reverse WCN direction so that larger values are more accessible
    summary["wcn_fill_r"] = -summary.wcn_fill
    summary = summary.drop(columns="wcn_fill")

    if scores:
        #Calculate max escape for each mutant
        summary["max_escape_experiment"] = summary[ablist].max(axis=1)
        #Calculate if escape>threshold for each mutant
        summary[
            "is_escape_experiment"] = summary["max_escape_experiment"] > thresh

    #Impute missing values for columns used to calculate EVEscape scores
    impute_cols = ["i", "evol_indices", "wcn_fill_r", "charge_ew-hydro"]

    df_imp = summary[impute_cols].copy()
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    df_imp = pd.DataFrame(imp.fit_transform(df_imp),
                          columns=df_imp.columns,
                          index=df_imp.index)
    df_imp = pd.concat([df_imp, summary[["wt", "mut"]]], axis=1)

    #Compute EVEscape scores
    summary["evescape"] = 0
    summary["evescape"] += np.log(
        logistic(
            standardization(df_imp["evol_indices"]) * 1 /
            temperatures["fitness"]))
    summary["evescape"] += np.log(
        logistic(
            standardization(df_imp["wcn_fill_r"]) * 1 /
            temperatures["surfacc"]))
    summary["evescape"] += np.log(
        logistic(
            standardization(df_imp["charge_ew-hydro"]) * 1 /
            temperatures["exchangability"]))

    summary = summary.drop(
        columns=[col for col in summary.columns if col == "wcn_fill"])

    summary = summary.rename(
        columns={
            "evol_indices": "fitness_eve",
            "wcn_fill_r": "accessibility_wcn",
            "charge_ew-hydro": "dissimilarity_charge_hydro"
        })

    summary = summary.round(decimals=7)

    return (summary)


##############################################
#Make Calculations
##############################################
flu = make_predictors(flu, flu_thresh, flu_ablist)
hiv = make_predictors(hiv, hiv_thresh, hiv_ablist)
rbd_bloom = make_predictors(rbd, rbd_thresh, rbd_ablist)
rbd_xie = make_predictors(rbd, rbd_xie_thresh, rbd_xie_ablist)
spike = make_predictors(spike, None, None, scores=False)
lassa = make_predictors(lassa, None, None, scores=False)
nipahg = make_predictors(nipahg, None, None, scores=False)
nipahf = make_predictors(nipahf, None, None, scores=False)

rbd_all = rbd_bloom.rename(
    columns={
        "max_escape_experiment": "max_escape_experiment_bloom",
        "is_escape_experiment": "is_escape_experiment_bloom"
    }).merge(
        rbd_xie.rename(
            columns={
                "max_escape_experiment": "max_escape_experiment_xie",
                "is_escape_experiment": "is_escape_experiment_xie"
            }))

rbd_all = rbd_all.rename(columns={
    "rbd_ace2_binding": "bloom_ace2_binding",
    "rbd_expression": "bloom_expression"
})
rbd_all["is_escape_experiment_all"] = (
    rbd_all["is_escape_experiment_bloom"]) | (
        rbd_all["is_escape_experiment_xie"])

flu = flu.drop(columns=flu_all_ab)
hiv = hiv.drop(columns=hiv_all_ab)
rbd_all = rbd_all.drop(columns=rbd_all_ab)
rbd_all = rbd_all.drop(columns="Naive Freq")
evescape = rbd_all.pop("evescape")
rbd_all.insert(10, "evescape", evescape)

##############################################
#Save Summaries with EVEscape/binarized experimetnal escape
##############################################
flu.to_csv("../results/summaries_with_scores/flu_h1_evescape.csv", index=False)
hiv.to_csv("../results/summaries_with_scores/hiv_env_evescape.csv",
           index=False)
rbd_all.to_csv("../results/summaries_with_scores/spike_rbd_evescape.csv",
               index=False)
spike.to_csv("../results/summaries_with_scores/full_spike_evescape.csv",
             index=False)
lassa.to_csv(
    "../results/summaries_with_scores/lassa_glycoprotein_evescape.csv",
    index=False)
nipahg.to_csv(
    "../results/summaries_with_scores/nipah_glycoprotein_evescape.csv",
    index=False)
nipahf.to_csv("../results/summaries_with_scores/nipah_fusion_evescape.csv",
              index=False)


#######################################################################
#Save Site-Level Summaries with EVEscape/binarized experimetnal escape
#######################################################################
# 将 make_site 函数应用于各自的数据集，生成了每个病原体或蛋白的位点级数据摘要，这些摘要反映了各个位点的平均特征，包括逃逸能力、突变效应等重要生物学信息。
def make_site(summary_init):

    summary = summary_init.copy()
    summary = summary.groupby(['i', 'wt']).agg('mean').reset_index()

    return (summary)


flu_site = make_site(flu)
hiv_site = make_site(hiv)
rbd_all_site = make_site(rbd_all)
spike_site = make_site(spike)
lassa_site = make_site(lassa)
nipahg_site = make_site(nipahg)
nipahf_site = make_site(nipahf)

# 保存上面得到的数据
flu_site.to_csv('../results/summaries_with_scores/flu_h1_evescape_sites.csv',
                index=False)
hiv_site.to_csv('../results/summaries_with_scores/hiv_env_evescape_sites.csv',
                index=False)
rbd_all_site.to_csv(
    '../results/summaries_with_scores/spike_rbd_evescape_sites.csv',
    index=False)
spike_site.to_csv(
    '../results/summaries_with_scores/full_spike_evescape_sites.csv',
    index=False)
lassa_site.to_csv(
    '../results/summaries_with_scores/lassa_glycoprotein_evescape_sites.csv',
    index=False)
nipahg_site.to_csv(
    '../results/summaries_with_scores/nipah_glycoprotein_evescape_sites.csv',
    index=False)
nipahf_site.to_csv(
    '../results/summaries_with_scores/nipah_fusion_evescape_sites.csv',
    index=False)
