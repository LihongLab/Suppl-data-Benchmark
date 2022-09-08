# conda activate python38

import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from scipy import integrate
import math
from scipy.stats import rankdata
# from numba import jit
#import multiprocessing as mp
import concurrent.futures
from pandarallel import pandarallel
from sklearn.metrics import roc_auc_score
from scipy import stats

# Pearson correlation
def pearson_corr_base(df):
    return df.iloc[:, 1].corr(df.iloc[:, 2], 'pearson')


def pearson_corr(df: pd.DataFrame) -> pd.Series:
    """ Calculate the pearson correlation for each drug.
    
    Args:
        df (pd.DataFrame): A DataFrame record drug_id,true_values,
        predict_values. [drug_id,true_values,predict_values]
    Return:
        series_pearson_cor (pd.Series): Series of pearson correlation efficience
        for each drug.
    """
    return df.groupby(df.iloc[:, 0]).apply(pearson_corr_base)


# Spearman correlation
def spearman_corr_base(df):
    return df.iloc[:, 1].corr(df.iloc[:, 2], 'spearman')


def spearman_corr(df: pd.DataFrame) -> pd.Series:
    """ Calculate the spearman's rank correlation for each drug.
    
    Args:
        df (pd.DataFrame): A DataFrame record drug_id,true_values,
        predict_values. [drug_id,true_values,predict_values]
    Return:
        series_pearson_cor (pd.Series): Series of pearson correlation efficience
        for each drug.
    """
    return df.groupby(df.iloc[:, 0]).apply(spearman_corr_base)


# RMSE

"""some measures for evaluation of prediction, tests and model selection

Created on Tue Nov 08 15:23:20 2011
Updated on Wed Jun 03 10:42:20 2020

Authors: Josef Perktold & Peter Prescott
License: BSD-3
URL: 
https://www.statsmodels.org/devel/_modules/statsmodels/tools
/eval_measures.html#rmse

"""


def mse_base(x1, x2, axis=0):
    """mean squared error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    mse : ndarray or float
       mean squared error along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass, for example
    numpy matrices will silently produce an incorrect result.

    np.nanmean Returns the average of the array elements. The average is taken
    over the flattened array by default, otherwise over the specified axis. 
    float64 intermediate and return values are used for integer inputs.
    For all-NaN slices, NaN is returned and a RuntimeWarning is raised.
    
    The arithmetic mean is the sum of the non-NaN elements along the axis 
    divided by the number of non-NaN elements.
    
    """
    x1 = np.asanyarray(x1, dtype='f8')
    x2 = np.asanyarray(x2, dtype='f8')
    return np.nanmean((x1 - x2) ** 2, axis=axis)


def rmse_base(df, axis=0):
    """root mean squared error

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame contains the true and predict values [drug_id,true, predict]
            x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    rmse : ndarray or float
       root mean squared error along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass, for example
    numpy matrices will silently produce an incorrect result.
    """
    x1 = np.asanyarray(df.iloc[:, 1], dtype='f8')
    x2 = np.asanyarray(df.iloc[:, 2], dtype='f8')
    return np.sqrt(mse_base(x1, x2, axis=axis))


def rmse(df):
    """root mean squared error for each drug
    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame contains the true and predict values [drug_id,true, predict]
    Returns
    -------
    rmse : ndarray or float
       root mean squared error along given axis.         
    """

    return df.groupby(df.iloc[:, 0]).apply(rmse_base)


# NRMSE
def nrmse_base(df):
    """ Normalized RMSE
    """
    x1 = np.asanyarray(df.iloc[:, 1], dtype='f8')
    df_max = x1.max()
    df_min = x1.min()
    return rmse_base(df) / (df_max - df_min)


def nrmse(df):
    return df.groupby(df.iloc[:, 0]).apply(nrmse_base)


# L-RMSE
def l_rmse_base(df):
    """ L-RMSE
    """
    x1 = np.asanyarray(df.iloc[:, 1], dtype='f8')
    df_median = np.median(x1)
    df_sigma = 1.4826 * np.median(np.abs(x1 - df_median))
    ll = df_median - 1.65 * df_sigma
    df_l = df[df.iloc[:, 1] <= ll]
    return rmse_base(df_l)


def l_rmse(df):
    return df.groupby(df.iloc[:, 0]).apply(l_rmse_base)


# R-RMSE
def r_rmse_base(df):
    """ R-RMSE
    """
    x1 = np.asanyarray(df.iloc[:, 1], dtype='f8')
    df_median = np.median(x1)
    df_sigma = 1.4826 * np.median(np.abs(x1 - df_median))
    # l = df_median - 1.65 * df_sigma
    rr = df_median + 1.65 * df_sigma
    df_r = df[df.iloc[:, 1] >= rr]
    return rmse_base(df_r)


def r_rmse(df):
    return df.groupby(df.iloc[:, 0]).apply(r_rmse_base)


# LR-RMSE
def lr_rmse_base(df):
    """ LR-RMSE
    """
    x1 = np.asanyarray(df.iloc[:, 1], dtype='f8')
    df_median = np.median(x1)
    df_sigma = 1.4826 * np.median(np.abs(x1 - df_median))
    ll = df_median - 1.65 * df_sigma
    rr = df_median + 1.65 * df_sigma
    df_lr = df[(df.iloc[:, 1] >= rr) | (df.iloc[:, 1] <= ll)]
    return rmse_base(df_lr)


def lr_rmse(df):
    return df.groupby(df.iloc[:, 0]).apply(lr_rmse_base)


# # Normalized discounted cumulative gain (NDCG) from zhihu
# def ndcg_zhihu(golden, current, n=-1):
#     """NDCG

#     Normalized discounted cumulative gain codes adopted from
#     https://zhuanlan.zhihu.com/p/136199536

#     Parameters
#     ----------
#     golden
#     current
#     n

#     Returns
#     -------

#     """
#     log2_table = np.log2(np.arange(2, 102))

#     def dcg_at_n(rel, n):
#         rel = np.asfarray(rel)[:n]
#         dcg = np.sum(np.divide(np.power(2, rel) - 1, log2_table[:rel.shape[0]]))
#         return dcg

#     ndcgs = []
#     for i in range(len(current)):
#         k = len(current[i]) if n == -1 else n
#         idcg = dcg_at_n(sorted(golden[i], reverse=True), n=k)
#         dcg = dcg_at_n(current[i], n=k)
#         tmp_ndcg = 0 if idcg == 0 else dcg / idcg
#         ndcgs.append(tmp_ndcg)
#     return 0. if len(ndcgs) == 0 else sum(ndcgs) / (len(ndcgs))


# Normalized discounted cumulative gain (NDCG)
def ndcg_base(df: pd.DataFrame) -> float:
    """

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame contains the true and predict values [drug_id,true, predict]

    Returns
    -------
    ndcg_score

    """
    x1 = np.asanyarray(df.iloc[:, 1], dtype='f8')
    x2 = np.asanyarray(df.iloc[:, 2], dtype='f8')
    dcg = np.sum(np.power(2,-x1) / np.log2(rankdata(x2) + 1))
    dcg_n = np.sum(np.power(2,-x1) / np.log2(rankdata(x1) + 1))
    
    return dcg / dcg_n


def ndcg(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(df.iloc[:, 0]).apply(ndcg_base)


# # Normalized weighted probabilistic c-index (NWPC)
# def nwpc_base(df: pd.DataFrame) -> pd.DataFrame:
#     """

#     Parameters
#     ----------
#     df: pd.DataFrame
#         A DataFrame contains the true and predict values [drug_id,true, predict]

#     Returns
#     -------

#     """


def pCIdx_erf(x: float):
    """Error function of pc-index

    Args:
        x:

    Returns:

    """
    def erf(t):
        return math.exp(- t ** 2)
    # erf = lambda t: math.exp(- t ** 2)
    return 2 / math.sqrt(math.pi) * integrate.quad(erf, 0, x)[0]


"""
import matplotlib.pyplot as plt

fig , axs = plt.subplots()
x = np.linspace(-5.0, 5.0)
v_pCIdx_erf = np.vectorize(lambda x: 0.5*(1+pCIdx_erf(x)))
y = v_pCIdx_erf(x)

axs.plot(x,y,'-')
axs.vlines(x=0, ymin=0, ymax=0.5,
           lw=1.5, colors='red',
           linestyles= '--')
axs.hlines(y=0.5, xmin=-5, xmax=0,
           lw=1.5, colors='red',
           linestyles= '--')

plt.show()
"""


def pCIdx_hp(
        si: float, sj: float,
        r_si_hat: float, r_sj_hat: float,
        sd: float
) -> float:
    """hp function of pc-index

    Args:
        si: (float)
            - True value of drug response of cell line i, IC50 or AUC.
            Be careful, IC50, GI50 or AUC, the lower the value is, the more
            effective is the drug.
        sj: (float)
            - True value of drug response of cell line j, IC50 or AUC.
            Be careful, IC50, GI50 or AUC, the lower the value is, the more
            effective is the drug.
        r_si_hat: (float)
            - Predicted rank of drug response of cell line i in the decreasing
            order based on IC50 or AUC. Be careful, the higher the value is, the
            more effective is the drug.
        r_sj_hat: (float)
            - Predicted rank of drug response of cell line j in the decreasing
            order based on IC50 or AUC. Be careful, the higher the value is, the
            more effective is the drug.
        sd: (float)
            - Standard variance of true drug response values in the given drug.

    Returns:
        hp function output: (float)
        Be careful, if the rank of the predicted value is consistent with the
        true value, the output of hp function ranges from 0.5 to 1.

    """
    if r_si_hat < r_sj_hat:
        return 0.5 * (1 + pCIdx_erf((si - sj) / (2 * sd)))
    elif r_si_hat > r_sj_hat:
        return 0.5 * (1 + pCIdx_erf((sj - si) / (2 * sd)))
    else:
        return 0.5


def pCIdx_base(df: pd.DataFrame) -> float:
    """

    Args:
        df: (pd.DataFrame)
            -  A DataFrame record drug_id,true_values, predict_values.
            `[drug_id,true_values,predict_values]`

    Returns:
        pc-index for the given drug

    """
    x1 = np.asanyarray(df.iloc[:, 1], dtype='f8')
    x2 = np.asanyarray(df.iloc[:, 2], dtype='f8')
    x1_sd = np.std(x1)
    # Rank the predict results in decreasing order
    rank_x2 = rankdata(- x2)
    runningSum = 0
    
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         results = [executor.submit(pCIdx_hp,x1[i],x1[j],rank_x2[i],rank_x2[j],x1_sd) for i in range(len(x1)) for j in range(len(x2))]

#         for f in concurrent.futures.as_completed(results):
#             runningSum.append(f.result())
            
#     runningSum = np.sum(runningSum)
            
            
    for j in range(len(x2)):
        for i in range(len(x1)):
            if i < j:
                runningSum = runningSum + pCIdx_hp(x1[i], x1[j],
                                                rank_x2[i], rank_x2[j],
                                                x1_sd)
            else:
                break

    return 2 / (len(x1) * (len(x1) - 1)) * runningSum

def pCIdx(df: pd.DataFrame) -> pd.Series:
    """

    Args:
        df: (pd.DataFrame)
            -  A DataFrame record drug_id,true_values, predict_values.
            `[drug_id,true_values,predict_values]`

    Returns:
        pc-index for all drugs
    """
    
    pandarallel.initialize()
#     pc_Index = dict()
#     futures_dict = dict()
#     with concurrent.futures.Executor() as executor:
#         for name,group in df.groupby(df.iloc[:,0]):
#             futures = executor.submit(pCIdx_base, group)
#             futures_dict.update({futures : name})
#         # futures = {executor.submit(pCIdx_base, group): name for name,group in df.groupby(df.iloc[:, 0])}
#         for fut in concurrent.futures.as_completed(futures_dict):
#             pc_Index.update({futures_dict[fut]:fut.result()})
#         # drug_weight.update({name: weight})
#         # original_task = futures[fut] 
#         # print(f"The result of {original_task} is {fut.result()}")
        
    
# #     for name,group in df.groupby(df.iloc[:, 0]):
# #         pc_Index.update({name: pCIdx_base(group)})
        
#     pc_Index_df = pd.DataFrame.from_dict(pc_Index,orient = 'index')
    pc_Index_df = df.groupby(df.iloc[:,0]).parallel_apply(pCIdx_base)
    
    return pc_Index_df

# Calculate the weights previously
def pCIdx_weight(df: pd.DataFrame) -> pd.Series:
    """

    Args:
        df: (pd.DataFrame)
            -  A DataFrame record drug_id,true_values, predict_values.
            `[drug_id,true_values,predict_values]`

    Returns:
        The weight w_d is calculated from the empirical null distribution
        of pc_d

    """
    x1 = np.asanyarray(df.iloc[:, 1], dtype='f8')
    pc_index_rand = []
    """
    def generate_rand_sample(_, df_):
        df_tmp = pd.DataFrame(
            {'drug_id': df_.iloc[:, 0],
             'true': np.asanyarray(df_.iloc[:, 1],dtype='f8'),
             'predict': np.random.rand(len(df_))
             }
        )
        return pCIdx(df_tmp)
    pc_index_rand = map(generate_rand_sample,[0]*10000,df)
    """
    for i in range(100):
        # np.random.seed(i)
        x1_rand = np.random.rand(len(x1))
        # x1_rand_rank = rankdata(x1_rand)
        df_tmp = pd.DataFrame({'drug_id': df.iloc[:, 0],
                               'true': x1,
                               'predict': x1_rand})
        pc_index_rand.append(pCIdx_base(df_tmp))

    mu_rand = np.median(pc_index_rand)
    sigma_rand = np.std(pc_index_rand)
    df_std = pd.DataFrame({'drug_id': df.iloc[:, 0],
                           'true': x1,
                           'predict': x1})
    pc_index_std = pCIdx_base(df_std)
    
    return (pc_index_std - mu_rand) / sigma_rand


def wpCIdx(df: pd.DataFrame, typ=None) -> float:
    """Weighted average of the pc-index scores for method M
    Args:
        df: (pd.DataFrame)
            -  A DataFrame record drug_id,true_values, predict_values.
            `[drug_id,true_values,predict_values]`

    Returns:
        Weighted average of the pc-index scores for method M
    """
    if typ == 'LNIC50':
        weights_inner = pd.read_csv("path_to_weight_file/GDSCrel82_V1_seqexp_wpcIndex_weight10000_LNIC50.csv", header=0, index_col =0)
    elif typ == 'AUC':
        weights_inner = pd.read_csv("path_to_weight_file/GDSCrel82_V1_seqexp_wpcIndex_weight10000_AUC.csv", header=0, index_col =0)
    elif typ == 'IC50':
        weights_inner = pd.read_csv("path_to_weight_file/GDSCrel82_V1_seqexp_wpcIndex_weight10000_IC50.csv", header=0, index_col =0)
    else:
        raise Exception('Specify the typ! LNIC50/AUC/IC50')

    pc_index = pCIdx(df)

    df_max = df.copy()
    df_max.iloc[:,2] = df_max.iloc[:,1]
    pc_index_max = pCIdx(df_max)

    df_min = df.copy()
    df_min.iloc[:,2] = - df_max.iloc[:,1]
    pc_index_min = pCIdx(df_min)
    
    # weights_inner = pd.read_csv("drug_weight1000repeat.csv", header=0, index_col =0)
    weights = weights_inner.loc[pc_index.index,:]
    # weights = df.groupby(df.iloc[:, 0]).apply(pCIdx_weight)
    # df_tmp = pd.DataFrame({
    #     'pc_index': pCIdx(df),
    #     "weights": df.groupby(df.iloc[:, 0]).apply(pCIdx_weight)
    # })
    wpc = pc_index.dot(weights) / np.sum(weights)
    wpc_max = pc_index_max.dot(weights) / np.sum(weights)
    wpc_min = pc_index_min.dot(weights) / np.sum(weights)

    return wpc, wpc_min, wpc_max




def norm_wpCIdx(df: pd.DataFrame, typ=None) -> float:
    """Normalized weight average of the pc-index scores for method M
    Args:
        df: (pd.DataFrame)
            -  A DataFrame record drug_id,true_values, predict_values.
            `[drug_id,true_values,predict_values]`

    Returns:
        Normalized weight average of the pc-index scores for method M
    """
    wpc_index, min_wpc_index, max_wpc_index = wpCIdx(df, typ)
    return (wpc_index - min_wpc_index) / (max_wpc_index - min_wpc_index)

def roc_auc_base(df, axis=0):
    """ROC-AUC
    """
    x1 = np.asanyarray(df.iloc[:, 1], dtype='f8')
    if x1.sum() == len(x1) or x1.sum() == 0:
        return 0.5
    else:
        x2 = np.asanyarray(df.iloc[:, 2], dtype='f8')
        return roc_auc_score(x1,x2)

def roc_auc(df):
    return df.groupby(df.iloc[:, 0]).apply(roc_auc_base)


# Clinical results analysis
def wilcox_pval(df, method):
#     methods = ['CRDNN', 'DrugCell', 'PaccMann', 'TGSA', 'VAEN']
#     pval_list = []
#     for method in methods:
    x1 = df[df['response_RS'] == 'Non-response'][method].values
    x2 = df[df['response_RS'] == 'Response'][method].values
    
    if np.abs(np.mean(x1) - np.mean(x2)) < 1e-6:
        pval = 1
    else:
        _ , pval = stats.mannwhitneyu(x1,x2)
#         pval_list.append(pval)    
    return pd.Series([pval], index = [method])


def effect_size(df, method):
    x1 = df[df['response_RS'] == 'Non-response'][method].values
    x2 = df[df['response_RS'] == 'Response'][method].values
    mean_diff = np.mean(x1) - np.mean(x2)
    return pd.Series([mean_diff], index = [method])

'''
drug_character['kurtosis'] = gdscv1_dr.groupby('DRUG_NAME')['LN_IC50'].apply(stats.kurtosis)
drug_character['skew'] = gdscv1_dr.groupby('DRUG_NAME')['LN_IC50'].apply(stats.skew)
drug_character['bimodality'] = drug_character.eval('(skew**2 + 1)/(kurtosis + (3*(sample_size - 1)**2) / ((sample_size - 2) * (sample_size - 3))) ')

def KS_test(s):
    sta, pval = stats.ks_2samp(s, gdscv1_dr['LN_IC50'])
    return sta
    
drug_character['KS_test'] = gdscv1_dr.groupby('DRUG_NAME')['LN_IC50'].apply(KS_test)
'''