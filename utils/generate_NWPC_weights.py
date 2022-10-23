# os.getcwd(): 'path_to_weight_file'
# import my_metrics.bbi_metrics as bbi
# python38

import os
import numpy as np
import pandas as pd
import time
import concurrent.futures
# import time
from scipy.stats import rankdata
# import pandas as pd
# import numpy as np
# from sklearn.metrics import ndcg_score
from scipy import integrate
import math

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
    return df.groupby(df.iloc[:, 0]).apply(pCIdx_base)

'''
# @jit(nopython=True)
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
'''
def generate_null(x1,df,i):
    np.random.seed(i)
    x1_rand = np.random.rand(len(x1))
    # x1_rand_rank = rankdata(x1_rand)
    df_tmp = pd.DataFrame({'drug_id': df.iloc[:, 0],
                           'true': x1,
                           'predict': x1_rand})
    return pCIdx_base(df_tmp)

def pCIdx_weight(df: pd.DataFrame):
    """

    Args:
        df: (pd.DataFrame)
            -  A DataFrame record drug_id,true_values, predict_values.
            `[drug_id,true_values,predict_values]`
            raw data, without any scale

    Returns:
        The weight w_d is calculated from the empirical null distribution
        of pc_d

    """
    # start = time.time()

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
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        results = [executor.submit(generate_null,x1,df,k) for k in range(1000)]

        for f in concurrent.futures.as_completed(results):
            pc_index_rand.append(f.result())

    mu_rand = np.median(pc_index_rand)
    sigma_rand = np.std(pc_index_rand)
    df_std = pd.DataFrame({'drug_id': df.iloc[:, 0],
                           'true': x1,
                           'predict': x1})
    pc_index_std = pCIdx_base(df_std)

    return (pc_index_std - mu_rand) / sigma_rand


def wpCIdx(df: pd.DataFrame) -> float:
    """Weighted average of the pc-index scores for method M
    Args:
        df: (pd.DataFrame)
            -  A DataFrame record drug_id,true_values, predict_values.
            `[drug_id,true_values,predict_values]`

    Returns:
        Weighted average of the pc-index scores for method M
    """
    pc_index = pCIdx(df)

    df_max = df.copy()
    df_max.iloc[:,2] = df_max.iloc[:,1]
    pc_index_max = pCIdx(df_max)

    df_min = df.copy()
    df_min.iloc[:,2] = - df_max.iloc[:,1]
    pc_index_min = pCIdx(df_min)

    weights = df.groupby(df.iloc[:, 0]).apply(pCIdx_weight)
    # df_tmp = pd.DataFrame({
    #     'pc_index': pCIdx(df),
    #     "weights": df.groupby(df.iloc[:, 0]).apply(pCIdx_weight)
    # })
    wpc = pc_index.dot(weights) / np.sum(weights)
    wpc_max = pc_index_max.dot(weights) / np.sum(weights)
    wpc_min = pc_index_min.dot(weights) / np.sum(weights)

    return wpc, wpc_min, wpc_max




def norm_wpCIdx(df: pd.DataFrame) -> float:
    """Normalized weight average of the pc-index scores for method M
    Args:
        df: (pd.DataFrame)
            -  A DataFrame record drug_id,true_values, predict_values.
            `[drug_id,true_values,predict_values]`

    Returns:
        Normalized weight average of the pc-index scores for method M
    """
    wpc_index, min_wpc_index, max_wpc_index = wpCIdx(df)
    return (wpc_index - min_wpc_index) / (max_wpc_index - min_wpc_index)


if __name__ == '__main__':
    # # Calculate drug weight for LN_IC50:
    # print('Start IC50:')
    # print(time.ctime())
    # df_full = pd.read_csv("/picb/bigdata/project/shenbihan/1-Benchmark/ProcessedData/paccmann_torch/full/GDSC1_fitted_dose_response_25Feb20_cellid_drugid_fullinfo_SIMPLE_20210909.csv",header=0)
    # df_full_for = df_full[['drug_id','LN_IC50','AUC']]
    # print(len(df_full_for))
    # print(df_full_for.columns)
    
    # drug_weight = dict() 
    # for name,group in df_full_for.groupby(df_full_for.columns[0]):
    #     start = time.time()
    #     print('drug:',name,',len:',len(group))
    #     weight = pCIdx_weight(group)
    #     drug_weight.update({name: weight})
    #     print(weight)
    #     end = time.time()
    #     print((end -start) /60, 'mins' )
    
    # pd.DataFrame.from_dict(drug_weight,orient = 'index').to_csv('drug_weight10000_LNIC50.csv')
    # print(time.ctime())
    
    # # Calculate drug weight for AUC:
    # print('Start AUC:')
    # df_full_for = df_full[['drug_id','AUC','LN_IC50']]
    # print(len(df_full_for))
    # print(df_full_for.columns)

    # drug_weight = dict() 
    # for name,group in df_full_for.groupby(df_full_for.columns[0]):
    #     start = time.time()
    #     print('drug:',name,',len:',len(group))
    #     weight = pCIdx_weight(group)
    #     drug_weight.update({name: weight})
    #     print(weight)
    #     end = time.time()
    #     print((end -start) /60, 'mins' )
    
    # pd.DataFrame.from_dict(drug_weight,orient = 'index').to_csv('drug_weight10000_AUC.csv')
    # print(time.ctime())

    # Calculate drug weight for IC50:
    print('Start AUC:')
    print(time.ctime())
    df_full = pd.read_csv("path_to_response_file/GDSCrel82_V1_DR_LNIC50andAUC_OMICS_expseq_DR.csv",header=0)
    df_full_for = df_full[['DRUG_NAME','AUC','LN_IC50']]
    print(df_full_for.head(5))
    # df_full_for.loc[:,'LN_IC50'] = np.exp(df_full_for['LN_IC50'])
    print(df_full_for.head(5))
    print(len(df_full_for))
    print(df_full_for.columns)
    
    drug_weight = dict() 
    for name,group in df_full_for.groupby(df_full_for.columns[0]):
        start = time.time()
        print('drug:',name,',len:',len(group))
        weight = pCIdx_weight(group)
        drug_weight.update({name: weight})
        print(weight)
        end = time.time()
        print((end -start) /60, 'mins' )
    
    pd.DataFrame.from_dict(drug_weight,orient = 'index').to_csv('GDSCrel82_V1_seqexp_wpcIndex_weight10000_AUC.csv')
    print(time.ctime())


#     weight = pCIdx_weight(df_full_for)
#     # drug_weight.update({name: weight})
#     print(weight)
#     end = time.time()
#     print((end -start) /60, 'mins' )
    
