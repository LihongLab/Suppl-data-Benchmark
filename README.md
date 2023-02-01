A systematic assessment of deep learning methods for drug response prediction: from in-vitro to clinical application

```powershell
│  README.md
│
├─dependencies # Environment required
│      crdnn_env.yml
│      drugcell_env.yml
│      moli_env.yml
│      paccmann_env.yml
│      tgsa_env.yml
│      vaen_env.yml
│      vaen_r_env.yml
│
├─hyper_grid # Hyperparameters search grid
│      hypergrid_crdnn.json
│      hypergrid_drugcell.json
│      hypergrid_moli.json
│      hypergrid_paccamnn.json
│      hypergrid_tgsa.json
│      hypergrid_vaen.json
│
└─utils # Utility scripts
        benchmark_metrics.py # Implement of benchmark metrics
        GDSCrel82_BinaryIC50.R # Script for binarization of IC50s
        generate_NWPC_weights.py # Generate NWPC weights
        smiles2graph.py # Generate molecular graphs from SMILES strings 
```

# Methods for assessments 

The environments and hyperparameter grids used in this study are available on `\dependencies` and `\hyper_grid`.

|              | Paradigm | **Tumor feature** | **Drug feature** | Repository |
| ------------ | --------------------- | ---------------- | ------------- | ------------- |
| **DrugCell**[^1] | MDL                  | M                     | Morgan FP        | https://github.com/idekerlab/DrugCell |
| **PaccMann**[^2] | MDL | E<sup>a</sup> | SMILES           | https://github.com/PaccMann/paccmann_predictor |
| **TGSA**[^3] | MDL | E<sup>a</sup>, M, C | Molecular graph  | https://github.com/violet-sto/TGSA |
| **CRDNN**[^4] | SDL  | E<sup>a</sup> | -                | https://github.com/TeoSakel/deep-drug-response |
| **VAEN**[^5] | SDL          | E<sup>b</sup> | -                | https://github.com/bsml320/VAEN |
| **MOLI**[^6] | SDL       | E<sup>a</sup>, M, C | -                | https://github.com/hosseinshn/MOLI |

*__MDL__: Multi Drug Learning; __SDL__: Single Drug Learning*  
*__E__: expression profiles; __M__: mutation status; __C__: Copy number variation; __FP__: Fingerprint; __SMILES__: Simplified Molecular-Input Line-Entry System*  
*__a__: z-score standardization*  
*__b__: rank normalization*  

# Dataset

* GDSC

  * Cell line genomic data were downloaded from [Cell Model Passports](https://cellmodelpassports.sanger.ac.uk/downloads)
  * Drug response data were downloaded from [GDSC bulk data dowload](ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.2) (release-8.2)

* CCLE

  * Cell line genomic data and drug response data were downloaded from [DepMap portal](https://depmap.org/portal/download/) (2021Q4)

* TCGA

  * Clinical response were curated by Ding _et al._[^7] and downloaded from [it portal](http://bioinfo.au.tsinghua.edu.cn/member/zding/drug_response/). 
  * Cell line genomic data were downloaded from [GDC client](https://portal.gdc.cancer.gov/).

* Bortezomib

  * Patients with relapsed myeloma enrolled in phase 2 and phase 3 clinical trials of bortezomib.[^8]  
  * Gene expression data were downloaded from GEO:GSE9782.


The curated datasets are available on https://zenodo.org/record/7060305#.YxnOEHZByUn

# Benchmark Metrics

The nine metrics are adapted from Chen _et al._[^9] and the implement is available on `utils\benchmark_metrics.py`

|Metrics                                            |Range  |Based  on |Formula                                                      |
| -------------------------------------------------- | ------ | --------- | ------------------------------------------------------------ |
|Root-mean-square error (RMSE)                                             |R+     |Value     |$$RMSE(\boldsymbol{y,\hat{y}}) = \sqrt{\frac{\sum_{i}(y_i-\hat{y}_i)^2}{n}}$$ |
|L-RMSE,  R-RMSE,  LR-RMSE                            |R+     |Value     | <img src=".\src\l_rmse.png" alt="l_rmse" style="zoom:40%;" /><br><img src=".\src\r_rmse.png" alt="r_rmse" style="zoom:40%;" /><br><img src=".\src\lr_rmse.png" alt="lr_rmse" style="zoom:40%;" /> <br> <img src=".\src\formula1.png" alt="formula1" style="zoom:40%;" /> |
|Pearson Correlation Coefficient (PCC)              |[-1,1] |Value     |$$PCC(\boldsymbol{y,\hat{y}})=\frac{\sum_i(\hat{y}_i - \mu(\boldsymbol{\hat{y}}))(y_i - \mu(\boldsymbol{y}))}{\sqrt{\sum_i(\hat{y}_i - \mu(\boldsymbol{\hat{y}}))^2}\sqrt{\sum_i(y_i - \mu(\boldsymbol{y}))^2}}$$ |
|Spearman Correlation Coefficient (SCC)             |[-1,1] |Rank      |$$SCC(\boldsymbol{y,\hat{y}})=PCC(\boldsymbol{r(y),r(\hat{y})})$$ |
|Normalized Discounted Cumulative Gain (NDCG)       |[0,1]  |Rank      |$$DCG(\boldsymbol{y,\hat{y}}) = \sum_{i=1}^{n}\frac{2^{-y_i}}{\log_2 (r(\hat{y}_i)+1 )}$$ <br> $$NDCG(\boldsymbol{y,\hat{y}}) = \frac{DCG(\boldsymbol{y,\hat{y}})}{DCG(\boldsymbol{y,y})}$$                                |
|Probabilistic C-index (PC), Normalized  Weighted Probabilistic C-index (NWPC) |[0,1]  |Rank      |<img src=".\src\pc.png" alt="pc" style="zoom:40%;" /> <br> <img src=".\src\formula2.png" alt="formula2" style="zoom:40%;"/> <br> $$erf(a) = \frac{2}{\sqrt{\pi}}\int_0^a e^{-t^2} dt$$ <br> $$WPC(M) = \frac{\sum_{d} w_d \cdot PC_d}{\sum_{d} w_d}$$ <br> $$NWPC = \frac{WPC - WPC_{min}}{WPC_{max} - WPC_{min}}$$ |
|ROC-AUC                                            |[0,1]  |Value     |ROC-AUC  provides an aggregate measure of performance across all possible  classification thresholds |

$y$: the observed response values; $\hat{y}$: the predicted response values; $n$: the number of samples; $r(\hat{y}_i)$ is the position of $\hat{y}_i$ on the sorted $\hat{y}$ in ascending order.

# Pipeline

## Assessment on cell line data
### 1. Data Preparation

* Transcriptome profiles: GDSC_EXP.csv

  * For PaccMann, TGSA, CRDNN and MOLI, gene expression matrix was z-score standardization.
  * For VAEN, gene expression matrix was rank normalization.

* Mutation profiles: GDSC_MUT.csv

* Copy number profiles: GDSC_CNV.csv

* Chemical representations:

  * SMILES: GDSC_drugannot.txt `CanonicalSMILESrdkit `

  * Morgan Fingerprints:

    ```Python
    from rdkit.Chem import AllChem
    mfp_ls =
    [
        list(
            AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(smiles),
                2,
                nBits=2048
            )
        ) for smiles in mfp['CanonicalSMILESrdkit']
    ]
    ```

  * Molecular Graph: 

    ```python
    from utils import smiles2graph
    smiles2graph.save_drug_graph(
        'GDSC_drugannot.txt',
        'GDSC'
    )
    ```

* Drug Response: GDSC_DR.txt

  * The script for binarization of IC50s: `utils\GDSCrel82_BinaryIC50.R`.

### 2. Model Construction

#### Multi Drug Learning (MDL)

Input:

* $D$, a set of labeled tuples;
* $t$, the number of repeats (default: 5);
* $k$, the number of folds for cross-validation (default: 5);
* $p$, the number of hyperparameters combination (default: 30);
* a MDL model. 

```bash
for i = 1 to t do
	stratify D on the tissue origin into k folds, $D = (D_1,...,D_k)$
	for j = 1 to k do
		take the $D_j$ as test set
		take the $D_{(j+1) \% k}$ as validation set
		take the remaining folds as training set
		apply max-min transformation of IC50s on training, validation and test set
		for n = 1 to p do
			select a random combination from hyperparameter grid
			use training set to derive a model, $M_n$
			use validation set to evaluate $M_n$
			retain the evaluation score, Pearson correlation coefficient
		done
		select the combination of hyperparameter with the highest evaluation score
		use training and validation set to fit the model
		return the predicted results on test set
	done
done    
```

#### Single Drug Learning (SDL)

Input:

* $D$, a set of labeled tuples;
* $t$, the number of repeats (default: 5);
* $k$, the number of folds for cross-validation (default: 5);
* $d$, the number of drugs in the dataset;
* $p$, the number of hyperparameters combination (default: 30);
* a SDL model.

```bash
for i = 1 to t do
	stratify D on the tissue origin into k folds, $D = (D_1,...,D_k)$
	for j = 1 to k do
		take the $D_j$ as test set
		take the $D_{(j+1) \% k}$ as validation set
		take the remaining folds as training set
		apply max-min transformation of IC50s on training, validation and test set
		for m = 1 to d do
			sample test set with drug m to obtain drug specific test set
			sample validation set with drug m to obtain drug specific validation set
			sample training set with drug m to obtain drug specific training set
			for n = 1 to p do
                select a random combination from hyperparameter grid
                use training set to derive a model, $M_n$
                use validation set to evaluate $M_n$
                retain the evaluation score, Pearson correlation coefficient
            done
            select the combination of hyperparameter with the highest evaluation score
            use training and validation set to fit the model
            return the predicted results on drug specific test set
         done
	done
done
```

### 3. Model Assessment

* Concatenate each fold to generate the list for assessment in the format of  `['drug','true', 'predict','binary_response','repeat_fold','cell_line','repeat']`

* Overall performance

```python
import pandas as pd
from utils import benchmark_metrics as bcm
df # result dataframe in the form of ['drug','true', 'predict','binary_response','repeat_fold','cell_line','repeat']
overall_performance = pd.DataFrame(
    {
    'PCC': bcm.pearson_corr(df[['repeat_fold','true', 'predict']]), 
    'SCC': bcm.spearman_corr(df[['repeat_fold','true', 'predict']]), 
    'RMSE': bcm.rmse(df[['repeat_fold','true', 'predict']]), 
    'L_RMSE': bcm.l_rmse(df[['repeat_fold','true', 'predict']]), 
    'R_RMSE': bcm.r_rmse(df[['repeat_fold','true', 'predict']]), 
    'LR_RMSE':bcm.lr_rmse(df[['repeat_fold','true', 'predict']]), 
    'NDCG':bcm.ndcg(df[['repeat_fold','true', 'predict']]),
    'NWPC': bcm.norm_wpCIdx(df[['repeat_fold','true', 'predict']])
    }
)
```

* Performance on single-drug level

```python
import pandas as pd
from utils import benchmark_metrics as bcm
df # result dataframe in the form of ['drug','true', 'predict','binary_response','repeat_fold','cell_line','repeat']
singledrug_performance = {
    'PCC': df.groupby('repeat')[['drug','true', 'predict']].apply(bcm.pearson_corr), 
    'SCC': df.groupby('repeat')[['drug','true', 'predict']].apply(bcm.spearman_corr), 
    'RMSE': df.groupby('repeat')[['drug','true', 'predict']].apply(bcm.rmse), 
    'L_RMSE': df.groupby('repeat')[['drug','true', 'predict']].apply(bcm.l_rmse), 
    'R_RMSE': df.groupby('repeat')[['drug','true', 'predict']].apply(bcm.r_rmse), 
    'LR_RMSE':df.groupby('repeat')[['drug','true', 'predict']].apply(bcm.lr_rmse), 
    'ROC_AUC':df.groupby('repeat')[['drug','binary_response', 'predict']].apply(bcm.roc_auc),
    'NDCG':df.groupby('repeat')[['drug','true', 'predict']].apply(bcm.ndcg),
    'PC': df.groupby('repeat')[['drug','true', 'predict']].apply(bcm.pCIdx)
}

```

## Assessment on clinical data

### 1. Data Preparation

#### 1.1 Cell line data

* Transcriptome profiles: GDSC_EXP.csv

  * For PaccMann, TGSA, CRDNN and MOLI, gene expression matrix was z-score standardization.
  * For VAEN, gene expression matrix was rank normalization.

* Mutation profiles: GDSC_MUT.csv

* Copy number profiles: GDSC_CNV.csv

* Chemical representations:

  * SMILES: GDSC_drugannot.txt `CanonicalSMILESrdkit `

  * Morgan Fingerprints:

    ```Python
    from rdkit.Chem import AllChem
    mfp_ls =
    [
        list(
            AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(smiles),
                2,
                nBits=2048
            )
        ) for smiles in mfp['CanonicalSMILESrdkit']
    ]
    ```

  * Molecular Graph: 

    ```python
    from utils import smiles2graph
    smiles2graph.save_drug_graph(
        'GDSC_drugannot.txt',
        'GDSC'
    )
    ```

* Drug Response: GDSC_DR.txt

  * The script for binarization of IC50s: `utils\GDSCrel82_BinaryIC50.R`.

#### 1.2 Patient data

* Transcriptome profiles: TCGA_EXP.csv

  * For PaccMann, TGSA, CRDNN and MOLI, gene expression matrix was z-score standardization.
  * For VAEN, gene expression matrix was rank normalization.
  * ComBat [^10] was used to adjust expression profiles of patients with the GDSC cell line dataset.
* Mutation profiles: TCGA_MUT.csv
* Copy number profiles: TCGA_CNV.csv
* Chemical representations:

  * SMILES: TCGA_drugannot.txt `CanonicalSMILESrdkit `

  * Morgan Fingerprints:

    ```Python
    from rdkit.Chem import AllChem
    mfp_ls =
    [
        list(
            AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(smiles),
                2,
                nBits=2048
            )
        ) for smiles in mfp['CanonicalSMILESrdkit']
    ]
    ```

  * Molecular Graph: 

    ```python
    from utils import smiles2graph
    smiles2graph.save_drug_graph(
        'TCGA_drugannot.txt',
        'TCGA'
    )
    ```
* Drug Response: TCGA_DR.txt

### 2. Model Construction
#### Multi Drug Learning (MDL)

Input:

* $D$, a set of labeled tuples;

* $h$, a set of hyperparameters with the highest evaluation score;

* a MDL model.

```bash
use D, h to fit the model M 
return the predicted results on patient data
```

#### Single Drug Learning (SDL)

Input:

* $D$, a set of labeled tuples;

* $h$, sets of hyperparameters with the highest evaluation score for each drug

* $d$, the number of drugs in the dataset;

* a SDL model.

```bash
for m = 1 to d do
	sample D to obtain drug specific dataset, $D_m$
	use $D_m$, $h_m$ to fit the model $M_m$
	return the predicted results on patient data
done
```


### 3. Model Assessment

* Concatenate the result of each drug to generate the list for assessment in the format of `['drug', 'response_RS', 'response_01', 'predict']`

* Performance
```python
import pandas as pd
from utils import benchmark_metrics as bcm
df # result dataframe in the form of ['drug', 'response_RS', 'response_01', 'predict']
singledrug_performance = {
    'pval': df.groupby('drug')[['response_RS', 'predict']].apply(bcm.wilcox_pval), 
    'effectsize': df.groupby('drug')[['response_RS', 'predict']].apply(bcm.effect_size), 
    'roc_auc':bcm.roc_auc(df[['drug','response_01','predict']])
}
```



Reference:

[^1]:Kuenzi, B. M., Park, J., Fong, S. H., Sanchez, K. S., Lee, J., Kreisberg, J. F., Ma, J., & Ideker, T. (2020). Predicting Drug Response and Synergy Using a Deep Learning Model of Human Cancer Cells. Cancer Cell, 38(5), 672-684.e676. https://doi.org/10.1016/j.ccell.2020.09.014

[^2]:Manica, M., Oskooei, A., Born, J., Subramanian, V., Saez-Rodriguez, J., & Rodriguez Martinez, M. (2019). Toward Explainable Anticancer Compound Sensitivity Prediction via Multimodal Attention-Based Convolutional Encoders. Mol Pharm, 16(12), 4797-4806. https://doi.org/10.1021/acs.molpharmaceut.9b00520

[^3]:Zhu, Y., Ouyang, Z., Chen, W., Feng, R., Chen, D. Z., Cao, J., & Wu, J. (2021). TGSA: Protein-Protein Association-Based Twin Graph Neural Networks for Drug Response Prediction with Similarity Augmentation. Bioinformatics. https://doi.org/10.1093/bioinformatics/btab650

[^4]:Sakellaropoulos, T., Vougas, K., Narang, S., Koinis, F., Kotsinas, A., Polyzos, A., Moss, T. J., Piha-Paul, S., Zhou, H., Kardala, E., Damianidou, E., Alexopoulos, L. G., Aifantis, I., Townsend, P. A., Panayiotidis, M. I., Sfikakis, P., Bartek, J., Fitzgerald, R. C., Thanos, D., Mills Shaw, K. R., Petty, R., Tsirigos, A., & Gorgoulis, V. G. (2019). A Deep Learning Framework for Predicting Response to Therapy in Cancer. Cell Rep, 29(11), 3367-3373 e3364. https://doi.org/10.1016/j.celrep.2019.11.017

[^5]:Jia, P., Hu, R., Pei, G., Dai, Y., Wang, Y. Y., & Zhao, Z. (2021). Deep generative neural network for accurate drug response imputation. Nat Commun, 12(1), 1740. https://doi.org/10.1038/s41467-021-21997-5

[^6]:Sharifi-Noghabi, H., Zolotareva, O., Collins, C. C., & Ester, M. (2019). MOLI: multi-omics late integration with deep neural networks for drug response prediction. Bioinformatics, 35(14), i501-i509. https://doi.org/10.1093/bioinformatics/btz318

[^7]: Ding, Z., Zu, S., & Gu, J. (2016). Evaluating the molecule-based prediction of clinical drug responses in cancer. Bioinformatics, 32(19), 2891-2895. https://doi.org/10.1093/bioinformatics/btw344
[^8]:Mulligan, G., Mitsiades, C., Bryant, B., Zhan, F., Chng, W. J., Roels, S., Koenig, E., Fergus, A., Huang, Y., Richardson, P., Trepicchio, W. L., Broyl, A., Sonneveld, P., Shaughnessy, J. D., Jr., Bergsagel, P. L., Schenkein, D., Esseltine, D. L., Boral, A., & Anderson, K. C. (2007). Gene expression profiling and correlation with outcome in clinical trials of the proteasome inhibitor bortezomib. Blood, 109(8), 3177-3188. https://doi.org/10.1182/blood-2006-09-044974
[^9]: Chen, J., & Zhang, L. (2020). A survey and systematic assessment of computational methods for drug response prediction. Brief Bioinform. https://doi.org/10.1093/bib/bbz164
[^10]: Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics, 8(1), 118-127. 

-------------------------------------------------------------------------------------------------------------------------
**Citation**: Please cite our paper if you find this benchmarking work is helpful to your research. **Bihan Shen\#, Fangyoumin Feng\#, Kunshi Li, Ping Lin, Liangxiao Ma, Hong Li\*.** [**A systematic assessment of deep learning methods for drug response prediction: from in vitro to clinical applications**. *Briefings in Bioinformatics* 24, 2 (2023).](https://doi.org/10.1093/bib/bbac605)

