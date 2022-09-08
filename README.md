# Supplementary material



Supplementary Data:  https://zenodo.org/record/7060305#.YxnOEHZByUn



## GDSC dataset

**Table S3.** GDSC gene expression profiles for 966 cancer cell lines, where each column represents a cell line in the form of its name and tissue collection site, and each row represents a gene in the form of the HGNC symbol.

 

**Table S4.** GDSC gene mutation profiles for 966 cancer cell lines, where each column represents a cell line in the form of its name and tissue collection site, and each row represents a gene in the form of the HGNC symbol. The wild type is coded as 1 and the wild type as 0.

 

**Table S5.** GDSC copy number variation profiles for 966 cancer cell lines, where each column represents a cell line in the form of its name and tissue collection site, and each row represents a gene in the form of the HGNC symbol. The copy-neutral is coded as 0 and the deletion or amplification as 1.

 

**Table S6.** GDSC drug response data for 966 cancer cell lines and 282 drugs in the form of the natural logarithm of the IC50 readout. The first column shows the cell line name and tissue collection site, the second column shows the drug name, and the third column shows the drug response readout.

 

**Table S7.** GDSC annotations for 282 drugs include drug name, PubChem CID, PubChem canonical SMILES, Rdkit canonical SMILES, Target Pathway, standard deviation, bimodality coefficient and density coverage.



## TCGA dataset

**Table S8.** TCGA gene expression profiles, where each column represents a patient in the form of TCGA patient ID, and each row represents a gene in the form of the HGNC symbol.

 

**Table S9.** TCGA gene mutation profiles, where each column represents a patient in the form of TCGA patient ID, and each row represents a gene in the form of the HGNC symbol. The wild type is coded as 1 and the wild type as 0.

 

**Table S10.** TCGA copy number variation profiles, where each column represents a patient in the form of TCGA patient ID, and each row represents a gene in the form of the HGNC symbol. The copy-neutral is coded as 0 and the deletion or amplification as 1.

 

**Table S11.** TCGA clinical response data. The first column shows the TCGA patient ID, the second column shows the drug name, the third column shows the clinical response category, the fourth column shows the cancer type, and the last column shows the clinical label as responder or non-responder.



## Benchmark Metrics

| File name                | Usage                                                        |
| ------------------------ | ------------------------------------------------------------ |
| benchmark_metrics.py     | Implement of benchmark metrics, where `xx_base`  calculates the overall performance and `xx` calculates the single-drug level performance on the corresponding metric. |
| generate_NWPC_weights.py | The script to calculate the drug weight for NWPC             |

