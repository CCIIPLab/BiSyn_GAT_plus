# Bi-Syntax Aware Graph Attention Network


This repository contains Pytorch implementation for "[BiSyn-GAT+: Bi-Syntax Aware Graph Attention Network for Aspect-based Sentiment Analysis](https://arxiv.org/abs/2204.03117)" (Findings of ACL 2022)

## Get Start
1. Prepare data
   
   We follow the dataset setting in https://github.com/muyeby/RGAT-ABSA, and provide the parsed data at directory **data**

2. Training
   
   ```
   bash run_bash/run_MAMS.sh
   ```

## Citation
**Please kindly cite our paper if this paper and the code are helpful**
```
@inproceedings{Liang-etal-2022-bisyn,
	title = "Bi{S}yn-{GAT}+: {B}i-{S}yntax {A}ware {G}raph {A}ttention {N}etwork for {A}spect-based {S}entiment {A}nalysis",
	author = "Liang, Shuo  and
	Wei, Wei  and
	Mao, Xianling  and
	Wang, Fei  and
	He, Zhiyong",
	booktitle = "Findings of the Association for Computational Linguistics",
	year = "2022",
}
```
