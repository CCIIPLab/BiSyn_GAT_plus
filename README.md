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
**Please kindly cite our paper if this paper and the code are helpful.**
```
@inproceedings{liang-etal-2022-bisyn,
    title = "{B}i{S}yn-{GAT}+: Bi-Syntax Aware Graph Attention Network for Aspect-based Sentiment Analysis",
    author = "Liang, Shuo  and
      Wei, Wei  and
      Mao, Xian-Ling  and
      Wang, Fei  and
      He, Zhiyong",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.144",
    pages = "1835--1848",
    abstract = "Aspect-based sentiment analysis (ABSA) is a fine-grained sentiment analysis task that aims to align aspects and corresponding sentiments for aspect-specific sentiment polarity inference. It is challenging because a sentence may contain multiple aspects or complicated (e.g., conditional, coordinating, or adversative) relations. Recently, exploiting dependency syntax information with graph neural networks has been the most popular trend. Despite its success, methods that heavily rely on the dependency tree pose challenges in accurately modeling the alignment of the aspects and their words indicative of sentiment, since the dependency tree may provide noisy signals of unrelated associations (e.g., the {``}conj{''} relation between {``}great{''} and {``}dreadful{''} in Figure 2). In this paper, to alleviate this problem, we propose a Bi-Syntax aware Graph Attention Network (BiSyn-GAT+). Specifically, BiSyn-GAT+ fully exploits the syntax information (e.g., phrase segmentation and hierarchical structure) of the constituent tree of a sentence to model the sentiment-aware context of every single aspect (called intra-context) and the sentiment relations across aspects (called inter-context) for learning. Experiments on four benchmark datasets demonstrate that BiSyn-GAT+ outperforms the state-of-the-art methods consistently.",
}

```
