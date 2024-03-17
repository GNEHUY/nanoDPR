# nanoDPR
[Dense Passage Retrieval (DPR)](https://arxiv.org/pdf/2004.04906.pdf) is a widely-recognized technique and the foundation of retrieval-augmented LLM. The authors of the paper have provided an excellent open-source [repository](https://github.com/facebookresearch/DPR). However, the original repository is primarily driven by academic research and includes numerous configurable options. In contrast, this repository aims to offer a simplified replication of the DPR model on the Natural Questions dataset, allowing for a clear and straightforward understanding of DPR without compromising any details. With approximately 300 lines of code, we can train a DPR from scratch and achieve results comparable to those presented in the original paper.

In short, this repo enables:
- training a dense retriever from scratch on Natural Question dataset
- loading the original checkpoint provided by the official repo
- evaluating dense retriever

## Requirements
```bash
# install pytorch according to the cuda version (https://pytorch.org/get-started/previous-versions/)
# install faiss (https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
pip install transformers==4.30.2 accelerate==0.20.3 wandb wget spacy
```

## Data
```python
python utils/download_data.py --resource data.wikipedia_split.psgs_w100
python utils/download_data.py --resource data.retriever.nq
python utils/download_data.py --resource data.retriever.qas.nq
```

## Training from scratch
First configure distributed setting and wandb setting:
```bash
accelerate config
wandb login
```
Then launch training with:
```bash
accelerate launch train_dpr.py
```
After training, we would get a trained **query encoder** and a **doc encoder**. 


## Evaluation
To evaluate the performance of retriever on the Natural Question dataset, firstly use **doc encoder** to encode all wikipedia passages:
```
## for nanoDPR
accelerate launch doc2embedding.py \
    --pretrained_model_path your/own/nanoDPR/model \
    --output_dir embedding/nanoDPR

## for official DPR
accelerate launch doc2embedding.py \
    --pretrained_model_path facebook/dpr-ctx_encoder-single-nq-base \
    --output_dir embedding/DPR
```
Then test DPR with:
```
## for nanoDPR
python test_dpr.py --embedding_dir embedding/nanoDPR --pretrained_model_path your/own/nanoDPR/model

## for official DPR
python test_dpr.py --embedding_dir embedding/DPR --pretrained_model_path facebook/dpr-question_encoder-single-nq-base
```
Here we provide our trained **query encoder** and **doc encoder** [here](https://drive.google.com/drive/folders/1-6ZdaaxU1eaBacKdSwaQBhllCaea7lEC?usp=share_link).

## Results
Here we show our replicated results of DPR on the NQ dataset:
|          | Top-20 | Top-100 |
|:--------:|:------:|:-------:|
| [Reported](https://arxiv.org/pdf/2004.04906.pdf) |  78.4  |   85.4  |
|   Ours   |  79.1  |   85.9  |

We also report the training and evaluation cost (all experiments are conducted on 8*V100 32G):

|          | training | generate embedding |   build index  | search index |
|:--------:|:--------:|:------------------:|:--------------:|:------------:|
| Duration |  4h 29m  |      4h 42m        |       20m      |      58s     |

## My own results
Here I show my replicated results of DPR on the NQ dataset:
|          | Top-20 | Top-100 |
|:--------:|:------:|:-------:|
| [Reported](https://arxiv.org/pdf/2004.04906.pdf) |  78.4  |   85.4  |
|  Authors |  79.1  |   85.9  |
|   Mine   |  79.2  |   86.5  |

I also report the training and evaluation cost (all experiments are conducted on 4*RTX 3090 24G, 2\*64 Platinum 8358P CPU @ 2.60GHz):
(Using a single machine with 4 cards, FP16 precision, and not adopting DeepSeed, FSDP, or Megatron-LM)

|          | training | generate embedding |   build index  | search index |
|:--------:|:--------:|:------------------:|:--------------:|:------------:|
| Duration |  6h 52m  |      7h 55m        |       6m       |    1h 43m    |

I don't know why my search took so much time. I saw in the issue section of the original author's GitHub repository that someone spent 55 minutes, and they attributed it to the fact that time would be wasted when faiss uses 2 cpu because of the hypter-threading.
related faiss issuse : [facebookresearch/faiss#2953 (comment)](https://github.com/facebookresearch/faiss/issues/2953#issue-1804547006)