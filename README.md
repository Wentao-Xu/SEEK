# [SEEK: Segmented Embedding of Knowledge Graphs](https://arxiv.org/abs/2005.00856)
Source code fot the ACL 2020 paper "SEEK: Segmented Embedding of Knowledge Graphs".

## Model Training
```
make && ./main -dataset DB100K -num_thread 24 -model_path seek.model
```
## Link Prediction Task
```
./main -dataset DB100K -num_thread 24 -model_path seek.model -prediction 1
```
## Triple Classification Task
```
./main -dataset DB100K -num_thread 24 -model_path seek.model -classification 1
```
## Command Line Option

|Option | Description|
|:-----|:------------|
|-dataset|Dataset|
|-num_thread|Number of threads|
|-embed_dim|Dimension of embeddings|
|-num_seg|Number of segments|
|-neg_sample|Negatives samples|
|-num_epoch|Epochs for training|
|-model_path|Model path|
|-lambda|L2 weight regularization penalty|
|-lr|Init learning rate|

## Citation
Please cite the following paper if you use this code in your work.
```
@InProceedings{wentao2020seek,
      author={Xu, Wentao and Zheng, Shun and He, Liang and Shao, Bin and Yin, Jian and Liu, Tie-Yan},
      title={{SEEK: Segmented Embedding of Knowledge Graphs}},
      booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
      year={2020}
      }
```
