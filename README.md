# SEEK Model
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
