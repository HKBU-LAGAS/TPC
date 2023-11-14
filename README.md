# Three Phase Clustering (TPC) for Attributed Bipartite Graph Clustering

## Requirements
- Linux
- Python 3.6 or above
- numpy==1.23.3
- scipy==1.9.2
- sklearn==1.1.2
- munkres==1.1.4
  
## Datasets

| data name  | URL  |
|---|---|
| Cora  | [preprocessed](https://github.com/AnryYang/TPC/tree/main/datasets) [raw](https://github.com/chaoyanghe/bipartite-graph-learning)  |
| CiteSeer  | [preprocessed](https://github.com/AnryYang/TPC/tree/main/datasets) [raw](https://github.com/chaoyanghe/bipartite-graph-learning)  |
| MovieLens  | [preprocessed](https://drive.google.com/file/d/1pSxO5QKV3uCFNjQuS5XepeCJyQ6kDn6t/view?usp=sharing) [raw](https://grouplens.org/datasets/movielens/) |
| LastFM  | [preprocessed](https://drive.google.com/file/d/1pSxO5QKV3uCFNjQuS5XepeCJyQ6kDn6t/view?usp=sharing) [raw](https://snap.stanford.edu/data/feather-lastfm-social.html) | 
| Google  | [preprocessed](https://drive.google.com/file/d/1pSxO5QKV3uCFNjQuS5XepeCJyQ6kDn6t/view?usp=sharing) [raw](https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_restaurants) | 
| Amazon  | [preprocessed](https://drive.google.com/file/d/1pSxO5QKV3uCFNjQuS5XepeCJyQ6kDn6t/view?usp=sharing) [raw](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) | 

## Usage
```shell
$ cd src/
$ sh exe.sh Cora
```

## Parameter Analysis
```shell
$ cd src/
$ sh vary_alpha.sh Cora
$ sh vary_gamma.sh Cora
$ sh vary_tf.sh Cora
$ sh vary_tg.sh Cora
$ sh vary_d.sh Cora
```
