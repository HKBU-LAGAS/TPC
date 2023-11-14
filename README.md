# Three Phase Clustering (TPC) for Attributed Bipartite Graph Clustering

## Requirements
- Linux
- Python 3.6 or above
- numpy==1.23.3
- scipy==1.9.2
- sklearn==1.1.2
- munkres==1.1.4
  
## Datasets

| data name  | Raw  | Preprocessed  |
|---|---|---|
| Cora  |   |   |
| CiteSeer  |   |   |
| MovieLens  |   |   |
| LastFM  |   |   |
| Google  |   |   |
| Amazon  |   |   |

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
