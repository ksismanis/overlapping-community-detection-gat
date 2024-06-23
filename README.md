# Overlapping Community Detection with Graph Attention Network

Pytorch implementation of the **Neural Overlapping Community Detection** method from
["Overlapping Community Detection with Graph Neural Networks"](http://www.kdd.in.tum.de/nocd).

Implementing a Graph attention network.

## Usage
The main algorithm and other utilities are implemented in the `nocd` package that can be installed as
```bash
python setup.py install
```
A Jupyter notebook [nocd-gat.ipynb](nocd-gat.ipynb) and a python test file  [nocd-gat.py](nocd-gat.py)
are included

## Requirements
```
numpy=1.16.4
pytorch=1.2.0
scipy=1.3.1
torch-geometric
```

Based on the parer:

@article{
    shchur2019overlapping,
    title={Overlapping Community Detection with Graph Neural Networks},
    author={Oleksandr Shchur and Stephan G\"{u}nnemann},
    journal={Deep Learning on Graphs Workshop, KDD},
    year={2019},
}
```
