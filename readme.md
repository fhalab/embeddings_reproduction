# Code to reproduce the paper *Learned Protein Embeddings for Machine Learning*.

## Installation

`embeddings_reproduction` can be installed with `pip` from the command line using the following command:

```bash
$ pip install git+https://github.com/fhalab/embeddings_reproduction.git
```

It can also be installed in editable mode (`-e`) from the source with:

```bash
$ git clone https://github.com/fhalab/embeddings_reproduction.git
$ cd embeddings_reproduction
$ pip install -e .
```

The second option might be necessary depending on how your computer handles Git-LFS. Because some of the files are large, the connection might time out.

### Computing Environment

This was originally developed using Anaconda Python 3.5 and the following packages and versions:

```
gensim==1.0.1
numpy==1.13.1
pandas==0.20.3
scipy==0.19.1
sklearn==0.19.0
matplotlib==2.0.2
seaborn==0.8.1
```

## File structure

The repository is divided into code, inputs and outputs. Inputs contains all the unlabeled sequences used to build docvec models, the labeled sequences used to build Gaussian process regression models, and AAIndex, ProFET, and one-hot encodings of the labeled sequences. Code contains Python implementations of Gaussian process regression and the mismatch string kernel in addition to Jupyter notebooks that reproduce the analyses in the paper. Outputs contains all the embeddings produced during the course of analysis and csvs storing the results of the cross-validation over embedding hyperparameters, the negative controls, and the results of varying the embedding dimension or the number of unlabeled sequences. Note that while code to train docvec models is provided, the actual docvec models produced by gensim are not included in the repository because they are too large. These are at freely available at http://cheme.caltech.edu/~kkyang/.
