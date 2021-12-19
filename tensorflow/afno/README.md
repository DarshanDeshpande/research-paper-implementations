## Introduction
This repository contains the code for <a href="https://arxiv.org/abs/2111.13587.pdf">Adaptive Fourier Neural Operators: Efficient Token Mixers For Transformers</a> (John Guibas et al., 2021)

# Architecture
This paper proposes Adaptive <a href="https://arxiv.org/abs/2010.08895">Fourier Neural Operators</a> (Zongyi Li et al., 2021) as an improvement to the FNO and <a href="https://arxiv.org/abs/2107.00645v1">Global Filter Networks for Image Classification</a>, Yongming Rao et al., 2021) mechanisms.


<img src="https://i.imgur.com/hVqai83.png" height=400px width=500px>

The use of adaptive weight sharing, sparsification of frequency with soft thresholding and shrinkage gives AFNO a quasi-linear complexity and makes it a more desirable substitute for previous proposals of self-attention.

# Requirements
All requirements can be installed by running
```sh
pip install -r requirements.txt
```

## Usage
You can incorporate the AFNO self-attention module into your own code by simply calling
```py
from afno import AFNO
...
attention = AFNO(k=4)
attnetion(x)
...
```