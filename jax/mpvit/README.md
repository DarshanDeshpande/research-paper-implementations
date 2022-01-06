## Introduction
This directory contains the set of models proposed in <a href="https://arxiv.org/abs/2112.11010">MPViT: Multi-Path Vision Transformer for Dense Prediction</a> (Youngwan Lee et al., 2021).

This implementation contains JAX code for:

1. <a href="https://arxiv.org/abs/1905.02244v5">Hardswish</a> (Andrew Howard et al., 2019)
2. <a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf">Depthwise Convolution</a> (François Chollet, 2017)
3. <a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf">Depthwise Separable Convolution</a> (François Chollet, 2017)
4. Factorized Attention as proposed in <a href="https://arxiv.org/abs/2104.06399">Co-Scale Conv-Attentional Image Transformers</a> (Weijian Xu et al., 2021)
5. <a href="https://arxiv.org/abs/2112.11010">MPViT: Multi-Path Vision Transformer for Dense Prediction</a> (Youngwan Lee et al., 2021).

## Architecture Details
This paper proposes a multi-path architecture called MPViT. Unlike the fixed non-overlapping patches proposed in ViT, the authors propose a multi-stage structure where patches of different sizes are created using convolutional embeddings. Since the patch extraction involves variable scales, effective local and global features can be extracted from the image. These features are then aggregated to achieve consistency. The combination of the Multi-Patch Embedding and Multi-Scale Transformer blocks is stacked, while reducing the height and width dimensions at every stage by multiples of 4. 

<!-- <img src="https://i.imgur.com/0Sc798S.png" height=auto width=450px> -->

<img src="https://i.imgur.com/us0HYU1.png">

All four implementations:

1. Tiny
2. XSmall
3. Small
4. Base

are included in this repository. Although, the official code for the paper is yet to be released as of the creation of this directory, every model aims to be as true to the paper as possible. The accuracy is ensured by double checking the paper directions and matching the trainable parameters for each model.

## Requirements
All requirements can be installed by running
```sh
pip install -r requirements.txt
```
## Usage
To use the model, you must load your dataset (not included in this repository), create and call your desired model as follows:

```py
from models import Base # You can import Tiny/XSmall/Small/Base

drop, key = random.split(random.PRNGKey(0), 2)
model = Base(attach_head=True, num_classes=1000)
x = jnp.zeros([1, 224, 224, 3])
variables = model.init({"params": key, "dropout": drop}, x, True)
params, batch_stats = variables["params"], variables["batch_stats"]
out, batch_stats = model.apply(
    {"params": params, "batch_stats": batch_stats},
    x,
    True,
    mutable=["batch_stats"],
    rngs={"dropout": drop},
)
```