## Introduction
This directory contains the PatchConvNet models proposed in <a href="https://arxiv.org/abs/2112.13692v1">Augmenting Convolutional networks with attention-based aggregation</a> (Hugo Touvron et al., 2021).

## Architecture Details
This paper proposes an attention augmented convolutional architecture. This directory provides a JAX implementation for

1. Drop Path
2. Squeeze and Excitation Layer (<a href="https://arxiv.org/abs/1709.01507">Squeeze-and-Excitation Networks</a> (Jie Hu et al. 2019))
3. PatchConvNet

<img src="https://i.imgur.com/4xHce08.png">

The paper claims to achieve competitive results while having lesser computational parameters. The authors propose three configurations, small, large and big, each with increasing complexity and this repository contains all three implementations.

<img src="https://i.imgur.com/qd4We3d.png">

## Requirements
All requirements can be installed by running
```sh
pip install -r requirements.txt
```
## Usage
To use the model, you must load your dataset (not included in this repository), create your desired model as follows:

```py
from models import S60

rng1, rng2, rng3 = random.split(random.PRNGKey(0), 3)
model = S60(attach_head=True, num_classes=1000)
x = jnp.zeros([1, 224, 224, 3])
params = model.init({"params": rng1, "dropout": rng2, "drop_path": rng3}, x, False)["params"] # Here, inputs=x and deterministic=False
logits = model.apply({"params": params}, x, False, rngs={"dropout": rng2, "drop_path": rng3})
```
Note that all models provided in this repository are as closely implemented as (og github). If you have any issues with the implementation then please raise an issue.