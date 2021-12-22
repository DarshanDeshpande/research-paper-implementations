## Introduction
This directory contains the PoolFormer models proposed in <a href="https://arxiv.org/abs/2111.11418">MetaFormer is Actually What You Need for Vision</a> (Weihao Yu et al., 2021).

## Architecture Details
This paper proposes a general architecture called Metaformer where the original Multi-Headed Attention from the original Transformer paper is substituted with a placeholder mixer module. To prove this, the authors propose Poolformer, an architecture where the attention mechanism is substituted with Average Pooling layers, which achieves competitve results on well recognized benchmarks.

<img src="https://user-images.githubusercontent.com/15921929/144710761-1635f59a-abde-4946-984c-a2c3f22a19d2.png">

## Requirements
All requirements can be installed by running
```sh
pip install -r requirements.txt
```
## Usage
To use the model, you must load your dataset (not included in this repository), create your desired model as follows:

```py
from models import PoolFormer_S12
rng = random.PRNGKey(0)
s12 = PoolFormer_S12()
x = jnp.ones([1, 256, 256, 3])
params = s12.init({"params": rng}, x)["params"]
sample_out = s12.apply({"params": params}, x)
```
Note that all models provided in this repository are headless. An external head (MLP classifier) must be applied on top of this model.