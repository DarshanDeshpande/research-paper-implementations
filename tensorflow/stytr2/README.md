## Introduction
This directory contains the model and default configuration for <a href="https://arxiv.org/abs/2105.14576v2">StyTr^2: Unbiased Image Style Transfer with Transformers</a> (Yingying Deng et al., 2021).

## Architecture Details
The architecture proposes a fully transformer based model for style transfer. The major additions in the paper are:

1. Content Aware Positional Encoding: It is a robust positional encoding technique that considers scale variations of images
2. Modified Transformer Decoder Layer: The original transformer decoder is modified to accomodate the style and content images
3. Self-supervised training: The network is, additionally, trained in a self-supervised manner which gives it an added advantage over the networks trained using only perceptual style and content losses

<img src="https://i.imgur.com/bXlpkZR.png">

## Requirements
All requirements can be installed by running
```sh
pip install -r requirements.txt
```
## Usage
To use the model, you must load your dataset (not included in this repository), create the model and then call the `Trainer` instance as follows:

```py
from model import create_model
from trainer import Trainer

model = create_model()
trainer = Trainer(model)
trainer.compile(optimizer)
trainer.fit(tf_dataset, epochs=1)
...
trainer.save_model('/path/to/save/dir')
```

Any changes to the model hyperparameters or loss function weights can be made by editing the `model/config` file.