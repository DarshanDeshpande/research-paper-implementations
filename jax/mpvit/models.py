import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Iterable

from layers import (
    ConvolutionalStem,
    MultiScalePatchEmbedding,
    MultiPathTransformerBlock,
)


class Model(nn.Module):
    mlp_ratio: int = 2
    channels_list: Iterable = (64, 96, 176, 216)
    num_layers_list: Iterable = (1, 2, 4, 1)
    attach_head: bool = True
    num_classes: int = 1000
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        stem = ConvolutionalStem(64)(inputs, deterministic=deterministic)
        mptb = stem
        for i, j in zip(self.channels_list, self.num_layers_list):
            mspe = MultiScalePatchEmbedding(features=i, strides=2)(mptb, deterministic)
            mptb = MultiPathTransformerBlock(
                features=i,
                dim=i,
                num_heads=8,
                num_encoder_layers=j,
                mlp_ratio=self.mlp_ratio,
            )(mspe, deterministic)

        if self.attach_head:
            # Global avg pooling
            x = jnp.mean(mptb, 1)
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x)
            return x

        return mptb


def Tiny(attach_head=True, num_classes=1000):
    return Model(
        mlp_ratio=2,
        channels_list=(64, 96, 176, 216),
        num_layers_list=(1, 2, 4, 1),
        attach_head=attach_head,
        num_classes=num_classes,
    )


def XSmall(attach_head=True, num_classes=1000):
    return Model(
        mlp_ratio=4,
        channels_list=(64, 128, 192, 256),
        num_layers_list=(1, 2, 4, 1),
        attach_head=attach_head,
        num_classes=num_classes,
    )


def Small(attach_head=True, num_classes=1000):
    return Model(
        mlp_ratio=4,
        channels_list=(64, 128, 216, 288),
        num_layers_list=(1, 3, 6, 3),
        attach_head=attach_head,
        num_classes=num_classes,
    )


def Base(attach_head=True, num_classes=1000):
    return Model(
        mlp_ratio=4,
        channels_list=(128, 224, 368, 480),
        num_layers_list=(1, 3, 8, 3),
        attach_head=attach_head,
        num_classes=num_classes,
    )
