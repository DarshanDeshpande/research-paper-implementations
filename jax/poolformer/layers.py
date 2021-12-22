import jax
from jax import random
import jax.numpy as jnp

import flax.linen as nn


class MLP(nn.Module):
    mlp_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(self.mlp_dim, kernel_init=nn.initializers.xavier_uniform())(inputs)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        return x


class TransformerEncoder(nn.Module):
    mlp_dim: int
    pool_size: int
    stride: int

    @nn.compact
    def __call__(self, inputs):
        norm = nn.LayerNorm()(inputs)
        att = nn.avg_pool(
            norm,
            (self.pool_size, self.pool_size),
            strides=(self.stride, self.stride),
            padding="SAME",
        )
        att = att - norm
        add = inputs + att
        x = nn.LayerNorm()(add)
        x = MLP(self.mlp_dim, self.mlp_dim)(x)
        return add + x


class AddPositionEmbs(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2], inputs.shape[3])
        pe = self.param(
            "pos_embedding", nn.initializers.normal(stddev=0.02), pos_emb_shape
        )
        return inputs + pe
