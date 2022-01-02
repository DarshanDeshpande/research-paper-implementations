import unittest
from jax import random
import jax.numpy as jnp
import numpy as np

from layers import (
    ConvolutionalStem,
    TrunkBlock,
    DropPath,
    AttentionPoolingBlock,
    PatchConvNet,
)


class LayerTests(unittest.TestCase):
    def test_conv_stem(self):
        sample_image = jnp.zeros([1, 224, 224, 3])
        convstem = ConvolutionalStem(emb_dim=768)
        params = convstem.init({"params": random.PRNGKey(0)}, sample_image)["params"]
        output_shape = convstem.apply({"params": params}, sample_image).shape
        self.assertEqual(output_shape, (1, 196, 768))

    def test_trunk(self):
        arr = jnp.zeros([1, 196, 768])
        trunk = TrunkBlock(768)
        params = trunk.init({"params": random.PRNGKey(0)}, arr)["params"]
        output_shape = trunk.apply({"params": params}, arr).shape
        self.assertEqual(output_shape, (1, 196, 768))

    def test_drop_path(self):
        droppath = DropPath(0.3)
        params_key, drop_path_key, key = random.split(random.PRNGKey(0), 3)

        arr = random.uniform(key, [3, 2, 2, 2])
        params = droppath.init(
            {"params": params_key, "drop_path": drop_path_key}, arr, False
        )
        output = droppath.apply(
            {"params": params}, arr, False, rngs={"drop_path": drop_path_key}
        )

        test_arr = [
            [
                [[0.07269315, 0.9130809], [1.2821915, 0.98061377]],
                [[0.45805728, 0.3266367], [1.4167114, 0.10965841]],
            ],
            [
                [[0.6914027, 1.1517124], [1.3579687, 0.5189223]],
                [[0.7914342, 0.3071068], [0.3616304, 0.057035]],
            ],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
        ]

        assert np.testing.assert_allclose(output, test_arr) is None

    def test_attention_pooling_block(self):
        params, key = random.split(random.PRNGKey(0), 2)
        attblock = AttentionPoolingBlock(768, 4)
        arr = jnp.zeros([1, 196, 768])
        cls_token = jnp.zeros((1, 1, 768))
        params = attblock.init(
            {"params": params, "dropout": key}, [arr, cls_token], True
        )["params"]
        output_shape = attblock.apply(
            {"params": params}, [arr, cls_token], True, rngs={"dropout": key}
        ).shape

        self.assertEqual(output_shape, (1, 1, 768))


class PatchConvNetTest(unittest.TestCase):
    def test_output_shape(self):
        rng1, rng2, rng3 = random.split(random.PRNGKey(0), 3)
        num_classes = 100
        model = PatchConvNet(
            attach_head=True, out_classes=num_classes, depth=1, dim=384, dropout=0.3
        )
        x = jnp.zeros([1, 224, 224, 3])
        params = model.init(
            {"params": rng1, "dropout": rng2, "drop_path": rng3}, x, False
        )["params"]
        logits_shape = model.apply(
            {"params": params}, x, False, rngs={"dropout": rng2, "drop_path": rng3}
        ).shape
        self.assertEqual(logits_shape, (1, num_classes))


if __name__ == "__main__":
    unittest.main()
