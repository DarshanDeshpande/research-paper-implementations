import tensorflow as tf
import unittest

from afno import AFNO


class Test(unittest.TestCase):
    def test_shape_and_rank(self):
        sample_tensor = tf.zeros(shape=[1, 32, 32, 254], dtype=tf.float32)
        afno_layer = AFNO(k=4)
        attention = afno_layer(sample_tensor, training=True)
        self.assertEqual(tf.rank(attention), 4)
        self.assertEqual(tf.reduce_all(tf.shape(attention) == [1, 32, 32, 254]), True)


if __name__ == "__main__":
    unittest.main()
