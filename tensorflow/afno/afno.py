import tensorflow as tf
import tensorflow_addons as tfa


class AFNO(tf.keras.layers.Layer):
    """
    AFNO with adaptive weight sharing and adaptive masking.
    """

    def __init__(self, k, *args, **kwargs):
        self.k = k
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        d = (input_shape[-1] // 2) + 1
        self.mlp_block = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d / self.k, activation="relu"),
                tf.keras.layers.Dense(d / self.k, activation="linear"),
            ]
        )

    def call(self, input, training=True):
        temp = input
        x = tf.signal.rfft2d(
            tf.cast(input, tf.float32),
        )
        batch, h, w, d = x.shape
        x = tf.reshape(x, [batch, h, w, self.k, d // self.k])
        x = self.mlp_block(x)
        x = tf.reshape(x, [batch, h, w, d])
        x = tfa.activations.softshrink(x)
        x = tf.signal.irfft2d(tf.cast(x, tf.complex64))
        return x + temp

    def get_config(self):
        return {"k": self.k}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
