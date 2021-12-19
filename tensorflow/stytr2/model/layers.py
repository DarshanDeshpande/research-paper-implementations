import tensorflow as tf


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, projection_dim):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.projection = tf.keras.layers.Dense(units=projection_dim)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        linear_projection = self.projection(patches)
        return linear_projection

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CAPE(tf.keras.layers.Layer):
    def __init__(self, patch_size, projection_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection_dim = projection_dim
        self.patch_size = patch_size

    def build(self, input_shape):
        self.pe = tf.keras.layers.Dense(self.projection_dim)
        self.avg_pool = tf.keras.layers.AveragePooling2D(
            (18, 18), strides=(self.patch_size, self.patch_size), padding="same"
        )
        self.interpolation_weight = tf.Variable(
            name="interpolation_weight",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, self.projection_dim)
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        pool = self.avg_pool(inputs)
        reshaped = tf.keras.layers.Reshape([-1, pool.shape[-1]])(pool)
        # reshaped = tf.reshape(pool, [pool.shape[0], -1, pool.shape[-1]])
        encoding = self.pe(reshaped)
        return self.interpolation_weight * encoding

    def get_config(self):
        config = super().get_config()
        config.update(
            {"projection_dim": self.projection_dim, "patch_size": self.patch_size}
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.ff_dim, activation="relu"),
                tf.keras.layers.Dense(self.embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.att_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.ff_dim, activation="relu"),
                tf.keras.layers.Dense(self.embed_dim),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        style, content = inputs
        attention = self.att_1(query=content, value=style, key=style)
        norm = self.norm1(content + attention)
        norm = self.dropout1(norm, training=training)
        attention = self.att_2(query=norm, value=style, key=style)
        norm = self.norm2(attention + norm)
        ffn = self.ffn(norm)
        ffn = self.dropout2(ffn, training=training)
        out = self.norm3(norm + ffn)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
