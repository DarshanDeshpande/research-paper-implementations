import tensorflow as tf
from .layers import Patches, CAPE, TransformerEncoder, TransformerDecoderLayer
from .config import hyperparameters

IMG_HEIGHT = hyperparameters["IMG_HEIGHT"]
IMG_WIDTH = hyperparameters["IMG_WIDTH"]
PROJECTION_DIM = hyperparameters["PROJECTION_DIM"]
PATCH_SIZE = hyperparameters["PATCH_SIZE"]
EMB_DIM = hyperparameters["EMB_DIM"]
NUM_DECODER_LAYERS = hyperparameters["NUM_DECODER_LAYERS"]
NUM_HEADS = hyperparameters["NUM_HEADS"]
DECODER_OUTPUT_DIM = hyperparameters["DECODER_OUTPUT_DIM"]


def create_model():
    content_image = tf.keras.layers.Input([IMG_HEIGHT, IMG_WIDTH, 3])
    style_image = tf.keras.layers.Input([IMG_HEIGHT, IMG_WIDTH, 3])

    content_patches = Patches(PATCH_SIZE, PROJECTION_DIM)(content_image)
    content_cape = CAPE(PATCH_SIZE, PROJECTION_DIM)(content_image)
    content_encoded = content_cape + content_patches
    content_encoder = TransformerEncoder(EMB_DIM, NUM_HEADS, PROJECTION_DIM)(
        content_encoded
    )

    style_patches = Patches(PATCH_SIZE, PROJECTION_DIM)(style_image)
    style_encoder = TransformerEncoder(EMB_DIM, NUM_HEADS, PROJECTION_DIM)(
        style_patches
    )

    for _ in range(NUM_DECODER_LAYERS):
        content_encoder = TransformerDecoderLayer(EMB_DIM, NUM_HEADS, PROJECTION_DIM)(
            [style_encoder, content_encoder]
        )

    reshaped_output = tf.keras.layers.Reshape(
        [IMG_HEIGHT // PATCH_SIZE, IMG_WIDTH // PATCH_SIZE, PROJECTION_DIM]
    )(content_encoder)

    x = reshaped_output
    for filter_size in [128, 64, 3]:
        x = tf.keras.layers.Conv2D(
            filter_size, (3, 3), padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.UpSampling2D()(x)

    model = tf.keras.Model(inputs=[content_image, style_image], outputs=x)
    return model
