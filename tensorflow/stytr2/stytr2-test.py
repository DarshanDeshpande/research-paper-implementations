import tensorflow as tf
import unittest
import shutil

from model.model import create_model
from trainer.trainer import Trainer


class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = create_model()

    def test_output_shape(self):
        style = tf.random.normal([1, 256, 256, 3])
        content = tf.random.normal([1, 256, 256, 3])
        output = self.model([style, content])
        self.assertEqual(output.shape, [1, 256, 256, 3])

    def test_trainer(self):
        # Create sample dataset
        style = tf.random.normal([4, 256, 256, 3])
        content = tf.random.normal([4, 256, 256, 3])
        dataset = tf.data.Dataset.from_tensor_slices((style, content))
        dataset = dataset.batch(2)

        trainer = Trainer(self.model)
        trainer.compile(tf.keras.optimizers.Adam(0.0005))
        trainer.fit(dataset, epochs=1)

    def test_model_save(self):
        self.model.save("temp/test_save", save_format="tf")
        shutil.rmtree("temp/")


if __name__ == "__main__":
    unittest.main()
