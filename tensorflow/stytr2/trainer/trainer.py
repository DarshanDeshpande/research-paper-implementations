import tensorflow as tf
from utils.utils import load_vgg, mean_squared_error
from model.config import loss_params

vgg19 = load_vgg()


class Trainer(tf.keras.Model):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_tracker = tf.keras.metrics.Mean(name="overall_loss")

    def train_step(self, batch):
        content, style = batch

        with tf.GradientTape() as tape:
            output = self.model([content, style])

            c_loss, s_loss, loss_id1, loss_id2, = (
                [],
                [],
                0,
                [],
            )

            with tape.stop_recording():
                true_features = vgg19(content, training=False)
                pred_features = vgg19(style, training=False)

            for num_feature, (feature_cs, feature_c) in enumerate(
                zip(true_features, pred_features)
            ):
                c_loss.append(mean_squared_error(feature_cs, feature_c))

                mean_cs = tf.math.reduce_mean(feature_cs)
                mean_c = tf.math.reduce_mean(feature_c)
                std_cs = tf.math.reduce_std(feature_c)
                std_c = tf.math.reduce_std(feature_c)

                mean_diff = mean_squared_error(mean_cs, mean_c)
                std_diff = mean_squared_error(std_cs, std_c)

                s_loss.append(mean_diff + std_diff)

            c_loss = tf.reduce_mean(c_loss)
            s_loss = tf.reduce_mean(s_loss)

            c = self.model([content, content], training=False)
            s = self.model([style, style], training=False)
            loss_id1 = mean_squared_error(content, c) + mean_squared_error(style, s)

            with tape.stop_recording():
                icc = vgg19(c, training=False)
                ic = vgg19(content, training=False)
                is_ = vgg19(s, training=False)
                iss = vgg19(style, training=False)

            for feature_cc, feature_c, feature_ss, feature_s in zip(icc, ic, is_, iss):
                loss_id2.append(
                    mean_squared_error(feature_cc, feature_c)
                    + mean_squared_error(feature_ss, feature_s)
                )

            loss_id2 = tf.reduce_mean(loss_id2)

            loss = (
                loss_params["lambda_c"] * c_loss
                + loss_params["lambda_s"] * s_loss
                + loss_params["lambda_id1"] * loss_id1
                + loss_params["lambda_id2"] * loss_id2
            )

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"overall_loss": self.loss_tracker.result()}

    def predict(self, batch):
        content, style = batch
        pred = self.model([content, style], training=False)
        return pred

    def save_model(self, save_dir):
        self.model.save(save_dir, save_format="tf")
        print("Model saved successfully")
