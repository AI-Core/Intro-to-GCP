import tensorflow as tf
import tensorflow_hub as hub
# model = 
# tf.keras.models.load_model('efficientnet_b0_classification_1/')
# model.build()

expect_img_size = 224
m = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/classification/1")
])
m.build([None, expect_img_size, expect_img_size, 3])  # Batch input shape.

tf.saved_model.save(m, 'custom')
