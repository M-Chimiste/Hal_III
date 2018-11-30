import sys
import os
import numpy as np
stderr = sys.stderr
GPU_PERCENT = 0.1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class AI:
    def __init__(self):
        with open(os.devnull, 'w') as sys.stderr:
            from keras.models import load_model
            import keras.backend.tensorflow_backend as backend
            import tensorflow as tf
            self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_PERCENT)
            backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options)))
            self.model = load_model("BasicCNN-25-epochs-0.001-LR-STAGE1")
            self.model.predict(np.random.randn(1, 33, 33, 3))
            sys.stderr = stderr



    def prediction(self,input):
        choice = self.model.predict(input)
        final_choice = np.argmax(choice)
        return final_choice