
# import tensorflow 
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
class my_model:
      def __init__(self,units=None,shap_1=None,shap_2=None,activation='relu'):
            self.units = units
            self.shap_1 = shap_1
            self.shap_2 = shap_2
            self.activation = activation
            
      #################--           create_model           --#################
      def create_model(self):
            return tf.keras.models.Sequential([
                  # Shape [batch, time, features] => [batch, time, lstm_units]
                  keras.layers.LSTM(self.units, input_shape=(self.shape_1, self.shape_2),activation=self.activation),
                  # keras.BatchNormalization(),
                  keras.layers.Dense(units=1)
            ])

# class train_model:
#       def __init__(self):
#             self.name = ""




class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)
model = MyModel()
print(model.summary())