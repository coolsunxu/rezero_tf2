
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Rezero_Dense(keras.Model):
	def __init__(self,outputs):
		super(Rezero_Dense,self).__init__()
		self.outputs = outputs
		self.resweight = tf.Variable(0.0,trainable=True)
		self.fc = layers.Dense(self.outputs,activation='relu',
								kernel_initializer='glorot_normal')
									
	def call(self,x):
		x = x + self.resweight * self.fc(x)
		return x

"""
a = tf.random.normal([32, 512])
model = Rezero_Dense(512)
ou = model(a)
print(ou.shape)
"""
