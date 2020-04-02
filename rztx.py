
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow_addons as tfa

from MultiHeadAttention import MultiHeadAttention

class RZTXEncoderLayer(keras.Model):
	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
		super(RZTXEncoderLayer,self).__init__()
		# d_model = E  Q:[L,N,E] K:[S,N,E] V:[S,N,E] bs = N
		self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout) # 自注意力模型,等待tensorflow更新多头
		# Implementation of Feedforward model
		self.linear1 = layers.Dense(dim_feedforward) # 线性1
		self.dropout = layers.Dropout(dropout)
		self.linear2 = layers.Dense(d_model) # 线性2
		self.dropout1 = layers.Dropout(dropout)
		self.dropout2 = layers.Dropout(dropout)
		self.resweight = tf.Variable(0.0,trainable=True) # 学习参数alpha

		if activation == "relu":
			self.activation = activations.relu
		elif activation == "gelu":
			self.activation = tfa.activations.gelu

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = activations.relu
		super().__setstate__(state)

	def call(self, src, mask=None):
		# Self attention layer
		src2 = src
		src2,_ = self.self_attn(src2, src2, src2, mask) # [l,bs,emb]
		src2 = src2 * self.resweight
		src = src + self.dropout1(src2) # [l,bs,emb]

		# Pointiwse FF Layer 全连接层
		src2 = src            
		src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
		src2 = src2 * self.resweight
		src = src + self.dropout2(src2)
		return src

class RZTXDecoderLayer(keras.Model):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(RZTXDecoderLayer,self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = layers.Dense(dim_feedforward)
        self.dropout = layers.Dropout(dropout)
        self.linear2 = layers.Dense(d_model)

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)
        self.resweight = tf.Variable(0.0,trainable=True)

        if activation == "relu":
            self.activation = activations.relu
        elif activation == "gelu":
            self.activation = tfa.activations.gelu

    def call(self, tgt, memory, tgt_mask=None, memory_mask=None):
        
        tgt2,_ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2) * self.resweight
        # Q = tgt; K = memory; V = memory
        tgt2,_ = self.multihead_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2) * self.resweight

        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2) * self.resweight
        return tgt

"""
encoder_layer = RZTXEncoderLayer(d_model=512, nhead=8)
src = tf.random.normal([32, 10, 512]) # [bs,q,emb]
out = encoder_layer(src)
print(out.shape)

decoder_layer = RZTXDecoderLayer(d_model=512, nhead=8)
memory = tf.random.normal([32, 10, 512])
tgt = tf.random.normal([32, 20, 512])
out = decoder_layer(tgt, memory)
print(out.shape)
"""
