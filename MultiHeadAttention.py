
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
class MultiHeadAttention(keras.Model):
	# https://machinetalk.org/2019/04/29/create-the-transformer-with-tensorflow-2-0/
	def __init__(self, model_size, h, dropout):
		super(MultiHeadAttention, self).__init__()
		self.query_size = model_size // h
		self.key_size = model_size // h
		self.value_size = model_size // h
		self.h = h
		self.wq = [layers.Dense(self.query_size) for _ in range(h)]
		self.wk = [layers.Dense(self.key_size) for _ in range(h)]
		self.wv = [layers.Dense(self.value_size) for _ in range(h)]
		self.wo = layers.Dense(model_size)
		self.dropout = layers.Dropout(dropout)
	def call(self, query, value):
		# query has shape (batch, query_len, model_size)
		# value has shape (batch, value_len, model_size)
		heads = []
		for i in range(self.h):
			score = self.dropout(tf.matmul(self.wq[i](query), self.wk[i](value), transpose_b=True))

			# Here we scale the score as described in the paper
			score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
			# score has shape (batch, query_len, value_len)

			alignment = tf.nn.softmax(score, axis=2)
			# alignment has shape (batch, query_len, value_len)

			head = tf.matmul(alignment, self.wv[i](value))
			# head has shape (batch, decoder_len, value_size)
			heads.append(head)

		# Concatenate all the attention heads
		# so that the last dimension summed up to model_size
		heads = tf.concat(heads, axis=2)
		heads = self.wo(heads)
		# heads has shape (batch, query_len, model_size)
		return heads
"""

class MultiHeadAttention(keras.Model):
	# https://www.tensorflow.org/tutorials/text/transformer
	def __init__(self, d_model, num_heads, dropout):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model

		assert d_model % self.num_heads == 0

		self.depth = d_model // self.num_heads

		self.wq = layers.Dense(d_model)
		self.wk = layers.Dense(d_model)
		self.wv = layers.Dense(d_model)
		self.dropout = layers.Dropout(dropout)

		self.dense = layers.Dense(d_model)
		
	def scaled_dot_product_attention(self, q, k, v, mask):
		"""计算注意力权重。
		q, k, v 必须具有匹配的前置维度。
		k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
		虽然 mask 根据其类型（填充或前瞻）有不同的形状，
		但是 mask 必须能进行广播转换以便求和。

		参数:
		q: 请求的形状 == (..., seq_len_q, depth)
		k: 主键的形状 == (..., seq_len_k, depth)
		v: 数值的形状 == (..., seq_len_v, depth_v)
		mask: Float 张量，其形状能转换成
			  (..., seq_len_q, seq_len_k)。默认为None。

		返回值:
		输出，注意力权重
		"""
		matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
		# 缩放 matmul_qk
		dk = tf.cast(tf.shape(k)[-1], tf.float32)
		scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
		# 将 mask 加入到缩放的张量上。
		if mask is not None:
			scaled_attention_logits += (mask * -1e9)

		# softmax 在最后一个轴（seq_len_k）上归一化，因此分数
		# 相加等于1。
		attention_weights = self.dropout(tf.nn.softmax(scaled_attention_logits, axis=-1))  # (..., seq_len_q, seq_len_k)
		output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

		return output, attention_weights
		
	def split_heads(self, x, batch_size):
		"""分拆最后一个维度到 (num_heads, depth).
		转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
		"""
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3])

	def call(self, q, k, v, mask):
		batch_size = tf.shape(q)[0]

		q = self.wq(q)  # (batch_size, seq_len, d_model)
		k = self.wk(k)  # (batch_size, seq_len, d_model)
		v = self.wv(v)  # (batch_size, seq_len, d_model)

		q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

		# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
		# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
		scaled_attention, attention_weights = self.scaled_dot_product_attention(
			q, k, v, mask)

		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

		concat_attention = tf.reshape(scaled_attention, 
									  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

		output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
			
		return output, attention_weights
