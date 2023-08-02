import tensorflow as tf

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"
        
        self.values = tf.keras.layers.Dense(self.head_dim, use_bias=False)
        self.keys = tf.keras.layers.Dense(self.head_dim, use_bias=False)
        self.queries = tf.keras.layers.Dense(self.head_dim, use_bias=False)
        self.fc_out = tf.keras.layers.Dense(embed_size)
        
    def call(self, values, keys, query, mask):
        # Split the embedding into self.num_heads different pieces
        values = tf.reshape(self.values(values), shape(batch_size, -1, self.num_heads, self.head_dim))
        keys = tf.reshape(self.keys(keys), shape(batch_size, -1, self.num_heads, self.head_dim))
        queries = tf.reshape(self.queries(query), shape(batch_size, -1, self.num_heads, self.head_dim))
        
        values = values.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        queries = queries.transpose(0, 2, 1, 3)
        
        # Calculate dot product attention
        dot_product = tf.matmul(queries, keys, transpose_b=True)
        
        if mask is not None:
            dot_product = dot_product + (mask * -1e9)
        
        attention_weights = tf.nn.softmax(dot_product, axis=-1)
        
        output = tf.matmul(attention_weights, values)
        
        output = tf.reshape(output, shape=(batch_size, -1, self.embed_size))
        output = self.fc_out(output)
        
        return output
