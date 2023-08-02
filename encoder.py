class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embed_size, num_heads, ff_dim, input_vocab_size, maximum
