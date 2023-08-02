class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, embed_size):
        super(PositionalEncoding, self).__init__()
        self.embed_size = embed_size
        self.positional_encoding = self.positional_encoding(max_position, self.embed_size)
        
    def get_angles(self, pos, i, embed_size):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_size))
        return pos * angle_rates
    
    def positional_encoding(self, position, embed_size):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(embed_size)[np.newaxis, :],
                                     embed_size)
        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
