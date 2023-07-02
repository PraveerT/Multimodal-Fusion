import random

class SparseMultiHeadAttention(Layer):
    def __init__(self, hidden_units, num_heads, head_size, dropout_rate):
        super(SparseMultiHeadAttention, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_size = head_size
        self.depth = hidden_units // num_heads
        assert self.depth * num_heads == hidden_units, f"hidden_units ({hidden_units}) must be divisible by num_heads ({num_heads})"
        self.dropout_rate = dropout_rate
        unique_id = random.randint(1,10000)  # This generates a unique ID for each layer
        self.query_transform = Dense(hidden_units, name=f'query_transform_{unique_id}')
        self.key_transform = Dense(hidden_units, name=f'key_transform_{unique_id}')
        self.value_transform = Dense(hidden_units, name=f'value_transform_{unique_id}')
        self.attention_dropout = Dropout(dropout_rate, name=f'attention_dropout_{unique_id}')
        self.output_transform = Dense(hidden_units, name=f'output_transform_{unique_id}')
        
        # Initialize the attention weights for each head
        self.attention_weights = self.add_weight(
            shape=(self.num_heads, self.head_size),
            initializer='glorot_uniform',
            trainable=True,
            name=f'attention_weights_{unique_id}'
        )

    def call(self, query, keys, values):
        query = self.query_transform(query)
        keys = self.key_transform(keys)
        values = self.value_transform(values)

        # Split the heads
        query = tf.reshape(query, [-1] + list(query.shape[1:-1]) + [self.num_heads, self.depth])
        keys = tf.reshape(keys, [-1] + list(keys.shape[1:-1]) + [self.num_heads, self.depth])
        values = tf.reshape(values, [-1] + list(values.shape[1:-1]) + [self.num_heads, self.depth])

        # Compute attention scores
        scores = tf.matmul(query, keys, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(self.head_size, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # Apply attention
        attended = tf.matmul(attention_weights, values)
        attended = tf.reshape(attended, [-1] + list(attended.shape[1:-2]) + [self.hidden_units])

        # Apply final linear transformation
        return self.output_transform(attended)