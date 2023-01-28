
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
   
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
​
    # Feed Forward Part
    x = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(res)
    x = keras.layers.BatchNormalization()(x)
    
    x = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
   
    x = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
​
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    
    return x + res
​
def transformer_encoder_O(inputs, head_size, num_heads, ff_dim, dropout=0):
​
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    return x + inputs
​
​
def build_model_1DCNNT(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
​
    outputs = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    return keras.Model(inputs, outputs)
   
def build_model_3DCNNT(input_shape,dropout):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = keras.layers.Conv3D(8, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.Conv3D(8, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv3D(16, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = keras.layers.Reshape((-1,1,128))(x)
    for _ in range(4):
        x = transformer_encoder_O(x, 256, 4, 4, 0.4)
    outputs =layers.Reshape((128,))(x)
    return keras.Model(inputs, outputs)
