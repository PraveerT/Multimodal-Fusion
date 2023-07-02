# For single_modality_transformer_model
def single_modality_transformer_model(input_shape, num_classes, transformer_block, name, sparse_attention=False):
    inputs = Input(input_shape)
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'),name=f'1-{name}')(inputs)
    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu'),name=f'2-{name}')(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)),name=f'3-{name}')(x)
    x = layers.TimeDistributed(layers.Flatten(),name=f'4-{name}')(x)
    transformer = transformer_block()  # Initialize the transformer
    if sparse_attention:
        x = transformer(x, x, x)  # Pass the input to the transformer
    else:
        x = transformer(x)
    x = layers.Reshape((-1, sequence_length, hidden_units // sequence_length),name=f'5-{name}')(x)  # Adding reshape to adjust tensor dimensions
    x = layers.TimeDistributed(layers.GlobalAveragePooling1D(),name=f'6-{name}')(x)
    return Model(inputs, x)

# For fusion_transformer_model_sparse
def fusion_transformer_model(models, num_classes, transformer_block, sparse_attention=False):
    inputs = [model.input for model in models]
    outputs = [model.output for model in models]
    x = layers.Concatenate(axis=-1)(outputs)  # Concatenating the features of each modality
    transformer = transformer_block()  # Initialize the transformer
    if sparse_attention:
        x = transformer(x, x, x)  # Pass the input to the transformer
    else:
        x = transformer(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

