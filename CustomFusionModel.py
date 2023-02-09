#Version control 1

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os,shutil
from tensorflow import keras
from IPython.display import clear_output #clear_output(wait=True)
# plt.style.use('dark_background')
import pickle
import tensorflow as tf
from numpy.random import seed
from keras.layers import MultiHeadAttention


from keras.layers import Input, Dense, Flatten, LayerNormalization, Dropout
from keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling1D
from keras.models import Model
from keras_transformer import MultiHeadSelfAttention, PositionEmbedding

from keras.layers import Layer, Dropout, LayerNormalization

class Transformer(Layer):
    def __init__(self, num_heads, units, activation, dropout, **kwargs):
        self.num_heads = num_heads
        self.units = units
        self.activation = keras.activations.get(activation)
        self.dropout = Dropout(dropout)
        super(Transformer, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[-1]
        self.q_dense = keras.layers.Dense(self.units, activation=self.activation)
        self.k_dense = keras.layers.Dense(self.units, activation=self.activation)
        self.v_dense = keras.layers.Dense(self.units, activation=self.activation)
        self.out_dense = keras.layers.Dense(dim, activation=self.activation)
        super(Transformer, self).build(input_shape)

    def get_config(self):
        config = {
            'num_heads': self.num_heads,
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'dropout': self.dropout
        }
        base_config = super(Transformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        q = self.q_dense(inputs)
        k = self.k_dense(inputs)
        v = self.v_dense(inputs)
        dim = self.units
        heads = self.num_heads
        batch_size, seq_len, dim = inputs.shape
        depth = dim // heads
        q *= depth ** -0.5
        x = keras.backend.batch_dot(q, k, axes=[2, 2])
        x = keras.backend.softmax(x)
        x = self.dropout(x)
        x = keras.backend.batch_dot(x, v)
        x = keras.backend.permute_dimensions(x, (0, 2, 1, 3))
        x = keras.backend.reshape(x, (batch_size, seq_len, dim))
        x = self.out_dense(x)
        x = LayerNormalization()(x)
        return x


def CustomMultiHeadAttention(n_heads, head_size, name=None):
    def _multi_head_attention(inputs):
        # Linear projections for queries, keys, and values
        queries = tf.keras.layers.Dense(n_heads * head_size, activation='relu')(inputs)
        keys = tf.keras.layers.Dense(n_heads * head_size, activation='relu')(inputs)
        values = tf.keras.layers.Dense(n_heads * head_size, activation='relu')(inputs)
        
        # Reshape the queries, keys, and values
        queries = tf.reshape(queries, (-1, tf.shape(queries)[1], n_heads, head_size))
        keys = tf.reshape(keys, (-1, tf.shape(keys)[1], n_heads, head_size))
        values = tf.reshape(values, (-1, tf.shape(values)[1], n_heads, head_size))
        
        # Transpose the queries, keys, and values for calculation of dot product
        queries = tf.transpose(queries, [0, 2, 1, 3])
        keys = tf.transpose(keys, [0, 2, 3, 1])
        values = tf.transpose(values, [0, 2, 1, 3])
        
        # Calculate dot product between queries and keys
        dot_product = tf.matmul(queries, keys)
        
        # Scale the dot product
        dot_product = dot_product / (head_size ** 0.5)
        
        # Apply softmax to the dot product to obtain weights
        weights = tf.nn.softmax(dot_product)
        
        # Calculate the dot product between the values and the weights to obtain the final values
        attended_values = tf.matmul(weights, values)
        
        # Transpose the attended values to the original shape
        attended_values = tf.transpose(attended_values, [0, 2, 1, 3])
        
        # Reshape the attended values to have a shape of (batch_size, sequence_length, n_heads * head_size)
        attended_values = tf.reshape(attended_values, (-1, tf.shape(attended_values)[1], n_heads * head_size))
        
        # Apply the final linear layer to obtain the output of the multi-head attention mechanism
        attention_output = tf.keras.layers.Dense(n_heads * head_size, activation='relu')(attended_values)
        
        return attention_output
    
    return tf.keras.layers.Lambda(
        _multi_head_attention, 
        name=name if name is not None else f"multi_head_attention_{n_heads}_{head_size}"
    )

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = keras.layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def A(input_shape,dropout):
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    return keras.Model(inputs, x)
  
    

def B(input_shape,dropout):
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    x = keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    return keras.Model(inputs, x)
  
  

def B_Attention(input_shape,dropout):
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Transformer(num_heads=8, units=64, activation='relu')(x)
    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    return keras.Model(inputs, x)

def C(input_shape,dropout):
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    x = keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(12, activation="softmax")(x)

    return keras.Model(inputs, x)




def C_Attention(input_shape,dropout):
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = MultiHeadSelfAttention(head_num=8, name='Self-Attention')(x)
    # Layer normalization
    x = LayerNormalization(epsilon=1e-6)(x)
    # Dropout
    x = Dropout(0.5)(x)
    # Global Average Pooling1D
    x = GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(12, activation="softmax")(x)

    return keras.Model(inputs, x)


def EarlyMerge(Model_A,Model_B,Model_C,lr_schedule,METRICS):
    merged = keras.layers.Add(name="MERGE")([Model_A.output,Model_B.output,Model_C.output])
    x = keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(merged)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(12, activation="softmax")(x)
    
    Mergemodel = keras.Model(inputs=[Model_A.input,Model_B.input, Model_C.input], outputs=x)
# Mergemodel.summary()

    Mergemodel.compile(loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=METRICS)
    
    return Mergemodel

def EarlyMergeIntermediateAttention(Model_A,Model_B,Model_C,lr_schedule,METRICS):
    merged = keras.layers.Add(name="MERGE")([Model_A.output,Model_B.output,Model_C.output])
    x = keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(merged)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Reshape((-1,x.shape[-1]))(x)
    for _ in range(4):
        x = transformer_encoder(x,2048,16,16, 0.1)
    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(12, activation="softmax")(x)
    Mergemodel = keras.Model(inputs=[Model_A.input,Model_B.input, Model_C.input], outputs=x)
# Mergemodel.summary()

    Mergemodel.compile(loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=METRICS)
    
    return Mergemodel



def Merge(Model_A,Model_B,Model_C,lr_schedule,METRICS):
    merged = keras.layers.Concatenate(name="MERGE")([Model_A.output,Model_B.output,Model_C.output])
    output = keras.layers.Flatten()(merged)
    output = keras.layers.Dense(128, activation="relu",name="FD")(output)
    output = keras.layers.Dropout(0.4)(output)
    output = keras.layers.Dense(1024, activation="relu")(output)
    output = keras.layers.Dropout(0.4)(output)
    output = keras.layers.Dense(12, activation="softmax")(output)
    
    Mergemodel = keras.Model(inputs=[Model_A.input,Model_B.input, Model_C.input], outputs=output)
# Mergemodel.summary()

    Mergemodel.compile(loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=METRICS)
    
    return Mergemodel


def Merge_Attention(Model_A,Model_B,Model_C,lr_schedule,METRICS):
    merged = keras.layers.Concatenate(name="MERGE_ATT")([Model_A.output,Model_B.output,Model_C.output])
    merged=keras.layers.Reshape((-1,merged.shape[-1]))(merged)
    x=merged
    for _ in range(4):
        x = transformer_encoder(x, 2048,16,16, 0.1)
    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    output = keras.layers.Flatten()(x)
    output = keras.layers.Dense(128, activation="relu",name="FD")(output)
    output = keras.layers.Dropout(0.4)(output)
    output = keras.layers.Dense(1024, activation="relu",name="ML3")(output)
    output = keras.layers.Dropout(0.4)(output)
    output = keras.layers.Dense(12, activation="softmax",name="ML4")(output)




    MergeAttmodel = keras.Model(inputs=[Model_A.input,Model_B.input, Model_C.input], outputs=output)
    # Mergemodel.summary()
    MergeAttmodel.compile(loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=METRICS)
    
    return MergeAttmodel

def Merge_late_Attention_II(Model_A,Model_B,Model_C,lr_schedule,METRICS):
    merged = keras.layers.Concatenate(name="MERGE_ATT")([Model_A.output,Model_B.output,Model_C.output])
    print (merged.shape)
    layer = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(0,1,2))
    output_tensor = layer(merged, merged)
    output = keras.layers.Flatten()(output_tensor)
    output = keras.layers.Dense(128, activation="relu",name="FD")(output)
    output = keras.layers.Dropout(0.4)(output)
    output = keras.layers.Dense(1024, activation="relu",name="ML3")(output)
    output = keras.layers.Dropout(0.4)(output)
    output = keras.layers.Dense(12, activation="softmax",name="ML4")(output)




    MergeAttmodel = keras.Model(inputs=[Model_A.input,Model_B.input, Model_C.input], outputs=output)
    # Mergemodel.summary()
    MergeAttmodel.compile(loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=METRICS)
    
    return MergeAttmodel
