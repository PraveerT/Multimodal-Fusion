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


def build_model_3DCNN_F(input_shape,dropout):
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    x = keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    return keras.Model(inputs, x)
  
  
def transformer_encoder_OA(inputs, head_size, num_heads, ff_dim, dropout=0):

    x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = keras.layers.Dropout(dropout)(x)
    return x + inputs

def build_model_3DCNN_with_F_ATT(input_shape,dropout):
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    x = keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Reshape((-1,128))(x)
    for _ in range(4):
        x = transformer_encoder_OA(x, 2048,16,16, 0.1)


    return keras.Model(inputs, x)

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

def MergeAtt(Model_A,Model_B,Model_C,lr_schedule,METRICS):
    merged = keras.layers.Concatenate(name="MERGE_ATT")([Model_A.output,Model_B.output,Model_C.output])
    merged=keras.layers.Reshape((-1,merged.shape[1]))(merged)
    x=merged
    for _ in range(4):
        x = transformer_encoder_OA(x, 2048,16,16, 0.1)
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
