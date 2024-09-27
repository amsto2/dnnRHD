from tensorflow import keras
from tensorflow.keras.layers import Multiply, Input, Conv1D, DepthwiseConv1D, SeparableConv1D, LeakyReLU, BatchNormalization, ReLU,  \
     Activation, Add, Dropout, MaxPool1D, GlobalAvgPool1D, AvgPool1D,Dense, Concatenate,Reshape,Flatten
from tensorflow.keras import Model
from keras.callbacks import History, ModelCheckpoint
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model

# InceptionTime model adapted from https://github.com/hfawaz/InceptionTime
def _inception_module( input_tensor, stride=1, activation='linear', use_bottleneck=True,bottleneck_size=32, kernel_size=11):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    kernel_size_s = [3, 5, 9, 11]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []
    nb_filters = 32
    for i in range(len(kernel_size_s)):
        conv_list.append(Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                             strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = Conv1D(filters=nb_filters, kernel_size=1,
                                 padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = Concatenate(axis=2)(conv_list)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    return x

def _shortcut_layer( input_tensor, out_tensor):
    shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                     padding='same', use_bias=False)(input_tensor)
    shortcut_y = BatchNormalization()(shortcut_y)

    x = Add()([shortcut_y, out_tensor])
    x = Activation('relu')(x)
    return x

def build_model( input_shape, num_classes, depth, use_bottleneck,bottleneck_size, kernel_size=11):
    use_residual = True
    input_layer = Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x, use_bottleneck= True,bottleneck_size=32)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = GlobalAvgPool1D()(x)

    output_layer = Dense(num_classes, activation='softmax')(gap_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def model_inceptionTime(input_shape, num_classes,ks,nb_epochs):
    output_directory = './'
    nb_filters = 16
    use_residual = True
    use_bottleneck = True
    depth = 4
    kernel_size = ks - 1
    callbacks = None
    batch_size = 16
    bottleneck_size = 32
    build=True
    verbose = False
    num_classes=num_classes
    model = build_model(input_shape, num_classes,depth, use_bottleneck,bottleneck_size)
    return model
