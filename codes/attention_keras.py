import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL

# Determine the input data format, whether it is channels_first or channels_last
channel_axis = 1 if K.image_data_format() == "channels_first" else 3



class channel_attention3d(tf.keras.layers.Layer):
    """ 
    channel attention module 
    
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    def __init__(self, ratio=8, **kwargs):
        self.ratio = ratio
        super(channel_attention3d, self).__init__(**kwargs)

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = tf.keras.layers.Dense(channel // self.ratio,
                                                 activation='relu',
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')
        self.shared_layer_two = tf.keras.layers.Dense(channel,
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')
        super(channel_attention3d, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        channel = inputs.get_shape().as_list()[-1]

        avg_pool = tf.keras.layers.GlobalAveragePooling3D()(inputs)    
        avg_pool = tf.keras.layers.Reshape((1, 1, 1, channel))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = tf.keras.layers.GlobalMaxPooling3D()(inputs)
        max_pool = tf.keras.layers.Reshape((1, 1, 1, channel))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        feature = tf.keras.layers.Add()([avg_pool, max_pool])
        feature = tf.keras.layers.Activation('sigmoid')(feature)

        return tf.keras.layers.multiply([inputs, feature])

class spatial_attention3d(tf.keras.layers.Layer):
    """ spatial attention module 
        
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    def __init__(self, kernel_size=7, **kwargs):
        self.kernel_size = kernel_size
        super(spatial_attention3d, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv3d = tf.keras.layers.Conv3D(filters=1, 
                                                kernel_size=self.kernel_size,
                                                strides=1, 
                                                padding='same', 
                                                activation='sigmoid',
                                                kernel_initializer='he_normal',    
                                                use_bias=False)
        super(spatial_attention3d, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(inputs)
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(inputs)
        concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
        feature = self.conv3d(concat)	
        
        return tf.keras.layers.multiply([inputs, feature])

class channel_attention2d(tf.keras.layers.Layer):
    """ 
    channel attention module 
    
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    def __init__(self, ratio=8, **kwargs):
        self.ratio = ratio
        super(channel_attention2d, self).__init__(**kwargs)

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = tf.keras.layers.Dense(channel // self.ratio,
                                                 activation='relu',
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')
        self.shared_layer_two = tf.keras.layers.Dense(channel,
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')
        super(channel_attention2d, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        channel = inputs.get_shape().as_list()[-1]

        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)    
        avg_pool = tf.keras.layers.Reshape((1, 1, channel))(avg_pool)
        avg_pool = tf.keras.layers.Dense(channel,
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')(avg_pool)
        avg_pool = tf.keras.layers.Dense(channel,
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')(avg_pool)

        max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
        max_pool = tf.keras.layers.Reshape((1, 1, channel))(max_pool)
        max_pool = tf.keras.layers.Dense(channel,
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')(max_pool)
        max_pool = tf.keras.layers.Dense(channel,
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')(max_pool)

        feature = tf.keras.layers.Add()([avg_pool, max_pool])
        feature = tf.keras.layers.Activation('sigmoid')(feature)

        return tf.keras.layers.multiply([inputs, feature])


class spatial_attention2d(tf.keras.layers.Layer):
    """ spatial attention module 
        
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    def __init__(self, kernel_size=7, **kwargs):
        self.kernel_size = kernel_size
        
        self.c = tf.keras.layers.Concatenate(axis=-1)
        self.mul = tf.keras.layers.multiply

        super(spatial_attention2d, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv2d = tf.keras.layers.Conv2D(filters=1, 
                                                kernel_size=self.kernel_size,
                                                strides=1, 
                                                padding='same', 
                                                activation='sigmoid',
                                                kernel_initializer='he_normal',    
                                                use_bias=False)
        self.lam1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))
        self.lam2 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))

        super(spatial_attention2d, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        avg_pool = self.lam1(inputs)
        max_pool = self.lam2(inputs)
        concat = self.c([avg_pool, max_pool])
        feature = self.conv2d(concat)	
        multi = self.mul([inputs, feature])
        return self.ad([multi, inputs])

class attention2d(tf.keras.layers.Layer):

    def __init__(self, kernel_size=7, **kwargs):
        self.kernel_size = kernel_size
        self.reduction_ratio = 0.125
        self.channel = 4
        self.ad = KL.Add()
        self.mul = KL.Multiply()
        self.pool1 = KL.GlobalMaxPooling2D()
        self.pool2 = KL.GlobalAvgPool2D()

        super(attention2d, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):

        self.dense1 = tf.keras.layers.Dense(units=1,
        #int(self.channel * self.reduction_ratio),
                                                 activation='relu',
                                                 kernel_initializer='he_normal')

        self.dense2 = tf.keras.layers.Dense(units=4,
        #int(self.channel),
                                                 activation='relu',
                                                 kernel_initializer='he_normal')

        self.con2d = KL.Conv2D(filters=1, 
                                                kernel_size=(3,3),
                                                padding='same', 
                                                activation='sigmoid',
                                                kernel_initializer='he_normal', 
                                                use_bias=False)
    
        self.resh = KL.Reshape(target_shape=(1,1,int(self.channel)))

        self.lam1 = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))
        self.lam2 = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))

        super(attention2d, self).build(input_shape)

    def call(self, inputs, chan):

        # get channel
        channel = chan # 4 or int(inputs.shape[3])
        maxpool_channel = self.pool1(inputs)
        avgpool_channel = self.pool2(inputs)
        # max path
        mlp_1_max = self.dense1(maxpool_channel)
        mlp_2_max = self.dense2(mlp_1_max)
        mlp_2_max = self.resh(mlp_2_max)
        # avg path
        mlp_1_avg = self.dense1(avgpool_channel)
        mlp_2_avg = self.dense2(mlp_1_avg)
        mlp_2_avg = self.resh(mlp_2_avg)
    
        channel_attention_feature = self.ad([mlp_2_max, mlp_2_avg])
        
        channel_att = self.mul([channel_attention_feature, inputs])
        # get space 
        maxpool_spatial = self.lam1(channel_att)
        avgpool_spatial = self.lam2(channel_att)
        max_avg_pool_spatial = KL.concatenate([maxpool_spatial, avgpool_spatial],axis=3)
        spatial_att = self.con2d(max_avg_pool_spatial)
        
        # block 
        refined_feature = self.mul([channel_att, spatial_att])
        output = self.ad([refined_feature, inputs])
        return output