
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import  *

class Involution2D():
    def __init__(self, filters, kernel_size = 3, strides = 1, padding = 'SAME', dilation_rate = 1, groups = 16, reduce_ratio = 4):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        
    def reduce_mapping(self, x):
        x = Conv2D(self.filters// self.reduce_ratio, 1, padding = self.padding)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    def span_mapping(self, x):
        return Conv2D(self.kernel_size * self.kernel_size * self.groups, self.kernel_size, padding = self.padding)(x)
    
    
    def __call__(self, x):
        if self.strides > 1:
            x_ = AveragePooling2D(self.strides)(x)
        else:
            x_ = x
        weight = self.span_mapping(self.reduce_mapping(x_))
        _, h, w, c = K.int_shape(weight)
        weight = Reshape((h, w, self.groups, self.kernel_size * self.kernel_size))(weight)
        weight = Lambda(K.expand_dims, arguments = {'axis':4})(weight)
        if self.filters != c:
            x = Conv2D(self.filters, 1, padding = self.padding)(x)
        out = Lambda(tf.extract_image_patches, arguments = {"ksizes":[1, self.kernel_size, self.kernel_size, 1],
                                                           "strides" :[1, self.strides, self.strides, 1], 
                                                           'rates' : [1, self.dilation_rate, self.dilation_rate, 1], 
                                                           'padding': self.padding})(x)
        #After tf.extract_image_patches, the channels are mixed
        #eg. what we want is [[channel_0[0, 0], channel_0[0, 1]...],  
        #                      [channel_1[0, 0], channel_1[0, 1]...], ... stacked in depth dimention]
        #but it is really [[channel_0[0, 0], channel_1[0, 0]...],
        #                   [channel_0[0, 1], channel_1[0, 1]...]..., stacked in depth  dimention]
        #so the following three lines is to get the wanted patches 
        out = Reshape((h, w, self.kernel_size * self.kernel_size, self.filters))(out)
        out = Permute((1, 2, 4, 3))(out)
        out = Reshape((h, w, self.kernel_size * self.kernel_size * self.filters), name = "get_right_patch")(out)
        
        out = Reshape((h, w, self.groups, self.filters // self.groups, self.kernel_size * self.kernel_size))(out)
        out = Multiply()([weight, out])
        out = Lambda(tf.reduce_sum, arguments = {'axis':-1})(out)
        out = Reshape((h, w, self.filters))(out)
        return out
        
