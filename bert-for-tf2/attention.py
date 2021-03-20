# coding=utf-8
#
# created by kpe on 15.Mar.2019 at 12:52
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

from bert.layer import Layer


class AttentionLayer(Layer):
    print('AttentionLayer __init__')
    class Params(Layer.Params):
        num_heads         = None
        size_per_head     = None
        initializer_range = 0.02
        query_activation  = None
        key_activation    = None
        value_activation  = None
        attention_dropout = 0.1
        negative_infinity = -10000.0  # used for attention scores before softmax

    @staticmethod
    def create_attention_mask(from_shape, input_mask):
        print('AttentionLayer create_attention_mask')
        """
        Creates 3D attention.
        :param from_shape:  [batch_size, from_seq_len, ...]
        :param input_mask:  [batch_size, seq_len]
        :return: [batch_size, from_seq_len, seq_len]
        """
        print('from_shape = ')
        print(from_shape)
        #from_shape = Tensor("bert/encoder/layer_0/attention/self/Shape_1:0",shape = (3,))
        #这里的from_shape为输入的from_tensor(None,128,768)的对应形状
        print('input_mask = ')
        print(input_mask)
        #input_mask = (None,128),为全一构成的(None,128)的对应数组
        mask = tf.cast(tf.expand_dims(input_mask, axis=1), tf.float32)                   # [B, 1, T]
        print('mask = ')
        print(mask)
        #mask = (None,1,128)
        ones = tf.expand_dims(tf.ones(shape=from_shape[:2], dtype=tf.float32), axis=-1)  # [B, F, 1]
        print('ones = ')
        print(ones)
        #ones = (None,128,1)
        mask = ones * mask  # broadcast along two dimensions
        print('new mask = ')
        print(mask)
        #mask = (None,128,128)
        print('AttentionLayer create_attention_mask finish')
        return mask  # [B, F, T]

    def _construct(self, **kwargs):
        print('AttentionLayer _construct')
        super()._construct(**kwargs)
        self.query_activation = self.params.query_activation
        self.key_activation   = self.params.key_activation
        self.value_activation = self.params.value_activation

        self.query_layer = None
        self.key_layer   = None
        self.value_layer = None

        self.supports_masking = True
        print('AttentionLayer _construct finish')

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        print('AttentionLayer build')
        self.input_spec = keras.layers.InputSpec(shape=input_shape)
        print('input_shape = ')
        print(input_shape)
        dense_units = self.params.num_heads * self.params.size_per_head  # N*H
        #
        # B, F, T, N, H - batch, from_seq_len, to_seq_len, num_heads, size_per_head
        #
        print('self.params = ')
        print(self.params)
        print('dense_units = ')
        print(dense_units)
        print('self.query_activation = ')
        print(self.query_activation)
        print('self.key_activation = ')
        print(self.key_activation)
        print('self.value_activation = ')
        print(self.value_activation)
        #dense_units = 768
        #self.query_activation = None,self.key_activation = None
        #self.value_activation = None
        self.query_layer = keras.layers.Dense(units=dense_units, activation=self.query_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="query")
        self.key_layer   = keras.layers.Dense(units=dense_units, activation=self.key_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="key")
        self.value_layer = keras.layers.Dense(units=dense_units, activation=self.value_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="value")
        #这几个部分相当于输入的词装入矩阵之中的x*WQ=Q,x*WK=K,x*WV=V
        print('self.query_layer = ')
        print(self.query_layer)
        #keras.layers.Dense为经常使用到的全连接层，dense
        print('(((self.weights = )))')
        print(self.weights)
        self.dropout_layer = keras.layers.Dropout(self.params.attention_dropout)

        super(AttentionLayer, self).build(input_shape)
        print('AttentionLayer build finish')

    def compute_output_shape(self, input_shape):
        print('AttentionLayer compute_output_shape')
        from_shape = input_shape

        # from_shape         # [B, F, W]   [batch_size, from_seq_length, from_width]
        # input_mask_shape   # [B, F]

        output_shape = [from_shape[0], from_shape[1], self.params.num_heads * self.params.size_per_head]
        print('AttentionLayer compute_output_shape finish')
        return output_shape  # [B, F, N*H]

    # noinspection PyUnusedLocal
    def call(self, inputs, mask=None, training=None, **kwargs):
        print('AttentionLayer call')
        from_tensor = inputs
        to_tensor   = inputs
        print('from_tensor = ')
        print(from_tensor)
        #from_tensor = (None,128,768)
        #to_tensor = (None,128,768)
        if mask is None:
            sh = self.get_shape_list(from_tensor)
            #这个对应的self.get_shape_list查阅百度之后发现没有找到这个函数
            #然后找寻当前这个类，也没有找到对应的get_shape_list函数
            #此时就要考虑去它的父类之中找寻对应的get_shape_list方法
            #因为父类定义的方法是可以在子类之中直接进行调用的
            #找寻之后发现在/python3.8/site-packages/params_flow/layer.py
            #之中可以找到对应的get_shape_list函数的定义，
            #这里的get_shape_list函数是返回from_tensor的一个对应形状的数组
            print('sh = ')
            print(sh)
            data = sh[:2]
            print('data = ')
            print(data)
            #data[0] = 
            print('data type = ')
            print(type(data))
            #sh = [<tf.Tensor 'bert/encoder/layer_0/attention/self/strided_slice_1:0' shape=() dtype=int32>,128,768]
            mask = tf.ones(sh[:2], dtype=tf.int32)
            #mask = tf.ones(data,dtype=tf.int32)
            #print('$$$$$$mask = ')
            #print(mask)
            #mask = Tensor("bert/encoder/layer_0/attention/self/ones:0", shape=(None, 128), dtype=int32)
        attention_mask = AttentionLayer.create_attention_mask(tf.shape(input=from_tensor), mask)
        #注意这里传入的是tf.shape(input=from_tensor),对应内容为Tensor(...,shape = (3,))
        #mask = (None,128)
        print('attention_mask = ')
        print(attention_mask)
        #  from_tensor shape - [batch_size, from_seq_length, from_width]
        #attention_mask = (None,128,128),通过之前的(None,128)扩展一个维度为(None,1,128)
        #ones为一个(None,128,1)的对应的全1构成的矩阵数组，然后attention_mask为ones*mask的结果
        #attention_mask = (None,128,128)为相乘的结果
        input_shape  = tf.shape(input=from_tensor)
        print('input_shape = ')
        print(input_shape)
        #input_shape = Tensor("bert/.../Shape_2:0",shape=(3,))
        batch_size, from_seq_len, from_width = input_shape[0], input_shape[1], input_shape[2]
        print('batch_size = ')
        print(batch_size)
        #batch_size = Tensor("bert/...strided_slice_6:0",shape=(),dtype=int32)
        print('from_seq_len = ')
        print(from_seq_len)
        #from_seq_len = Tensor("bert/...strided_slice_7:0",shape=(),dtype=int32)
        print('from_width = ')
        print(from_width)
        #from_width = Tensor("bert/...strided_slice_8:0",shape=(),dtype=int32)
        to_seq_len = from_seq_len
        
        # [B, F, N*H] -> [B, N, F, H]
        # B:batch_size,F:from_seq_length,W:from_width,N:self.params.num_heads
        def transpose_for_scores(input_tensor, seq_len):
            print('transpose_for_scores')
            print('input_tensor = ')
            print(input_tensor)
            #input_tensor = Tensor("bert/...BiasAdd:0",shape=(None,128,768))
            print('seq_len = ')
            print(seq_len)
            #seq_len = Tensor("bert/...strided_slice_7:0",shape=())
            output_shape = [batch_size, seq_len,
                            self.params.num_heads, self.params.size_per_head]
            print('output_shape = ')
            print(output_shape)
            #output_shape = [shape=(),shape=(),12,64]
            #output_tensor = Tensor("bert/...:0",shape=(None,None,12,64),dtype=float32)
            #将768拆分成为12个64,12*64=768
            output_tensor = K.reshape(input_tensor, output_shape)
            print('output_tensor = ')
            print(output_tensor)
            #output_tensor = (None,None,12,64)
            #切换过来之后为(None,12,None,64)
            return tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])  # [B,N,F,H]

        print('from_tensor = ')
        print(from_tensor)
        #from_tensor = Tensor("bert/...",shape=(None,128,768),dtype=float32)
        query = self.query_layer(from_tensor)  # [B,F, N*H] [batch_size, from_seq_len, N*H]
        #上面的units = dense_units = 768对应的是输出的维度
        #将输入句子的词嵌入装进矩阵X中，将其乘以我们训练的权重矩阵
        #X*WQ = Q
        key   = self.key_layer(to_tensor)      # [B,T, N*H]
        #X*WK = K
        value = self.value_layer(to_tensor)    # [B,T, N*H]
        #X*WV = V
        print('```query = ')
        print(query)
        #query = Tensor("bert/...BiasAdd:0",shape=(None,128,768),dtype=float32)
        #key = Tensor("bert/...BiasAdd:0",shape=(None,128,768),dtype=float32)
        #value = Tensor("bert/.../key/BiasAdd:0",shape=(None,128,768),dtype=float32)
        #这里的query,key和经过全连接层之后维度保持不变
        print('```key = ')
        print(key)
        print('```value = ')
        print(value)

        query = transpose_for_scores(query, from_seq_len)           # [B, N, F, H]
        key   = transpose_for_scores(key,   to_seq_len)             # [B, N, T, H]
        #B:batch_size,N:self.params.num_heads,F:from_seq_length,W:from_width
        #B:批次大小，N:注意力头数，F:每个批次的长度，W:每个单词数值的权重
        print('after transpose')
        print('###query = ')
        print(query)
        print('###key = ')
        print(key)
        print('###value = ')
        print(value)
        #transpose之后query = (None,12,None,64),key = (None,12,None,64),value = (None,128,768)
        transpose_key = tf.transpose(key)
        print('%%%self.weights1 = %%%')
        print(self.weights)
        print('transpose_key = ')
        print(transpose_key)
        #transpose_key = Tensor("bert/...:0",shape=(64,None,12,None),dtype=float32)
        #这里要使用tf.transpose的对应函数，def  transpose(a,perm=None,name="transpose")
        #Transposes 'a'.Permutes the dimensions according to 'perm',这里面是将key的形状
        #进行翻转过来,transpose_key = (64,None,12,None)
        attention_scores = tf.matmul(query, key, transpose_b=True)  # [B,N,F,H]*[B,N,H,T] = [B,N,F,T]
        #这里面有一个是头数，这是将多头混为一个四维的矩阵操作
        #transpose_b = True,b在进行乘法计算前进行转置
        #注意这里的转置操作并不是进行维度的完全翻转，而只是翻转最后的两个维度
        print('attention_scores = ')
        print(attention_scores)
        #attention_scores = (None,12,None,None)
        #(None,12,None,64)*(None,12,64,None) = (None,12,None,None)
        #这里两个矩阵相乘操作的时候[B,N,F,H]*[B,N,H,T] = [B,N,F,T]
        attention_scores = attention_scores / tf.sqrt(float(self.params.size_per_head))
        #这里使用公式(Q*KT)/(根号dk)
        print('***attention_scores = ')
        print(attention_scores)
        print('%%%self.weights2 = %%%')
        print(self.weights)
        #attention_scores = (None,12,None,None)，对应着数组形状[B,N,F,T]
        #对应的为batch:批次,num:注意力头数,from_seq_len:每一个批次的长度
        if attention_mask is not None:
            print('attention_mask is not None')
            attention_mask = tf.expand_dims(attention_mask, axis=1)  # [B, 1, F, T]
            #原先的attention_mask = (None,128,128),为对应的
            # {1, 0} -> {0.0, -inf}
            print('attention_mask = ')
            print(attention_mask)
            #attention_mask = (None,1,128,128)
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * self.params.negative_infinity
            print('adder = ')
            print(adder)
            #adder = (None,1,128,128),dtype = float32
            print('negative_infinity = ')
            print(self.params.negative_infinity)
            #negative_infinity = -10000.0
            #attention_scores = tf.add(attention_scores, adder)  # adding to softmax -> its like removing them entirely
            #上面的tf.add(attention_scores,adder)原先有这步操作，现在去除了
            print('attention_scores = ')
            print(attention_scores)
            #attention_scores = (None,12,128,128),原先的attention_scores = (None,12,None,None),加上了adder之后
            #得到了(None,12,128,128),如果去除掉tf.add(attention_scores,adder)这步操作之后，
            #attention_scores = (None,12,None,None),这里对应的形状为[B,N,F,F]
        # scores to probabilities
        attention_probs = tf.nn.softmax(attention_scores)           # [B, N, F, T]
        #对之前求得的值使用softmax，softmax((Q*KT)/(根号dk))
        #这里的attention_scores的对应值应该为(None,12,None,None)，感觉内容为[B,N,F,T]
        print('attention_probs = ')
        print(attention_probs)
        print('%%%self.weights3 = %%%')
        print(self.weights)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout_layer(attention_probs,
                                             training=training)    # [B, N, F, T]
        print('after dropout attention_probs = ')
        print(attention_probs)
        # [B,N,F,T] = [batch_size,self.params.num_heads,from_seq,to_seq_len]
        value = tf.reshape(value, [batch_size, to_seq_len,
                                   self.params.num_heads, self.params.size_per_head])
        print('value = ')
        print(value)
        #value = (None,None,12,64),[B,T,N,H]
        value = tf.transpose(a=value, perm=[0, 2, 1, 3])                                # [B, N, T, H]
        print('value1 = ')
        print(value)
        #value = (None,12,None,64)
        print('%%%self.weights4 = %%%')
        print(self.weights)
        context_layer = tf.matmul(attention_probs, value)                               # [B, N, F, H]
        #对上面的soft((Q*KT)/(根号dk))的结果再乘上一个对应的v值,这里调整形状就是为了两者能够很好地相乘
        #attention_probs = [B,N,F,T],value = [B,N,T,H],相乘之后为[B,N,F,H]
        #按照代码注释的说法，
        print('context_layer = ')
        print(context_layer)
        #context_layer = (None,12,None,64)
        context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])                # [B, F, N, H]
        print('after transpose')
        print('***context_layer = ***')
        print(context_layer)
        #after transpose,context_layer = (None,None,12,64)
        output_shape = [batch_size, from_seq_len,
                        self.params.num_heads * self.params.size_per_head]
        print('output_shape = ')
        print(output_shape)
        print('%%%self.weights5 = %%%')
        print(self.weights)
        #output_shape = [<tf.Tensor 'bert/encoder....' shape=(),dtype=int32>,<tf.Tensor 'bert/encoder...'
        #shape=(),dtype=int32>,768]
        context_layer = tf.reshape(context_layer, output_shape)
        print('context_layer = ')
        print(context_layer)
        #这里虽然output_shape为一个list类型的数值，但是context_layer可以转化成为一个相应的Tensor类型的数值
        #context_layer = Tensor("bert/encoder...",shape=(None,None,768),dtype=float32)
        print('AttentionLayer call finish')
        return context_layer                                                            # [B, F, N*H]

    # noinspection PyUnusedLocal
    def compute_mask(self, inputs, mask=None):
        print('AttentionLayer compute_mask')
        print('mask = ')
        print(mask)
        return mask   # [B, F]

