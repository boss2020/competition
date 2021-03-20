# coding=utf-8
#
# created by kpe on 20.Mar.2019 at 16:30
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python import keras

from params_flow import LayerNormalization

from bert.attention import AttentionLayer
from bert.layer import Layer


class ProjectionLayer(Layer):
    print('ProjectionLayer __init__')
    class Params(Layer.Params):
        hidden_size        = None
        hidden_dropout     = 0.1
        initializer_range  = 0.02
        adapter_size       = None       # bottleneck size of the adapter - arXiv:1902.00751
        adapter_activation = "gelu"
        adapter_init_scale = 1e-3

    def _construct(self, **kwargs):
        print('ProjectionLayer _construct')
        super()._construct(**kwargs)
        self.dense      = None
        self.dropout    = None
        self.layer_norm = None

        self.adapter_down = None
        self.adapter_up   = None

        self.supports_masking = True
        print('ProjectionLayer _construct finish')

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        print('ProjectionLayer build')
        assert isinstance(input_shape, list) and 2 == len(input_shape)
        out_shape, residual_shape = input_shape
        print('out_shape = ')
        print(out_shape)
        #out_shape = (None,None,768),residual_shape = (None,None,768)(B,F,N*H)
        print('residual_shape = ')
        print(residual_shape)
        self.input_spec = [keras.layers.InputSpec(shape=out_shape),
                           keras.layers.InputSpec(shape=residual_shape)]
        #self.params.hidden_size = None
        self.dense = keras.layers.Dense(units=self.params.hidden_size,
                                        kernel_initializer=self.create_initializer(),
                                        name="dense")
        #self.params.hidden_dropout = 0.1
        self.dropout    = keras.layers.Dropout(rate=self.params.hidden_dropout)
        self.layer_norm = LayerNormalization(name="LayerNorm")

        if self.params.adapter_size is not None:
            self.adapter_down = keras.layers.Dense(units=self.params.adapter_size,
                                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                       stddev=self.params.adapter_init_scale),
                                                   activation=self.get_activation(self.params.adapter_activation),
                                                   name="adapter-down")
            #上面有定义self.params.adapter_activation="gelu",self.params.adapter_init_scale=1e-3
            self.adapter_up   = keras.layers.Dense(units=self.params.hidden_size,
                                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                       stddev=self.params.adapter_init_scale),
                                                   name="adapter-up")
        super(ProjectionLayer, self).build(input_shape)
        print('ProjectionLayer build finish')

    def call(self, inputs, mask=None, training=None, **kwargs):
        print('ProjectionLayer call')
        output, residual = inputs
        print('output1 = ')
        print(output)
        print('residual = ')
        print(residual)
        #output = (None,None,768),residual = (None,128,768)
        
        output = self.dense(output)
        print('output2 = ')
        print(output)
        print('self.weights = ')
        print(self.weights)
        #这里的self.dense中的units = 768,所以输出的output = (None,None,768)
        output = self.dropout(output, training=training)
        print('output3 = ')
        print(output)
        print('self.weights = ')
        print(self.weights)
        #output = (None,None,768)
        if self.adapter_down is not None:
            #这个if之中的内容不会被调用
            adapted = self.adapter_down(output)
            print('adapted1 = ')
            print(adapted)
            adapted = self.adapter_up(adapted)
            print('adapted2 = ')
            print(adapted)
            output = tf.add(output, adapted)
            print('output4 = ')
            print(output)
        output = self.layer_norm(tf.add(output, residual))
        print('output5 = ')
        print(output)
        print('self.weights = ')
        print(self.weights)
        #output = (None,128,768)
        print('ProjectionLayer call finish')
        return output


class TransformerSelfAttentionLayer(Layer):
    class Params(ProjectionLayer.Params,
                 AttentionLayer.Params):
        #print('TransformerSelfAttentionLayer Params')
        hidden_size         = None
        num_heads           = None
        hidden_dropout      = None
        attention_dropout   = 0.1
        initializer_range   = 0.02

    def _construct(self, **kwargs):
        print('TransformerSelfAttentionLayer _construct')
        super()._construct(**kwargs)
        params = self.params
        if params.hidden_size % params.num_heads != 0:
            raise ValueError("The hidden_size:[{}] is not a multiple of num_heads:[{}]".format(params.hidden_size,
                                                                                               params.num_heads))
        self.size_per_head = params.hidden_size // params.num_heads
        print('TransformerSelfAttentionLayer size_per_head = ')
        print(self.size_per_head)
        #self.size_per_head = 64
        assert params.size_per_head is None or self.size_per_head == params.size_per_head
        self.attention_layer     = None
        self.attention_projector = None

        self.supports_masking = True
        print('TransformerSelfAttentionLayer _construct finish')

    def build(self, input_shape):
        print('TransformerSelfAttentionLayer build')
        self.input_spec = keras.layers.InputSpec(shape=input_shape)
        print('input_shape = ')
        print(input_shape)
        self.attention_layer = AttentionLayer.from_params(
            self.params,
            size_per_head=self.size_per_head,
            name="self",
        )
        print('self.size_per_head = ')
        print(self.size_per_head)
        self.attention_projector = ProjectionLayer.from_params(
            self.params,
            name="output",
        )
        super(TransformerSelfAttentionLayer, self).build(input_shape)
        print('TransformerSelfAttentionLayer build finish')

    def call(self, inputs, mask=None, training=None):
        print('TransformerSelfAttentionLayer call')
        layer_input = inputs

        #
        # TODO: is it OK to recompute the 3D attention mask in each attention layer
        #
        attention_head   = self.attention_layer(layer_input, mask=mask, training=training)
        print('attention_head = ')
        print(attention_head)
        print('///self.weights1 = ///')
        print(self.weights)
        attention_output = self.attention_projector([attention_head, layer_input], mask=mask, training=training)
        print('attention_output = ')
        print(attention_output)
        print('///self.weights2 = ///')
        print(self.weights)
        print('TransformerSelfAttentionLayer call finish')
        return attention_output


class SingleTransformerEncoderLayer(Layer):
    """
    Multi-headed, single layer for the Transformer from 'Attention is All You Need' (arXiv: 1706.03762).

    See also: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """
    print('SingleTransformerEncoderLayer __init__')
    class Params(TransformerSelfAttentionLayer.Params,
                 ProjectionLayer.Params):
        #print('SingleTransformerEncoderLayer Params')
        intermediate_size       = None
        intermediate_activation = "gelu"

    def _construct(self, **kwargs):
        print('SingleTransformerEncoderLayer _construct')
        super()._construct(**kwargs)
        params = self.params
        if params.hidden_size % params.num_heads != 0:
            raise ValueError("The hidden_size:[{}] is not a multiple of num_heads:[{}]".format(params.hidden_size,
                                                                                               params.num_heads))
        self.size_per_head = params.hidden_size // params.num_heads

        self.self_attention_layer = None
        self.intermediate_layer   = None
        self.output_projector     = None
        print('SingleTransformerEncoderLayer _construct finish')

        self.supports_masking = True

    def build(self, input_shape):
        print('SingleTransformerEncoderLayer build')
        self.input_spec = keras.layers.InputSpec(shape=input_shape)  # [B, seq_len, hidden_size]

        self.self_attention_layer = TransformerSelfAttentionLayer.from_params(
            self.params,
            name="attention"
        )
        print('build after attention!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('self.params.intermediate_size = ')
        print(self.params.intermediate_size)
        self.intermediate_layer = keras.layers.Dense(
            name="intermediate",
            units=self.params.intermediate_size,
            activation=self.get_activation(self.params.intermediate_activation),
            kernel_initializer=self.create_initializer()
        )
        print('build after intermediate!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.output_projector = ProjectionLayer.from_params(
            self.params,
            name="output",
        )
        print('build after ProjectionLayer!!!!!!!!!!!!!!!!!!!!!!!!!!')
        super(SingleTransformerEncoderLayer, self).build(input_shape)
        print('SingleTransformerEncoderLayer build finish')

    def call(self, inputs, mask=None, training=None):
        print('SingleTransformerEncoderLayer call')
        layer_input = inputs

        attention_output    = self.self_attention_layer(layer_input, mask=mask, training=training)
        print('~~~self.weights1 = ~~~')
        print(self.weights)
        # intermediate
        intermediate_output = self.intermediate_layer(attention_output)
        print('~~~self.weights2 = ~~~')
        print(self.weights)
        # output
        layer_output = self.output_projector([intermediate_output, attention_output], mask=mask)
        print('~~~self.weights3 = ~~~')
        print(self.weights)
        print('SingleTransformerEncoderLayer call finish')
        return layer_output


class TransformerEncoderLayer(Layer):
    """
    Multi-headed, multi-layer Transformer from 'Attention is All You Need' (arXiv: 1706.03762).

    Implemented for BERT, with support for ALBERT (sharing encoder layer params).

    See also: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """
    print('TransformerEncoderLayer __init__')
    class Params(SingleTransformerEncoderLayer.Params):
        #print('TransformerEncoderLayer Params')
        num_layers     = None
        out_layer_ndxs = None   # [-1]

        shared_layer   = False  # False for BERT, True for ALBERT

    def _construct(self, **kwargs):
        print('TransformerEncoderLayer _construct')
        super()._construct(**kwargs)
        self.encoder_layers   = []
        self.shared_layer     = None  # for ALBERT
        self.supports_masking = True
        print('TransformerEncoderLayer params = ')
        print(self.params)
        r"""
        bert_params = 
        {'initializer_range': 0.02, 
        'max_position_embeddings': 512(无), 
        'hidden_size': 768, 
        'embedding_size': None, 'project_embeddings_with_bias': True, 
        'vocab_size': 30522, 'use_token_type': True, 'use_position_embeddings': True, 
        'token_type_vocab_size': 2, 
        'hidden_dropout': 0.1, 'extra_tokens_vocab_size': None, 
        'project_position_embeddings': True, 'mask_zero': False, 'adapter_size': None, 
        'adapter_activation': 'gelu', 'adapter_init_scale': 0.001, 'num_heads': 12, 
        'size_per_head': None, 'query_activation': None, 'key_activation': None, 
        'value_activation': None, 'attention_dropout': 0.1, 'negative_infinity': -10000.0, 
        'intermediate_size': 3072, 'intermediate_activation': 'gelu', 'num_layers': 12, 
        'out_layer_ndxs': None, 'shared_layer': False}
        TransformerEncoderLayer params = 
        {'initializer_range': 0.02, 'hidden_size': 768, 
        'hidden_dropout': 0.1, 'adapter_size': None, 
        'adapter_activation': 'gelu', 'adapter_init_scale': 0.001, 
        'num_heads': 12, 'size_per_head': None, 'query_activation': None, 
        'key_activation': None, 'value_activation': None, 
        'attention_dropout': 0.1, 'negative_infinity': -10000.0, 
        'intermediate_size': 3072, 'intermediate_activation': 'gelu', 
        'num_layers': 12, 'out_layer_ndxs': None, 'shared_layer': False}
        """
        print('TransformerEncoderLayer _construct finish')

    def build(self, input_shape):
        print('TransformerEncoderLayer build')
        self.input_spec = keras.layers.InputSpec(shape=input_shape)

        # create all transformer encoder sub-layers
        if self.params.shared_layer:
            # ALBERT: share params
            self.shared_layer = SingleTransformerEncoderLayer.from_params(self.params, name="layer_shared")
        else:
            # BERT
            for layer_ndx in range(self.params.num_layers):
                encoder_layer = SingleTransformerEncoderLayer.from_params(
                    self.params,
                    name="layer_{}".format(layer_ndx),
                )
                self.encoder_layers.append(encoder_layer)
        super(TransformerEncoderLayer, self).build(input_shape)
        print('TransformerEncoderLayer build finish')

    def call(self, inputs, mask=None, training=None):
        print('TransformerEncoderLayer call')
        layer_output = inputs
        print('+++self.weights0 = +++')
        print(self.weights)
        layer_outputs = []
        for layer_ndx in range(self.params.num_layers):
            print('layer_ndx = ')
            print(layer_ndx)
            encoder_layer = self.encoder_layers[layer_ndx] if self.encoder_layers else self.shared_layer
            layer_input = layer_output
            print('mask = ')
            print(mask)
            #mask = None
            layer_output = encoder_layer(layer_input, mask=mask, training=training)
            print('+++self.weights1 = +++')
            print(self.weights)
            layer_outputs.append(layer_output)
        print('after cycle')
        print('+++self.weights2 = +++')
        print(self.weights)
        if self.params.out_layer_ndxs is None:
            # return the final layer only
            print('out_layer_ndxs is None')
            final_output = layer_output
            print('+++self.weights3 = +++')
            print(self.weights)
        else:
            print('out_layer_ndxs is not None')
            final_output = []
            for ndx in self.params.out_layer_ndxs:
                final_output.append(layer_outputs[ndx])
            final_output = tuple(final_output)
            print('+++self.weights3 = +++')
            print(self.weights)
        print('TransformerEncoderLayer finish')
        return final_output


