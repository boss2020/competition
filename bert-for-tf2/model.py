# coding=utf-8
#
# created by kpe on 28.Mar.2019 at 12:33
#

from __future__ import absolute_import, division, print_function

from tensorflow import keras
import params_flow as pf

from bert.layer import Layer
from bert.embeddings import BertEmbeddingsLayer
from bert.transformer import TransformerEncoderLayer
import sys

class BertModelLayer(Layer):
    """
    Implementation of BERT (arXiv:1810.04805), adapter-BERT (arXiv:1902.00751) and ALBERT (arXiv:1909.11942).

    See: https://arxiv.org/pdf/1810.04805.pdf - BERT
         https://arxiv.org/pdf/1902.00751.pdf - adapter-BERT
         https://arxiv.org/pdf/1909.11942.pdf - ALBERT

    """
    #def __init__(self):
    print('BertModelLayer __init__')
    #print('before __init__ self.params = ')
    #print(self.params)
    class Params(BertEmbeddingsLayer.Params,
         TransformerEncoderLayer.Params):
        print('BertModelLayer Params')
        pass
#print('after __init__ self.params = ')
#print(self.params)
#注意之前函数中的super.__init__()调用的是BertModelLayer之中的class Params
#的对应的函数内容

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        print('BertModelLayer _construct')
        print('origin self.params = ')
        print(self.params)
        super()._construct(**kwargs)
        print('after _construct self.params = ')
        print(self.params)
        #print('***bert.weights = ***')
        #print(self.weight
        #print(bert.weights)
        self.embeddings_layer = BertEmbeddingsLayer.from_params(
            self.params,
            name="embeddings"
        )
        #print('###bert.weights = ###')
        #print(bert.weights)
        # create all transformer encoder sub-layers
        self.encoders_layer = TransformerEncoderLayer.from_params(
            self.params,
            name="encoder"
        )
        #print('@@@bert.weights = @@@')
        #print(bert.weights)
        self.support_masking  = True
        print('BertModelLayer finish')

    # noinspection PyAttributeOutsideInit
    r"""def build(self, input_shape):
        print('BertModelLayer build')
        #print(sys._getframe().f_code.co_name)
        #print(sys._getframe().f_back.f_code.co_name)
        #print('input_shape = ')
        #print(input_shape)
        print(type(input_shape))
        #type(input_shape) = <class 'tensorflow.python.framework.tensor_shape.TensorShape'>
        if isinstance(input_shape, list):
            print('situation1')
            assert len(input_shape) == 2
            #判断len(input_shape)的长度是否为2，如果表达式为假，触发异常，
            #如果表达式为真，不执行任何操作
            input_ids_shape, token_type_ids_shape = input_shape
            print('input_ids_shape = ')
            print(input_ids_shape)
            print('token_type_ids_shape = ')
            print(token_type_ids_shape)
            self.input_spec = [keras.layers.InputSpec(shape=input_ids_shape),
                               keras.layers.InputSpec(shape=token_type_ids_shape)]
        else:
            print('situation2')
            input_ids_shape = input_shape
            print('input_ids_shape = ')
            print(input_ids_shape)
            #input_ids_shape = (None,128)
            self.input_spec = keras.layers.InputSpec(shape=input_ids_shape)
            #self.input_spec = InputSpec(shape=(None,128),ndim=2)
        super(BertModelLayer, self).build(input_shape)
        #调用BertModelLayer父类的build方法，等同于super().build(input_shape)
        print('input_spec = ')
        print(self.input_spec)
        #input_spec = InputSpec(shape=(None,128),ndim=2)
        print('BertModelLayer build finish')
        """

    def compute_output_shape(self, input_shape):
        print('BertModelLayer compute_output_shape')
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, _ = input_shape
        else:
            input_ids_shape = input_shape

        output_shape = list(input_ids_shape) + [self.params.hidden_size]
        print('BertModelLayer compute_output_shape finish')
        return output_shape

    def apply_adapter_freeze(self):
        print('BertModelLayer apply_adapter_freeze')
        """ Should be called once the model has been built to freeze
        all bet the adapter and layer normalization layers in BERT.
        """
        if self.params.adapter_size is not None:
            def freeze_selector(layer):
                return layer.name not in ["adapter-up", "adapter-down", "LayerNorm", "extra_word_embeddings"]
            pf.utils.freeze_leaf_layers(self, freeze_selector)
        print('BertModelLayer apply_adapter_freeze finish')

    def call(self, inputs, mask=None, training=None):
        print('BertModelLayer call')
        print('self.weights = ')
        print(self.weights)
        print(sys._getframe().f_code.co_name)
        #查看当前的函数
        print(sys._getframe().f_back.f_code.co_name)
        print('inputs = ')
        print(inputs)
        #inputs = Tensor("input_ids:0",shape=(None,128),dtype=int32)
        if mask is None:
            print('mask is None')
            #mask = self.embeddings_layer.compute_mask(inputs)
            #实际上返回的mask = None,这里简化操作
            mask = None
        #注意这里面在self.embeddings_layer之中放入mask的操作，先定义mask为self.embeddings_layer.compute_mask
        #然后在调用.call函数的时候直接放入定义好的mask值
        print('!!!!!!run to this!!!!!!')
        print('mask = ')
        print(mask)
        print('before embeddings')
        print('111self.weights1 = 111')
        print(self.weights)
        #self.weights = []
        embedding_output = self.embeddings_layer(inputs, mask=mask, training=training)
        print('after embeddings')
        print('111self.weights2 = 111')
        print(self.weights)
        #bert/embeddings/word_embeddings/embeddings:0,shape = (30522,768)
        #bert/embeddings/position_embeddings/embeddings:0,shape=(512,768)
        #bert/embeddings/LayerNorm/gamma:0,shape=(768,)
        #bert/embeddings/LayerNorm/beta:0,shape=(768,)
        print('!!!!!!embedding_output = !!!!!!')
        #mask = mask告诉self.encoders_layer这个函数输入的维度
        print(embedding_output)
        #embedding_output = Tensor("bert/embeddings/dropout/cond/Identity:0",shape=(None,128,768),
        #dtype = float32
        output           = self.encoders_layer(embedding_output, mask=mask, training=training)
        print('111self.weights3 = 111')
        print(self.weights)
        print('output = ')
        print(output)
        #调用了两次self.embeddings_layer.compute_mask函数
        #一次是self.embeddings_layer.compute_mask(inputs)
        #还有一次是self.embeddings_layer(inputs,mask=mask,training=training)

        #output = Tensor("bert/encoder/layer_11/output/LayerNorm/add_1:0",
        #shape = (None,128,768),dtype = float32)
        #这里的128为max_seq_len,每一句的最大长度,768为hidden_size,隐藏层大小
        print('BertModelLayer call finish')
        #print('***self.weights = !!!')
        #print(self.weights)
        return output   # [B, seq_len, hidden_size]

