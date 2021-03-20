# coding=utf-8
#
# created by kpe on 28.Mar.2019 at 12:33
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import params_flow as pf

from tensorflow import keras
from tensorflow.keras import backend as K

import bert


class PositionEmbeddingLayer(bert.Layer):
    class Params(bert.Layer.Params):
        #print('PositionEmbeddingLayer  Params')
        max_position_embeddings  = 512
        hidden_size              = 128
    #只改变上面的两个参数，其他的对应参数与bert中定义的一致
    #这里面的class Params为初始化类的部分，只要引入了PositionEmbeddingLayer就会跑
    #class Params之中的内容

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        print('PositionEmbeddingLayer _construct')
        super()._construct(**kwargs)
        self.embedding_table = None
        print('PositionEmbeddingLayer _construct finish')

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        print('PositionEmbeddingLayer build')
        # input_shape: () of seq_len
        r"""if input_shape is not None:
            assert input_shape.ndims == 0
            self.input_spec = keras.layers.InputSpec(shape=input_shape, dtype='int32')
        else:
            self.input_spec = keras.layers.InputSpec(shape=(), dtype='int32')
        """
        #K.floatx():以字符串形式返回默认的float类型，例如'float16','float32','float64'
        self.embedding_table = self.add_weight(name="embeddings",
                                               dtype=K.floatx(),
                                               shape=[self.params.max_position_embeddings, self.params.hidden_size],
                                               initializer=self.create_initializer())
        print('PositionEmbeddingLayer build finish')
        super(PositionEmbeddingLayer, self).build(input_shape)

    # noinspection PyUnusedLocal
    def call(self, inputs, **kwargs):
        # just return the embedding after verifying
        # that seq_len is less than max_position_embeddings
        print('PositionEmbeddingLayer call')
        seq_len = inputs

        assert_op = tf.compat.v2.debugging.assert_less_equal(seq_len, self.params.max_position_embeddings)

        with tf.control_dependencies([assert_op]):
            # slice to seq_len
            full_position_embeddings = tf.slice(self.embedding_table,
                                                [0, 0],
                                                [seq_len, -1])
        output = full_position_embeddings
        print('PositionEmbeddingLayer call finish')
        return output


class EmbeddingsProjector(bert.Layer):
    class Params(bert.Layer.Params):
        #print('EmbeddingsProjector Params')
        hidden_size                  = 768
        embedding_size               = None   # None for BERT, not None for ALBERT
        project_embeddings_with_bias = True   # in ALBERT - True for Google, False for brightmart/albert_zh

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        print('EmbeddingsProjector _construct')
        super()._construct(**kwargs)
        self.projector_layer      = None   # for ALBERT
        self.projector_bias_layer = None   # for ALBERT
        print('EmbeddingsProjector _construct finish')

    def build(self, input_shape):
        print('EmbeddingsProjector build')
        emb_shape = input_shape
        self.input_spec = keras.layers.InputSpec(shape=emb_shape)
        assert emb_shape[-1] == self.params.embedding_size

        # ALBERT word embeddings projection
        self.projector_layer = self.add_weight(name="projector",
                                               shape=[self.params.embedding_size,
                                                      self.params.hidden_size],
                                               dtype=K.floatx())
        if self.params.project_embeddings_with_bias:
            self.projector_bias_layer = self.add_weight(name="bias",
                                                        shape=[self.params.hidden_size],
                                                        dtype=K.floatx())
        super(EmbeddingsProjector, self).build(input_shape)
        print('EmbeddingsProjectore build finish')

    def call(self, inputs, **kwargs):
        print('EmbeddingsProjector call')
        input_embedding = inputs
        assert input_embedding.shape[-1] == self.params.embedding_size

        # ALBERT: project embedding to hidden_size
        output = tf.matmul(input_embedding, self.projector_layer)
        if self.projector_bias_layer is not None:
            output = tf.add(output, self.projector_bias_layer)
        print('EmbeddingsProjector call finish')
        return output


class BertEmbeddingsLayer(bert.Layer):
    class Params(PositionEmbeddingLayer.Params,
                 EmbeddingsProjector.Params):
        #print('BertEmbeddingsLayer Params')
        vocab_size               = None
        use_token_type           = True
        use_position_embeddings  = True
        token_type_vocab_size    = 2
        hidden_size              = 768
        hidden_dropout           = 0.1

        extra_tokens_vocab_size  = None  # size of the extra (task specific) token vocabulary (using negative token ids)

        #
        # ALBERT support - set embedding_size (or None for BERT)
        #
        embedding_size               = None   # None for BERT, not None for ALBERT
        project_embeddings_with_bias = True   # in ALBERT - True for Google, False for brightmart/albert_zh
        project_position_embeddings  = True   # in ALEBRT - True for Google, False for brightmart/albert_zh

        mask_zero                    = False

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        print('BertEmbeddingsLayer _construct')
        super()._construct(**kwargs)
        self.word_embeddings_layer       = None
        self.extra_word_embeddings_layer = None   # for task specific tokens (negative token ids)
        self.token_type_embeddings_layer = None
        self.position_embeddings_layer   = None
        self.word_embeddings_projector_layer = None   # for ALBERT
        self.layer_norm_layer = None

        self.support_masking = self.params.mask_zero
        print('self.params = ')
        print(self.params)
        r"""
        之前对应的model.py之中的bert_params的对应值为
        bert_params = 
        {'initializer_range': 0.02, 'max_position_embeddings': 512, 
        'hidden_size': 768, 'embedding_size': None, 'project_embeddings_with_bias': True, 
        'vocab_size': 30522, 'use_token_type': True, 'use_position_embeddings': True, 
        'token_type_vocab_size': 2, 'hidden_dropout': 0.1, 'extra_tokens_vocab_size': None, 
         'project_position_embeddings': True, 'mask_zero': False, 
        self.params只到上面一块params的对应内容
        'adapter_size': None, 
        'adapter_activation': 'gelu', 'adapter_init_scale': 0.001, 'num_heads': 12, 
        'size_per_head': None, 'query_activation': None, 'key_activation': None, 
        'value_activation': None, 'attention_dropout': 0.1, 'negative_infinity': -10000.0, 
        'intermediate_size': 3072, 'intermediate_activation': 'gelu', 'num_layers': 12, 
        'out_layer_ndxs': None, 'shared_layer': False}
        而当前模块对应的params内容为
        self.params = 
        {'initializer_range': 0.02, 'max_position_embeddings': 512, 'hidden_size': 768, 
        'embedding_size': None, 'project_embeddings_with_bias': True, 'vocab_size': 30522, 
        'use_token_type': True, 'use_position_embeddings': True, 'token_type_vocab_size': 2, 
        'hidden_dropout': 0.1, 'extra_tokens_vocab_size': None, 'project_position_embeddings': True, 
        'mask_zero': False}
        """
        print('BertEmbeddingsLayer _construct finish')

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        print('BertEmbeddingsLayer build')
        r"""if isinstance(input_shape, list):
            print('situation1')
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
            self.input_spec = [keras.layers.InputSpec(shape=input_ids_shape),
                               keras.layers.InputSpec(shape=token_type_ids_shape)]
        else:
            print('situation2')
            input_ids_shape = input_shape
            self.input_spec = keras.layers.InputSpec(shape=input_ids_shape)
        print('input_ids_shape = ')
        print(input_ids_shape)
        #这里调用situation2,input_ids_shape = (None,128)
        print('self.input_spec = ')
        print(self.input_spec)
        """
        r"""self.input_spec = InputSpec(shape=(None,128),ndim=2)
        这里的ndim是指放入的数组内层的维度
        比如b = [
                   [1,2,3],
                   [4,5,6],
                   [7,8,9]
               ]
        此时ndim = 2
        c = [
               [
                   [1,2,3],
                   [4,5,6]
               ]
            ]
        此时ndim = 3
        """
        # use either hidden_size for BERT or embedding_size for ALBERT
        print('self.params.embedding_size = ')
        print(self.params.embedding_size)
        # self.params.embedding-size = None
        #嵌入层将正整数(下标)转换为具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]]
        embedding_size = self.params.hidden_size if self.params.embedding_size is None else self.params.embedding_size
        # embedding_size = 768
        # 因为self.params.embedding_size = None,所以embedding_size = 768
        #input_dim = self.params.vocab_size = 30522
        #output_dim = embedding_size = 768
        #self.params.mask_zero = False
        r"""
        self.word_embeddings_layer = keras.layers.Embedding(
            input_dim = 30522,
            output_dim = 768,
            mask_zero = False,
            name = "word_embeddings"
        )
        """
        print('self.weight1 = ')
        print(self.weights)
        self.word_embeddings_layer = keras.layers.Embedding(
            input_dim=self.params.vocab_size,
            output_dim=embedding_size,
            mask_zero=self.params.mask_zero,
            name="word_embeddings"
        )
        print('self.weight2 = ')
        print(self.weights)
        #bert输入之中的对应的字向量
        #input_dim:词汇表的大小，即最大整数index+1。
        #output_dim:词向量的维度
        #mask_zero:是否把0看作一个应该被遮蔽的特殊的padding值
        #self.params.mask_zero = False
        #默认为False，所以感觉这里不需要特别写出来
        #self.params.extra_tokens_vocab_size = None
        
        #下面的两个if都没有运行
        if self.params.extra_tokens_vocab_size is not None:
            #这一段没有运行，应该也是属于ALBERT类的操作
            print('extra_tokens_vocab is not None')
            self.extra_word_embeddings_layer = keras.layers.Embedding(
                input_dim=self.params.extra_tokens_vocab_size + 1,  # +1 is for a <pad>/0 vector
                output_dim=embedding_size,
                mask_zero=self.params.mask_zero,
                embeddings_initializer=self.create_initializer(),
                name="extra_word_embeddings"
            )
            #input_dim:大或等于0的整数，字典长度，即输入数据最大下标+1
            #output_dim:大于0的整数，代表全连接嵌入的维度
            #mask_zero:布尔值，是否将输入中的'0'看作是应该被忽略的填充值，该参数在使用递归层处理变长
            #输入时有用。设置为True的时候，模型中的后续的层必须都支持masking，否则会抛出异常
            print('self.extra_word_embeddings_layer = ')
            print(self.extra_word_embeddings_layer)

        # ALBERT word embeddings projection
        if self.params.embedding_size is not None:
            print('self.params.embedding_size is not None')
            self.word_embeddings_projector_layer = EmbeddingsProjector.from_params(
                self.params, name="word_embeddings_projector")

        #到这之后才开始运行
        #self.params.project_position_embeddings = True
        position_embedding_size = embedding_size if self.params.project_position_embeddings else self.params.hidden_size
        #position_embedding_size = 768
        if self.params.use_token_type:
            print('self.params.token_type_vocab_size = ')
            print(self.params.token_type_vocab_size)
            #self.params.token_type_vocab_size = 2
            r"""
            self.token_type_embeddings_layer = keras.layers.Embedding(
                input_dim = 2,
                output_dim = 768,
                mask_zero = False,
                name = "token_type_embedding"
            )
            """
            self.token_type_embeddings_layer = keras.layers.Embedding(
                input_dim=self.params.token_type_vocab_size,
                output_dim=position_embedding_size,
                mask_zero=False,
                name="token_type_embeddings"
            )
        #之前self.params.use_token_type下面一行的print没有打完，在这个位置
        #下面一行报错invalid syntax
        if self.params.use_position_embeddings:
            print('self.params.use_position_embeddings = ')
            print(self.params.use_position_embeddings)
            #self.params.use_position_embeddings = True
            self.position_embeddings_layer = PositionEmbeddingLayer.from_params(
                self.params,
                name="position_embeddings",
                hidden_size=position_embedding_size
            )
            #这个里面只是调用了一下_construct的对应的函数

        self.layer_norm_layer = pf.LayerNormalization(name="LayerNorm")
        #目标文件在/enter/lib/pythono3.8/site-packages/params_flow/normalization.py
        self.dropout_layer    = keras.layers.Dropout(rate=self.params.hidden_dropout)

        super(BertEmbeddingsLayer, self).build(input_shape)
        print('BertEmbeddingsLayer build finish')

    def call(self, inputs, mask=None, training=None):
        print('BertEmbeddingsLayer call')
        if isinstance(inputs, list):
            print('isinstance situation1')
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            print('isinstance situation2')
            input_ids      = inputs
            token_type_ids = None
        #这里调用isinstance situation2
        #input_ids = Tensor("input_ids:0",shape=(None,128),dtype=int32)
        #token_type_ids = None
        input_ids = tf.cast(input_ids, dtype=tf.int32)
        #input_ids = Tensor("input_ids:0",shape=(None,128),dtype=int32)
        #这里的input_ids好像没什么变化
        print('self.weight3 = ')
        print(self.weights)
        print('input_ids = ')
        print(input_ids)
        if self.extra_word_embeddings_layer is not None:
            print('self.extra_word_embeddings_layer situation1')
            token_mask   = tf.cast(tf.greater_equal(input_ids, 0), tf.int32)
            extra_mask   = tf.cast(tf.less(input_ids, 0), tf.int32)
            token_ids    = token_mask * input_ids
            extra_tokens = extra_mask * (-input_ids)
            token_output = self.word_embeddings_layer(token_ids)
            extra_output = self.extra_word_embeddings_layer(extra_tokens)
            embedding_output = tf.add(token_output,
                                      extra_output * tf.expand_dims(tf.cast(extra_mask, K.floatx()), axis=-1))
        else:
            print('self extra_word_embeddings_layer situation2')
            embedding_output = self.word_embeddings_layer(input_ids)
            #embedding_output = Tensor("bert/embeddings/word_embeddings/embedding_lookup
            #/Identity_l:0",shape = (None,128,768),dtype = float32)
            #经历过这个层之后初始化了一个(30522,768)的权重矩阵
        print('self.weight4 = ')
        print(self.weights)
        print('embedding_output = ')
        print(embedding_output)

        # ALBERT: for brightmart/albert_zh weights - project only token embeddings
        # 下面两个对应的if内容都没有运转
        if not self.params.project_position_embeddings:
            print('not self.params.project_position_embeddings')
            if self.word_embeddings_projector_layer:
                print('have word_embeddings_projector_layer')
                embedding_output = self.word_embeddings_projector_layer(embedding_output)
        if token_type_ids is not None:
            print('token_type_ids is not None')
            token_type_ids    = tf.cast(token_type_ids, dtype=tf.int32)
            embedding_output += self.token_type_embeddings_layer(token_type_ids)
            print('token_type_ids = ')
            print(token_type_ids)
            print('embedding_output = ')
            print(embedding_output)
        print('self.weight5 = ')
        print(self.weights)
        if self.position_embeddings_layer is not None:
            print('self.position_embeddings_layer is not None')
            seq_len  = input_ids.shape.as_list()[1]
            emb_size = embedding_output.shape[-1]
            print('seq_len = ')
            print(seq_len)
            print('emb_size = ')
            print(emb_size)
            #seq_len = 128,emb_size = 768
            pos_embeddings = self.position_embeddings_layer(seq_len)
            # broadcast over all dimension except the last two [..., seq_len, width]
            #这里调用了PositionEmbeddingLayer build,PositionEmbeddingLayer build finish
            #以及PositionEmbeddingLayer call,PositionEmbeddingLayer call finish函数
            print('embedding_output.shape.ndims = ')
            print(embedding_output.shape.ndims)
            broadcast_shape = [1] * (embedding_output.shape.ndims - 2) + [seq_len, emb_size]
            embedding_output += tf.reshape(pos_embeddings, broadcast_shape)
            print('pos_embeddings = ')
            print(pos_embeddings)
            print('broadcast_shape = ')
            print(broadcast_shape)
            print('embedding_output = ')
            print(embedding_output)
            #pos_embeddings = Tensor("bert/embeddings/position_embeddings/Slice:0"
            #shape = (128,768),dtype = float32)
            #broadcast_shape = [1,128,768]
            #embedding_output = Tensor("bert/embeddings/dropout/cond/Identity:0",
            #shape=(None,128,768),dtype = float32)
        print('self.weight6 = ')
        print(self.weights)
        embedding_output = self.layer_norm_layer(embedding_output)
        #self.layer_norm_layer为上面定义的残差函数
        print('self.weight7 = ')
        print(self.weights)
        embedding_output = self.dropout_layer(embedding_output, training=training)
        #self.dropout_layer为上面定义的dropout内容

        # ALBERT: for google-research/albert weights - project all embeddings
        # 下面这个if的内容没有被调用过
        if self.params.project_position_embeddings:
            if self.word_embeddings_projector_layer:
                print('self.word_embeddings_projector_layer run to this')
                embedding_output = self.word_embeddings_projector_layer(embedding_output)
                #self.word_embeddings_projector_layer = EmbeddingsProjector.from_params(self.params, name="word_embeddings_projector")
        print('embedding_output = ')
        print(embedding_output)
        #embedding_output = Tensor("bert/embeddings/dropout/cond/Identity:0",shape=(None,128,768)
        #dtype=float32)
        print('BertEmbeddingsLayer call finish')
        return embedding_output   # [B, seq_len, hidden_size]

    def compute_mask(self, inputs, mask=None):
        print('BertEmbeddingsLayer compute_mask')
        if isinstance(inputs, list):
            print('situation1')
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
            print('input_ids = ')
            print(input_ids)
            print('token_type_ids = ')
            print(token_type_ids)
        else:
            print('situation2')
            input_ids      = inputs
            print('input_ids = ')
            print(input_ids)
            token_type_ids = None

        if not self.support_masking:
            print('return function')
            return None
        print('BertEmbeddingsLayer compute_mask finish')
        return tf.not_equal(input_ids, 0)
     
