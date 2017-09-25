import numpy as np
import tensorflow as tf
from six import string_types

DEFAULT_PADDING = 'SAME'


def layer(tf_op):
    def layer_decorated(self, *args, **kwargs):
        name = kwargs.setdefault('name', self.get_unique_name(tf_op.__name__))
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)

        layer_output = tf_op(self, layer_input, *args, **kwargs)


class Network():
    def __init__(self, inputs, trainable=True, is_trainging=False, num_classes=21):
        self.inputs = inputs
        # 当前操作层的输出节点list,下一操作层的输出节点
        self.layers = dict(inputs)
        self.trainable = trainable
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0), shape=[], name='use_dropout')
        self.setup(is_trainging, num_classes)

    def setup(self, is_trainging, num_classes):
        raise NotImplementedError('must be implemented by the subclass')

    def load(self, data_path, session, ignore_missing=False):
        '''
        加载权重参数w
        :param data_path: 
        :param session: 
        :param ignore_missing: 
        :return: 
        '''
        data_dict = np.load(data_path).items()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        assert len(args) != 0
        self.terminals = []  # 清空输入缓存
        for fed_layer in args:  # 输入存入缓存
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed:%s' % fed_layer)

            self.terminals.append(fed_layer)

        return self

    def get_output(self):
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        ident = sum(t.startwith(prefix) for t, _ in self.layers.items()) + 1
        # 追加索引
        return '%s_%d'.format(prefix, ident)

    def make_var(self, name, shape):
        '''新建tensorflow变量'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input,
             k_h, k_w, c_o, s_h, s_w,
             name, relu=True, padding=DEFAULT_PADDING,
             group=1, biased=True):
        '''2d卷积'''
        # 验证padding类型
        self.validate_padding(padding)
        # 输入通道
        c_i = input.get_shape().as_list()[-1]
        # 验证group参数
        assert c_i % group == 0
        assert c_o % group == 0
        convovle = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                output = convovle(input, kernel)
            else:
                # 拆分通道
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)

                output_groups = [convovle(i, k) for i, k in zip(input_groups, kernel_groups)]
                # 串接分组卷积结果
                output = tf.concat(3, output_groups)
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output
