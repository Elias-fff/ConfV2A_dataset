from keras.layers import Lambda, Dense, Layer, Conv1D
import tensorflow as tf


class TCNCell(Layer):
    """
    sumary_line:
    Chinese:让输入的时间序列[bs,seql,dim]提升kernel_size倍的感受野
    English: Double the receptive field of the input time series [bs, seql, dim]
    """
    def __init__(self, filters=32, ks=3, activation=None, name=None, **kwargs):
        self.filters = filters
        self.ks = ks
        self.activation = activation
        super(TCNCell, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3, f"Input shape should be [batch, timesteps, features], but got {input_shape}"
        bs, seq_l, dim = input_shape

        if seq_l == 1:
            # 当序列长度为1时，直接使用 Dense 层
            self.out = Dense(self.filters, activation=self.activation if self.activation else 'relu')
            # Dense 层会在第一次 call 时自动构建，但我们可以提前 build 它
            self.out.build((bs, dim))  # 注意：Dense 期望输入是 (batch, features)，所以我们传入 (bs, dim)
        else:
            # 需要 padding 到 ks 的整数倍
            if seq_l % self.ks != 0:
                self.maxlen = seq_l + self.ks - (seq_l % self.ks)
                self.pad_layer = Lambda(
                    lambda x: tf.pad(tensor=x, paddings=[[0, 0], [self.maxlen - seq_l, 0], [0, 0]], constant_values=0),
                    output_shape=(self.maxlen, dim)
                )
                assert self.maxlen % self.ks == 0, 'kernel size should be divisible by padded input length'
            else:
                self.maxlen = seq_l

            # 创建 Conv1D 层
            self.tcn_cell = Conv1D(filters=self.filters, kernel_size=self.ks, strides=self.ks,
                                   activation=self.activation, padding='valid')
            # Conv1D 会在 call 时自动构建，但我们可以显式 build
            self.tcn_cell.build((bs, self.maxlen, dim))

        self.built = True  # 手动标记为已构建（可选，因为父类会处理）

    def call(self, x):
        if x.shape[1] == 1 and hasattr(self, 'out'):
            # 输入形状 [bs, 1, dim] -> reshape -> [bs, dim] -> Dense -> [bs, filters] -> reshape -> [bs, 1, filters]
            return tf.expand_dims(self.out(tf.squeeze(x, axis=1)), axis=1)
        else:
            if hasattr(self, 'pad_layer'):
                x = self.pad_layer(x)
            return self.tcn_cell(x)

    def compute_output_shape(self, input_shape):
        bs, seq_l, dim = input_shape
        if seq_l == 1:
            return (bs, 1, self.filters)
        else:
            if seq_l % self.ks != 0:
                seq_l = seq_l + self.ks - (seq_l % self.ks)
            output_len = seq_l // self.ks
            return (bs, output_len, self.filters)

    def get_config(self):
        config = super(TCNCell, self).get_config()
        config.update({
            'filters': self.filters,
            'ks': self.ks,
            'activation': self.activation,
        })
        return config


class TCN(Layer):
    """
    input: (batch_size, seq_len, feature_dim)
    output: (batch_size, output_len, feature_dim)
    """

    def __init__(self, filters_list=[32, 64, 128], kernel_size_list=[3, 3, 3], seq_len=32, name='TCN', **kwargs):
        assert len(filters_list) == len(kernel_size_list), "filters_list and kernel_size_list must have the same length"
        self.l = len(filters_list)
        assert seq_len is not None and seq_len > 2 ** self.l, \
            f"seq_len is None or receptive field must be smaller than sequence length, please check"
        self.filters_list = filters_list
        self.kernel_size_list = kernel_size_list
        self.seql = seq_len
        self.print_receptive_field()
        super(TCN, self).__init__(name=name, **kwargs)

    def cala_receptive_field(self):
        ce_list = []
        for idx, ks in enumerate(self.kernel_size_list):
            if idx == 0:
                ce_list.append(ks)
            else:
                ce_list.append(ce_list[-1] * ks)
        return ce_list[-1]

    def print_receptive_field(self):
        ce = self.cala_receptive_field()
        print(f'当前的参数将会使感受野提升{ce}倍，即输出时间维度一个时刻能够反应其之前{ce}个时刻的特征')
        print(f'The current parameter will increase the receptive field by {ce} times, '
              f'which means that the output time dimension can reflect the features of {ce} times before it at one moment')

    def build(self, input_shape):
        bs, seql, dim = input_shape
        assert seql == self.seql, f'输入序列长度{seql}与设定的序列长度{self.seql}不一致 ' \
                                  f'The input sequence length {seql} does not match the set sequence length {self.seql}'

        self.tcn_cell_layers = []
        for i in range(self.l):
            layer = TCNCell(filters=self.filters_list[i], ks=self.kernel_size_list[i])
            # 手动构建每一层（可选，call 时会自动构建）
            layer.build((bs, seql // (self.kernel_size_list[0] ** i) if i > 0 else seql, 
                         self.filters_list[i-1] if i > 0 else dim))
            self.tcn_cell_layers.append(layer)

        self.built = True

    def call(self, x):
        for i in range(self.l):
            x = self.tcn_cell_layers[i](x)
        return x

    def compute_output_shape(self, input_shape):
        bs, seql, dim = input_shape
        for ks in self.kernel_size_list:
            if seql % ks != 0:
                seql = seql + ks - (seql % ks)
            seql = seql // ks
        return (bs, seql, self.filters_list[-1])

    def get_config(self):
        config = super(TCN, self).get_config()
        config.update({
            'filters_list': self.filters_list,
            'kernel_size_list': self.kernel_size_list,
            'seq_len': self.seql,
        })
        return config


if __name__ == '__main__':
    import numpy as np
    tcnnet = TCN()
    # 构建模型（可选，但推荐）
    tcnnet.build((1, 32, 768))
    out = tcnnet(np.zeros((1, 32, 768)))
    print("Output shape:", out.shape)  # 应该是 (1, 32/(3*3*3) 向上取整? 实际是逐层整除)