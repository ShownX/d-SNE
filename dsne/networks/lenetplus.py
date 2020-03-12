from mxnet import gluon


def _make_conv_block(block_index, num_chan=32, num_layer=2, stride=1, pad=2):
    out = gluon.nn.HybridSequential(prefix='block_%d_' % block_index)
    with out.name_scope():
        for _ in range(num_layer):
            out.add(gluon.nn.Conv2D(num_chan, kernel_size=3, strides=stride, padding=pad))
            out.add(gluon.nn.LeakyReLU(alpha=0.2))
        out.add(gluon.nn.MaxPool2D())

    return out


class LeNetPlus(gluon.nn.HybridBlock):
    """
    LeNetPlus model
    """
    def __init__(self, classes=10, feature_size=256, use_bn=False, use_inn=False, use_l2n=False, **kwargs):
        super(LeNetPlus, self).__init__(**kwargs)
        num_chans = [32, 64, 128]
        with self.name_scope():
            self.use_bn = use_bn
            self.use_inn = use_inn
            self.use_l2n = use_l2n

            self.features = gluon.nn.HybridSequential(prefix='')

            if self.use_inn:
                self.features.add(gluon.nn.InstanceNorm())

            for i, num_chan in enumerate(num_chans):
                if use_bn:
                    self.features.add(gluon.nn.BatchNorm())

                self.features.add(_make_conv_block(i, num_chan=num_chan))

            self.features.add(gluon.nn.Dense(feature_size))

            self.output = gluon.nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        features = self.features(x)

        if self.use_l2n:
            features = F.L2Normalization(features, mode='instance', name='l2n')

        outputs = self.output(features)
        return outputs, features
