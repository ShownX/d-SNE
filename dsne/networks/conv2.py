from mxnet import gluon


class Conv2(gluon.nn.HybridBlock):
    """
    LeNetPlus model
    """
    def __init__(self, classes=10, feature_size=256, use_bn=False, use_l2n=False, dropout=0.5, **kwargs):
        super(Conv2, self).__init__(**kwargs)
        with self.name_scope():

            self.use_bn = use_bn
            self.use_l2n = use_l2n

            self.features = gluon.nn.HybridSequential(prefix='')

            self.features.add(gluon.nn.Conv2D(6, 5, activation='relu'))
            if dropout > 0:
                self.features.add(gluon.nn.Dropout(0.5))
            self.features.add(gluon.nn.Conv2D(16, 5, activation='relu'))
            self.features.add(gluon.nn.MaxPool2D())
            if dropout > 0:
                self.features.add(gluon.nn.Dropout(0.5))

            self.features.add(gluon.nn.Dense(feature_size))

            self.output = gluon.nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        features = self.features(x)
        if self.use_l2n:
            features = F.L2Normalization(features, mode='instance')

        outputs = self.output(features)
        return outputs, features
