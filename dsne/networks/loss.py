from mxnet.gluon.loss import Loss


class dSNELoss(Loss):
    """
    dSNE Loss
    """
    def __init__(self, bs_src, bs_tgt, embed_size, margin=1, fn=False, weight=1, batch_axis=0, **kwargs):
        super(dSNELoss, self).__init__(weight, batch_axis, **kwargs)

        with self.name_scope():
            self._bs_src = bs_src
            self._bs_tgt = bs_tgt
            self._embed_size = embed_size
            self._margin = margin
            self._fn = fn

    def hybrid_forward(self, F, fts, ys, ftt, yt, *args, **kwargs):
        """
        :param F: Function
        :param yt: label for the target domain [N]
        :param ftt: features for the target domain [N, K]
        :param ys: label for the source domain [M]
        :param fts: features for the source domain [M, K]
        :return:
        """
        if self._fn:
            # Normalize ft
            fts = F.L2Normalization(fts, mode='instance')
            ftt = F.L2Normalization(ftt, mode='instance')

        fts_rpt = F.broadcast_to(fts.expand_dims(axis=0), shape=(self._bs_tgt, self._bs_src, self._embed_size))
        ftt_rpt = F.broadcast_to(ftt.expand_dims(axis=1), shape=(self._bs_tgt, self._bs_src, self._embed_size))

        dists = F.sum(F.square(ftt_rpt - fts_rpt), axis=2)

        yt_rpt = F.broadcast_to(yt.expand_dims(axis=1), shape=(self._bs_tgt, self._bs_src)).astype('int32')
        ys_rpt = F.broadcast_to(ys.expand_dims(axis=0), shape=(self._bs_tgt, self._bs_src)).astype('int32')

        y_same = F.equal(yt_rpt, ys_rpt).astype('float32')
        y_diff = F.not_equal(yt_rpt, ys_rpt).astype('float32')

        intra_cls_dists = dists * y_same
        inter_cls_dists = dists * y_diff

        max_dists = F.max(dists, axis=1, keepdims=True)
        max_dists = F.broadcast_to(max_dists, shape=(self._bs_tgt, self._bs_src))
        revised_inter_cls_dists = F.where(y_same, max_dists, inter_cls_dists)

        max_intra_cls_dist = F.max(intra_cls_dists, axis=1)
        min_inter_cls_dist = F.min(revised_inter_cls_dists, axis=1)

        loss = F.relu(max_intra_cls_dist - min_inter_cls_dist + self._margin)

        return loss