def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(
        n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))








class mIoU:
    def __init__(self):
        self.pred_list = []
        self.gt_list = []
        self.metric_name = type(self).__name__
        self.score_dict = {"metric_name": self.metric_name}
        self.hist = None

    def addBatch(self, model, batch, **options):
        pred_labels = ms.t2n(model.predict(batch, method="maskClasses"))
        gt_labels = ms.t2n(batch["maskClasses"]).squeeze()

        if gt_labels.shape[-1] != pred_labels.shape[-1]:
            pred_labels = ms.t2n(
                F.interpolate(
                    torch.FloatTensor(pred_labels[None, None]),
                    size=gt_labels.shape,
                    mode='bilinear',
                    align_corners=True)).squeeze()

        if self.hist is None:
            self.hist = np.zeros((model.n_classes, model.n_classes))
        self.hist += fast_hist(gt_labels.flatten().astype(int),
                               pred_labels.flatten().astype(int),
                               model.n_classes)

    def compute_score_dict(self):
        mIoUs = per_class_iu(self.hist)

        self.score_dict["score"] = np.nanmean(mIoUs)
        self.score_dict["per_class"] = mIoUs.tolist()

        return self.score_dict

    def is_best_score_dict(self, best_score_dict):
        best_flag = False
        best_score = best_score_dict["score"]
        curr_score = self.score_dict["score"]
        if best_score <= curr_score or best_score == -1:
            print("New best model: "
                  "%.3f=>%.3f %s" % (best_score, curr_score, self.metric_name))
            best_flag = True

        self.score_dict["best_flag"] = best_flag

        return best_flag