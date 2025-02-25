import logging
import numpy as np
import mxnet as mx

class LogMetricsCallback(object):
    def __init__(self, logging_dir, prefix=None):
        self.prefix = prefix
        self.itr = 0
        try:
            from tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter(logging_dir)
        except ImportError:
            logging.error('You can install tensorboard via `pip install tensorboard`.')

    def __call__(self, param):
        """Callback to log training speed and metrics in TensorBoard."""
        if param.eval_metric is None:
            return
        name_value = param.eval_metric.get_name_value()
        for name, value in name_value:
            if self.prefix is not None:
                name = '%s-%s' % (self.prefix, name)
            self.summary_writer.add_scalar(name, value, self.itr)
        self.itr += 1



class LossMetric(mx.metric.EvalMetric):
    """
    Calculate precision and recall for bounding box detection

    Parameters
    ----------
    threshold : float
    """

    def __init__(self, conf_threshold=0.85, eps=1e-5, allow_extra_outputs=True, grid=169, stride=32, W=416, H=416):
        self.eps = eps
        super(LossMetric, self).__init__('LossMetric', allow_extra_outputs=allow_extra_outputs)
        self.conf_threshold = conf_threshold
        self.stride =stride
        self.grid =grid
        self.W = W
        self.H = H
    def reset(self):
        """Clear the internal statistics to initial state."""
        self.num_inst = self.eps
        self.sum_tp = self.eps
        self.sum_tn = self.eps
        self.sum_fn = self.eps
        self.sum_fp = self.eps
        self.sum_conf = self.eps
        self.sum_x = self.eps
        self.sum_y = self.eps
        self.sum_h = self.eps
        self.sum_w = self.eps
        self.sum_cls = self.eps

    def update(self, labels, preds):
        """Updates the internal evaluation result.

            Parameters
            ----------
            labels : list of `NDArray`
                The labels of the data.
            preds : list of `NDArray`
                Predicted values.
            """
        # print "a new epoch"

        
        self.sum_loss = np.mean(preds[0].asnumpy())



        label = labels[0].asnumpy().reshape((-1, 169, 9))
        pred = ((preds[1] + 1) / 2).asnumpy().reshape((-1, 169,5,9))




        pred_adjust = np.zeros((32,169,9))
        for i in range(32):
            for j in range(169):
                index = 0
                val = 0
                for k in range(5) :
                    if pred[i,j,k,0] > val:
                        val = pred[i,j,k,0]
                        index = k
                pred_adjust[i,j,:] = pred[i,j,index,:]




        pred = pred_adjust
        # print pred.shape
        # print label.shape
        # conf = pred[:,:,:,0]x
        # print conf.shape
        # index = np.argmax(conf,axis=2)
        # print index
        c_label = label[:, :, 0]
        c_pred = pred[:, :,0]
        # print c_label.shape
        # print c_pred.shape
        boxes_pred = c_pred > self.conf_threshold
        # print boxes_pred.shape
        self.sum_tp = np.sum(c_label * boxes_pred)
        self.sum_tn = np.sum((1 - c_label) * (1 - boxes_pred))
        self.sum_fn = np.sum(c_label * (1 - boxes_pred))
        self.sum_fp = np.sum(boxes_pred * (1 - c_label))


        num_boxes = np.sum(c_label)
        self.sum_conf = np.sum(np.abs(c_pred - c_label)) / \
                        (self.grid * label.shape[0])
        self.sum_x = np.sum((np.abs(pred[:, :, 1] - label[:, :,1])) * c_label) * self.stride / num_boxes
        self.sum_y = np.sum((np.abs(pred[:, :, 2] - label[:, :,2])) * c_label) * self.stride / num_boxes
        self.sum_w = np.sum((np.abs(pred[:, :, 3] - label[:, :, 3])) * c_label) \
                     * self.W / num_boxes
        self.sum_h = np.sum((np.abs(pred[:, :,4] - label[:, :,4])) * c_label) \
                     * self.H / num_boxes
        self.sum_cls = np.mean(np.abs(pred[:, :, 5:] - label[:, :, 5:]) * c_label[:, :, np.newaxis])


        print "Accuracy is {}".format((self.sum_tp + self.sum_tn) / (
            self.sum_tp + self.sum_tn + self.sum_fp + self.sum_fn))

        print "Reall is {}".format(self.sum_tp / (self.sum_tp + self.sum_fn + 1e-6))
        print "Precission is {}".format(self.sum_tp / (self.sum_tp + self.sum_fp + 1e-6))
        print "Number of FN is {}".format(np.sum(c_label * (1 - boxes_pred)))
        print "Number of FP is {}".format(np.sum(boxes_pred * (1 - c_label)))


    def get(self):
        """Gets the current evaluation result.

      Returns
      -------
      names : list of str
         Name of the metrics.
      values : list of float
         Value of the evaluations.
      """
        names = ['c_accuracy', 'c_precision', 'c_recall', 'c_diff', 'x_diff', 'y_diff',
                 'w_diff', 'h_diff', 'loss', 'cls_diff']

        values = []
        values.append((self.sum_tp + self.sum_tn) / (
            self.sum_tp + self.sum_tn + self.sum_fp + self.sum_fn))
        values.append(self.sum_tp / (self.sum_tp + self.sum_fp + 1e-6))
        values.append(self.sum_tp / (self.sum_tp + self.sum_fn + 1e-6))
        values.extend([sum_val for sum_val in
                       (self.sum_conf, self.sum_x, self.sum_y, self.sum_w,
                        self.sum_h, self.sum_loss)])
        # self.sum_cls

        return names, values
