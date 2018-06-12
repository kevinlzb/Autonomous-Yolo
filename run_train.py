import mxnet as mx
from symbol import get_resnet_model
import numpy as np
import time
from tools.logging_metric import LogMetricsCallback, LossMetric

import logging
import sys

root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.DEBUG)


IMGROOT = "Data_txt/Data_rec/train_dataset"

def get_iterator(path,data_shape, label_width,batch_size, shuffle=False):
    iterator = mx.io.ImageRecordIter(path_imgrec=path,data_shape=data_shape,
                                     label_width=label_width,batch_size=batch_size,shuffle=shuffle)
    return iterator

if __name__ == "__main__":
    anchors = np.asarray([[0.163, 0.07], [0.46, 0.21], [0.29, 0.13], [0.69, 0.33], [0.06, 0.04]])
    sym = get_resnet_model('pretrained_models/resnet-34',0, anchors.tolist())
    _,arg_params,aux_params = mx.model.load_checkpoint('pretrained_models/resnet-34',0)

    train_data = get_iterator(path="Data_txt/Data_rec/train_dataset.rec",data_shape=(3,416,416),label_width=13*13*9,batch_size=32,shuffle=True)
    val_data = get_iterator(path="Data_txt/Data_rec/val_dataset.rec",data_shape=(3,416,416),label_width=13*13*9,batch_size=32)

    #allocate gpu/cpu mem to the sym
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(0))
    metric = mx.metric.create(LossMetric, allow_extra_outputs=True)
    tme =time.time()
    logtrain = LogMetricsCallback('logs/train_' + str(tme))

    #setup monitor for debugging
    def norm_stat(d):
        return mx.nd.norm(d) / np.sqrt(d.size)
    mon = None

    #save model
    checkpoint = mx.callback.do_checkpoint('drive_full_detect')

    #train
    mod.fit(train_data=train_data,
            eval_data=val_data,
            num_epoch=500,
            monitor=mon,
            eval_metric=LossMetric(0.5),
            optimizer='rmsprop',
            optimizer_params={'learning_rate':0.01, 'lr_scheduler': mx.lr_scheduler.FactorScheduler(300000, 0.1, 0.001)},
            initializer=mx.init.Xavier(magnitude=2, rnd_type='gaussian', factor_type='in'),
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=True,
            batch_end_callback=[mx.callback.Speedometer(batch_size=32, frequent=10, auto_reset=False), logtrain],
            # batch_end_callback=[mx.callback.Speedometer(batch_size=32, frequent=10, auto_reset=False), metric],
            epoch_end_callback=checkpoint,
            )


