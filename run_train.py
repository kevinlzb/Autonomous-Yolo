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
    anchors = np.asarray([[0.092, 0.127], [0.2595, 0.376], [0.16604, 0.2258], [0.038, 0.064], [0.37, 0.59]])
#    sym = get_resnet_model('Resnet-34/resnet-34',0)
    sym = get_resnet_model('pretrained_models/resnet-34', 0)
    _, args_params, aux_params = mx.model.load_checkpoint('pretrained_models/resnet-34', 0)
#    _,arg_params,aux_params = mx.model.load_checkpoint('Resnet-34/resnet-34',0)

    train_data = get_iterator(path="Data_txt/Data_rec/trainRoadImages.rec",data_shape=(3,224,224),label_width=7*7*9,batch_size=32,shuffle=True)
    val_data = get_iterator(path="Data_txt/Data_rec/valRoadImages.rec",data_shape=(3,224,224),label_width=7*7*9,batch_size=3)

    #allocate gpu/cpu mem to the sym
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
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
            num_epoch=30,
            monitor=mon,
            eval_metric=LossMetric(0.5),
            optimizer='rmsprop',
            optimizer_params={'learning_rate':0.01, 'lr_scheduler': mx.lr_scheduler.FactorScheduler(300000, 0.1, 0.001)},
            initializer=mx.init.Xavier(magnitude=2, rnd_type='gaussian', factor_type='in'),
            arg_params=args_params,
            aux_params=aux_params,
            allow_missing=True,
            batch_end_callback=[mx.callback.Speedometer(batch_size=32, frequent=10, auto_reset=False), logtrain],
            # batch_end_callback=[mx.callback.Speedometer(batch_size=32, frequent=10, auto_reset=False), metric],
            epoch_end_callback=checkpoint,
            )


