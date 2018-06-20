import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import mxnet as mx
import sys
import cv2

def decodeBox(yolobox, size, dscale):
    i, j, cx, cy, w, h, cls1, cls2, cls3, cls4 = yolobox
    cxt = j * dscale + cx * dscale
    cyt = i * dscale + cy * dscale
    wt = w * size
    ht = h * size
    cls = np.argmax([cls1, cls2, cls3, cls4])
    return [cxt, cyt, wt, ht, cls]


def bboxdraw(img, label, dscale=32):
    assert label.shape == (7, 7, 9)
    size = img.shape[1]
    ilist, jlist = np.where(label[:, :, 0] > 0.2)

#    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(np.uint8(img))
    rect = none
    for i, j in zip(ilist, jlist):
        cx, cy, w, h, cls1, cls2, cls3, cls4 = label[i, j, 1:]
        cxt, cyt, wt, ht, cls = decodeBox([i, j, cx, cy, w, h, cls1, cls2, cls3, cls4], size, dscale)
        # Create a Rectangle patch
        if cls == 0:
            rect = patches.Rectangle((cxt - wt / 2, cyt - ht / 2), wt, ht, linewidth=2, edgecolor='b', facecolor='none')
        elif cls == 1:
            rect = patches.Rectangle((cxt - wt / 2, cyt - ht / 2), wt, ht, linewidth=2, edgecolor='g', facecolor='none')
        elif cls == 2:
            rect = patches.Rectangle((cxt - wt / 2, cyt - ht / 2), wt, ht, linewidth=2, edgecolor='y', facecolor='none')
        elif cls == 3:
            rect = patches.Rectangle((cxt - wt / 2, cyt - ht / 2), wt, ht, linewidth=2, edgecolor='r', facecolor='none')
#        rect = patches.Rectangle((cxt - wt / 2, cyt - ht / 2), wt, ht, linewidth=1, edgecolor='r', facecolor='none')

#        # Add the patch to the Axes
        ax.add_patch(rect)

        name = "unkown"
        if cls == 0:
            name = "car"
        elif cls == 1:
            name = "pedestrian"
        elif cls == 2:
            name = "cyclist"
        elif cls == 3:
            name = "traffic lights"
        
        plt.text(x=int(cxt - wt / 2), y=int(cyt - ht / 2), s=str(name), bbox=dict(facecolor='red', alpha=0.5))
    plt.show()

if __name__ == "__main__":
    print(sys.executable)
    print(sys.version)
    print(sys.version_info)
    data = mx.io.ImageRecordIter(path_imgrec='Data_txt/DATA_rec/valRoadImages.rec',
                                 data_shape=(3, 224, 224),
                                 label_width=7 * 7 * 9,
                                 batch_size=1, )
    # get sym
    sym, args_params, aux_params = mx.model.load_checkpoint('drive_full_detect', 600)
    logit = sym.get_internals()['logit_output']
    mod = mx.mod.Module(symbol=logit, context=mx.cpu(0))
    mod.bind(data.provide_data)
    mod.init_params(allow_missing=False, arg_params=args_params, aux_params=aux_params,
                    initializer=mx.init.Xavier(magnitude=2, rnd_type='gaussian', factor_type='in'))
    out = mod.predict(eval_data=data, num_batch=10)

    data.reset()
    for i in range(10):
        batch = data.next()
        img = batch.data[0].asnumpy()[0].transpose((1, 2, 0))
        label = batch.label[0].asnumpy().reshape((7, 7, 9))
        pred = (out.asnumpy()[i] + 1) / 2
        print pred.shape
        print ("Prediction")
        bboxdraw(img, pred)
        print ("Ground Truth")
#        print (label)
        bboxdraw(img, label)
