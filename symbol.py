import mxnet as mx
import numpy as np


@mx.init.register
class MyConstant(mx.init.Initializer):
    def __init__(self, value):
        super(MyConstant, self).__init__(value=value)
        self.value = value

    def _init_weight(self, _, arr):
        arr[:] = mx.nd.array(self.value)



def expit_tensor(x):
    return 1 / (1 + mx.sym.exp(-x))

def loss_Yolov2(label, pred,anchors):
    sprob = 1
    snoob = 0.5
    scoor = 5
    size_H, size_W = (13,13)
    B = 5
    HW = size_H * size_W

    #Read anchors, they should be  0<w,h<1
    anchors_w,anchors_h = mx.sym.split(anchors, axis=1, num_outputs=2, name="anchor_split")

    #Read label
    label_reshape = mx.sym.reshape(label,[-1,size_H,size_W,9])
    anchors = mx.sym.reshape(label,[-1,size_H,size_W,9])
    prob_l, x_l, y_l, w_l, h_l, cls1,cls2,cls3,cls4 = mx.sym.split(label_reshape, axis=3, num_outputs=9, name="label_split")

    upperleft_x_l = x_l - w_l * size_W * 0.5
    upperleft_y_l = y_l - h_l * size_H * 0.5
    bottomright_x_l = x_l + w_l * size_W * 0.5
    bottomright_y_l = y_l + h_l * size_H * 0.5
    # upperleft_x_l = -w_l * size_W * 0.5
    # upperleft_y_l = -h_l * size_W * 0.5
    # bottomright_x_l = w_l * size_W * 0.5
    # bottomright_y_l = h_l * size_H * 0.5
    area_l = (w_l * h_l) * (size_W * size_H)

    pred_reshape = mx.sym.reshape(pred,[-1,size_H,size_W,B,9])
    prob_p, x_p, y_p, wr_p, hr_p,cls1,cls2,cls3,cls4 = mx.sym.split(pred_reshape,axis=4,num_outputs=9, name="pred_split")
    x_adjust = expit_tensor(x_p) - 0.5
    y_adjust = expit_tensor(y_p) - 0.5
    
    w_adjust = mx.sym.sqrt(mx.sym.broadcast_mul(mx.sym.exp(wr_p),mx.sym.reshape(anchors_w,shape=[1,1,B,1])))
    h_adjust = mx.sym.sqrt(mx.sym.broadcast_mul(mx.sym.exp(hr_p),mx.sym.reshape(anchors_h,shape=[1,1,B,1])))
    
    prob_adjust = expit_tensor(prob_p)
    cls1p = expit_tensor(cls1)
    cls2p = expit_tensor(cls2)
    cls3p = expit_tensor(cls3)
    cls4p = expit_tensor(cls4)


    # get predict upperleft and bottomright
    w_p = w_adjust ** 2
    h_p = h_adjust ** 2
    upperleft_x_p = x_p - w_p * size_W * 0.5
    upperleft_y_p = y_p - h_p * size_H * 0.5
    bottomright_x_p = x_p + w_p * size_W * 0.5
    bottomright_y_p = y_p + h_p * size_H * 0.5
    area_pred = (w_p * h_p) * (size_W * size_H)

    # caculate IOU
    intersect_upleft_x = mx.sym.broadcast_maximum(upperleft_x_p, mx.sym.expand_dims(upperleft_x_l, axis=-1))
    intersect_upleft_y = mx.sym.broadcast_maximum(upperleft_y_p, mx.sym.expand_dims(upperleft_y_l, axis=-1))
    intersect_botright_x = mx.sym.broadcast_minimum(bottomright_x_p, mx.sym.expand_dims(bottomright_x_l,axis=-1))
    intersect_botright_y = mx.sym.broadcast_minimum(bottomright_y_p, mx.sym.expand_dims(bottomright_y_l, axis=-1))
    intersect_w = mx.sym.maximum(intersect_botright_x - intersect_upleft_x,0)
    intersect_h = mx.sym.maximum(intersect_botright_y - intersect_upleft_y,0)
    intersect = intersect_h * intersect_w

    iou = intersect / mx.sym.broadcast_add(area_pred - intersect, mx.sym.expand_dims(area_l, axis=3))
    best_box = mx.sym.broadcast_equal(iou, mx.sym.max_axis(iou,axis=3,keepdims=True)) * 1.0
    prob_anchor_l = mx.sym.broadcast_mul(best_box, mx.sym.expand_dims(prob_l,axis=-1))
    x_anchor_l = mx.sym.broadcast_mul(best_box, mx.sym.expand_dims(x_l,axis=-1), name="x_anchor_l")
    y_anchor_l = mx.sym.broadcast_mul(best_box, mx.sym.expand_dims(y_l, axis=-1), name="y_anchor_l")
    w_anchor_l = mx.sym.broadcast_mul(best_box, mx.sym.expand_dims(w_l, axis=-1), name="w_anchor_l")
    h_anchor_l = mx.sym.broadcast_mul(best_box, mx.sym.expand_dims(h_l, axis=-1), name="h_anchor_l")

    mask = (prob_anchor_l * 5 + (1 - prob_anchor_l) * 0.5)

    loss_prob = mx.sym.LinearRegressionOutput(data=prob_adjust * mask, label=prob_anchor_l * mask, grad_scale=1,name="lossprob")
    loss_x = mx.sym.LinearRegressionOutput(data=mx.sym.broadcast_mul(x_adjust,best_box),label=x_anchor_l,grad_scale=scoor,name="lossx")
    loss_y = mx.sym.LinearRegressionOutput(data=y_adjust,label=y_anchor_l,grad_scale=scoor,name="lossy")
    loss_w = mx.sym.LinearRegressionOutput(data=w_adjust,label=w_anchor_l,grad_scale=scoor,name="lossw")
    loss_h = mx.sym.LinearRegressionOutput(data=h_adjust,label=h_anchor_l,grad_scale=scoor,name="lossh")

    loss_cls1 = mx.sym.LinearRegressionOutput(data=cls1p,label=cls1,grad_scale=scoor,name="losscls1")
    loss_cls2 = mx.sym.LinearRegressionOutput(data=cls2p,label=cls2,grad_scale=scoor,name="losscls2")
    loss_cls3 = mx.sym.LinearRegressionOutput(data=cls3p,label=cls3,grad_scale=scoor,name="losscls3")
    loss_cls4 = mx.sym.LinearRegressionOutput(data=cls4p,label=cls4,grad_scale=scoor,name="losscls4")

    loss = loss_prob + loss_x + loss_y + loss_w + loss_h + loss_cls1 + loss_cls2 + loss_cls3 + loss_cls4
    return loss



# Get pretrained imagenet model
def get_resnet_model(model_path,epoch,constr_arr):

    label = mx.symbol.Variable('softmax_label')

    anchors = mx.symbol.Variable('anchors', shape=(5,2), init= MyConstant(value = constr_arr), dtype=np.float32)
    anchors = mx.sym.BlockGrad(anchors)

    sym,args,aux = mx.model.load_checkpoint(model_path,epoch)
    #extract the last bn layer
    sym = sym.get_internals()['bn1_output']

    #append two layers
    sym = mx.sym.Activation(data=sym, act_type="relu", name="relu_final")
    sym = mx.sym.Convolution(data=sym, kernel=(3,3),
                             num_filter = 45, pad=(1,1),
                             stride = (1,1), no_bias=True,
                             )
    #get softsign
    sym = sym / (1 + mx.sym.abs(sym))
    logit = mx.sym.transpose(sym, axes= (0,2,3,1), name = "logit")

    #apply loss
    loss_ = loss_Yolov2(label,logit,anchors);
    loss = mx.sym.MakeLoss(loss_)
    out = mx.sym.Group([loss,mx.sym.BlockGrad(logit)])
    return out
