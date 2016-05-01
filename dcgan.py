import mxnet as mx
import numpy as np
from sklearn.datasets import fetch_mldata
import logging
import cv2

def make_dcgan_sym(ngf, ndf, nc):
    rand = mx.sym.Variable('rand')

    g1 = mx.sym.FullyConnected(rand, name='g1', num_hidden=ngf*8*4*4)
    g1 = mx.sym.Reshape(g1, target_shape=(0, ngf*8, 4, 4))
    gbn1 = mx.sym.BatchNorm(g1, name='gbn1', fix_gamma=False)
    gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')

    g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*4)
    gbn2 = mx.sym.BatchNorm(g2, name='gbn2', fix_gamma=False)
    gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')

    g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*2)
    gbn3 = mx.sym.BatchNorm(g3, name='gbn3', fix_gamma=False)
    gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')

    g4 = mx.sym.Deconvolution(gact3, name='g4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf)
    gbn4 = mx.sym.BatchNorm(g4, name='gbn4', fix_gamma=False)
    gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')

    g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=nc)
    gout = mx.sym.Activation(g5, name='gact5', act_type='tanh')

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    d1 = mx.sym.Convolution(data, name='d1', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf)
    dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

    d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*2)
    dbn2 = mx.sym.BatchNorm(d2, name='dbn2', fix_gamma=False)
    dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

    d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*4)
    dbn3 = mx.sym.BatchNorm(d3, name='dbn3', fix_gamma=False)
    dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

    d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*8)
    dbn4 = mx.sym.BatchNorm(d4, name='dbn4', fix_gamma=False)
    dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

    d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4,4), num_filter=1)
    d5 = mx.sym.Flatten(d5)

    dloss = mx.sym.LogisticRegressionOutput(data=d5, label=label, name='dloss')

    return gout, dloss

def get_mnist():
    mnist = fetch_mldata('MNIST original')
    np.random.seed(1234) # set seed for deterministic ordering
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p]
    X = X.reshape((70000, 28, 28))

    X = np.asarray([cv2.resize(x, (64,64)) for x in X])

    X = X.astype(np.float32)/(255.0/2) - 1.0
    X = X.reshape((70000, 1, 64, 64))
    X = np.tile(X, (1, 3, 1, 1))
    X_train = X[:60000]
    X_test = X[60000:]

    return X_train, X_test

class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim))]

def fill_buf(buf, i, img, shape):
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0]

    sx = (i%m)*shape[0]
    sy = (i/m)*shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img

def visual(title, X):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    buff = np.zeros((n*X.shape[1], n*X.shape[2], X.shape[3]), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    cv2.imshow(title, buff)
    cv2.waitKey(1)

def update(updater, infos, inputs):
    for i, info in enumerate(infos):
        name, weight, grad = info
        if name in inputs:
            continue
        updater(i, grad, weight)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # =============setting============
    ndf = 64
    ngf = 64
    nc = 3
    batch_size = 64
    Z = 100
    lr = 0.0002
    beta1 = 0.5
    ctx = mx.gpu(2)
    gout, dloss = make_dcgan_sym(ngf, ndf, nc)
    # mx.viz.plot_network(gout, shape={'rand': (batch_size, 100)}).view()
    # mx.viz.plot_network(gloss, shape={'data': (batch_size, nc, 64, 64)}).view()

    # ==============data==============
    X_train, X_test = get_mnist()
    train_iter = mx.io.NDArrayIter(X_train, batch_size=batch_size)
    rand_iter = RandIter(batch_size, Z)

    # ==============bind==============
    ginput_shapes = {'rand': (batch_size, Z)}
    gexec = gout.simple_bind(ctx=ctx, grad_req='write', **ginput_shapes)
    gargs = dict(zip(gout.list_arguments(), gexec.arg_arrays))
    rand = gargs['rand']

    dlinput_shapes = {'data': (batch_size,)+gexec.outputs[0].shape[1:], 'label': (batch_size,)}
    dlexec = dloss.simple_bind(ctx=ctx, grad_req='write', **dlinput_shapes)

    dlargs = dict(zip(dloss.list_arguments(), dlexec.arg_arrays))
    data = dlargs['data']
    label = dlargs['label']
    label[:] = 1

    dlgrad = dict(zip(dloss.list_arguments(), dlexec.grad_arrays))
    dldiff = dlgrad['data']

    # ===============init===============
    init = mx.init.Normal(0.02)
    for name, arr in dlargs.items() + gargs.items():
        if name not in dlinput_shapes and name not in ginput_shapes:
            init(name, arr)

    # =============optimizer============ 
    gopt = mx.optimizer.Adam(
        learning_rate=lr,
        wd=0.,
        beta1=beta1,
        rescale_grad=1.0/train_iter.batch_size,
        clip_gradient=5.0,
        sym=gout,
        param_idx2name={i:n for i, n in enumerate(gout.list_arguments())})
    dopt = mx.optimizer.Adam(
        learning_rate=lr,
        wd=0.,
        beta1=beta1,
        rescale_grad=1.0/train_iter.batch_size,
        clip_gradient=5.0,
        sym=dloss,
        param_idx2name={i:n for i, n in enumerate(dloss.list_arguments())})
    gupdater = mx.optimizer.get_updater(gopt)
    dupdater = mx.optimizer.get_updater(dopt)

    ginfo = zip(gout.list_arguments(),
                gexec.arg_arrays,
                gexec.grad_arrays)
    dlinfo = zip(dloss.list_arguments(),
                 dlexec.arg_arrays,
                 dlexec.grad_arrays)

    # ============printing==============
    def norm_stat(d):
        return mx.nd.norm(d)/np.sqrt(d.size)
    mon = mx.mon.Monitor(10, norm_stat, pattern=".*output|d1_backward_data", sort=True)
    #mon.install(gexec)
    #mon.install(dlexec)
    mon = None

    def fmetric(label, pred):
        return np.sum((pred.squeeze() > 0.5) == label)*1.0/label.size

    # =============train===============
    for epoch in range(100):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            rbatch = rand_iter.next()
            rand[:] = rbatch.data[0]

            if mon is not None:
                mon.tic()

            gexec.forward(is_train=True)

            # update generator
            data[:] = gexec.outputs[0]
            label[:] = 1
            dlexec.forward(is_train=True)
            dlexec.backward()
            gexec.backward(dldiff)
            update(gupdater, ginfo, ginput_shapes)

            err_g = -np.log(dlexec.outputs[0].asnumpy()).mean()

            # update discriminator on fake
            label[:] = 0
            dlexec.forward(is_train=True)
            dlexec.backward()
            update(dupdater, dlinfo, dlinput_shapes)

            acc = fmetric(label.asnumpy(), dlexec.outputs[0].asnumpy())
            err_d = -np.log(1. - dlexec.outputs[0].asnumpy()).mean()

            # update discriminator on real
            label[:] = 1
            data[:] = batch.data[0]
            dlexec.forward(is_train=True)
            dlexec.backward()
            update(dupdater, dlinfo, dlinput_shapes)

            acc += fmetric(label.asnumpy(), dlexec.outputs[0].asnumpy())
            err_d += -np.log(dlexec.outputs[0].asnumpy()).mean()

            acc /= 2
            err_d /= 2

            if mon is not None:
                mon.toc_print()

            t += 1
            if t % 10 == 0:
                print 'epoch:', epoch, 'iter:', t, 'metric:', acc, err_g, err_d
                #print dlexec.outputs[0].asnumpy().squeeze()
                visual('gout', gexec.outputs[0].asnumpy())
                #visual('data', batch.data[0].asnumpy())
                diff = dldiff.asnumpy()
                diff = (diff - diff.mean())/diff.std()
                visual('diff', diff)



