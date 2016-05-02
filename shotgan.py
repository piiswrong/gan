import mxnet as mx
import numpy as np
from sklearn.datasets import fetch_mldata
import logging
import cv2

def make_torch_sym():
    rand = mx.sym.Variable('rand')

    g1 = mx.sym.TorchModule(rand, lua_string='nn.SpatialFullConvolution(100, 64 * 8, 4, 4)', name='g1', num_data=1, num_params=2, num_outputs=1)
    gbn1 = mx.sym.TorchModule(data_0=g1, weight=mx.sym.Variable('gbn1_gamma'), bias=mx.sym.Variable('gbn1_beta'), lua_string='nn.SpatialBatchNormalization(64 * 8)', name='gbn1', num_data=1, num_params=2, num_outputs=1)
    gact1 = mx.sym.TorchModule(gbn1, lua_string='nn.ReLU(false)', name='gact1', num_data=1, num_params=0, num_outputs=1)

    g2 = mx.sym.TorchModule(gact1, lua_string='nn.SpatialFullConvolution(64 * 8, 64 * 4, 4, 4, 2, 2, 1, 1)', name='g2', num_data=1, num_params=2, num_outputs=1)
    gbn2 = mx.sym.TorchModule(data_0=g2, weight=mx.sym.Variable('gbn2_gamma'), bias=mx.sym.Variable('gbn2_beta'), lua_string='nn.SpatialBatchNormalization(64 * 4)', name='gbn2', num_data=1, num_params=2, num_outputs=1)
    gact2 = mx.sym.TorchModule(gbn2, lua_string='nn.ReLU(false)', name='gact2', num_data=1, num_params=0, num_outputs=1)

    g3 = mx.sym.TorchModule(gact2, lua_string='nn.SpatialFullConvolution(64 * 4, 64 * 2, 4, 4, 2, 2, 1, 1)', name='g3', num_data=1, num_params=2, num_outputs=1)
    gbn3 = mx.sym.TorchModule(data_0=g3, weight=mx.sym.Variable('gbn3_gamma'), bias=mx.sym.Variable('gbn3_beta'), lua_string='nn.SpatialBatchNormalization(64 * 2)', name='gbn3', num_data=1, num_params=2, num_outputs=1)
    gact3 = mx.sym.TorchModule(gbn3, lua_string='nn.ReLU(false)', name='gact3', num_data=1, num_params=0, num_outputs=1)

    g4 = mx.sym.TorchModule(gact3, lua_string='nn.SpatialFullConvolution(64 * 2, 64, 4, 4, 2, 2, 1, 1)', name='g4', num_data=1, num_params=2, num_outputs=1)
    gbn4 = mx.sym.TorchModule(data_0=g4, weight=mx.sym.Variable('gbn4_gamma'), bias=mx.sym.Variable('gbn4_beta'), lua_string='nn.SpatialBatchNormalization(64)', name='gbn4', num_data=1, num_params=2, num_outputs=1)
    gact4 = mx.sym.TorchModule(gbn4, lua_string='nn.ReLU(false)', name='gact4', num_data=1, num_params=0, num_outputs=1)

    g5 = mx.sym.TorchModule(gact4, lua_string='nn.SpatialFullConvolution(64, 3, 4, 4, 2, 2, 1, 1)', name='g5', num_data=1, num_params=2, num_outputs=1)
    gout = mx.sym.TorchModule(g5, lua_string='nn.Tanh()', name='gact5', num_data=1, num_params=0, num_outputs=1)

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    d1 = mx.sym.TorchModule(data, lua_string='nn.SpatialConvolution(3, 64, 4, 4, 2, 2, 1, 1)', name='d1', num_data=1, num_params=2, num_outputs=1)
    dact1 = mx.sym.TorchModule(d1, lua_string='nn.LeakyReLU(0.2, false)', name='dact1', num_data=1, num_params=0, num_outputs=1)

    d2 = mx.sym.TorchModule(dact1, lua_string='nn.SpatialConvolution(64, 64 * 2, 4, 4, 2, 2, 1, 1)', name='d2', num_data=1, num_params=2, num_outputs=1)
    dbn2 = mx.sym.TorchModule(data_0=d2, weight=mx.sym.Variable('dbn2_gamma'), bias=mx.sym.Variable('dbn2_beta'), lua_string='nn.SpatialBatchNormalization(64 * 2)', name='dbn2', num_data=1, num_params=2, num_outputs=1)
    dact2 = mx.sym.TorchModule(dbn2, lua_string='nn.LeakyReLU(0.2, false)', num_data=1, num_params=0, num_outputs=1)

    d3 = mx.sym.TorchModule(dact2, lua_string='nn.SpatialConvolution(64 * 2, 64 * 4, 4, 4, 2, 2, 1, 1)', name='d3', num_data=1, num_params=2, num_outputs=1)
    dbn3 = mx.sym.TorchModule(data_0=d3, weight=mx.sym.Variable('dbn3_gamma'), bias=mx.sym.Variable('dbn3_beta'), lua_string='nn.SpatialBatchNormalization(64 * 4)', name='dbn3', num_data=1, num_params=2, num_outputs=1)
    dact3 = mx.sym.TorchModule(dbn3, lua_string='nn.LeakyReLU(0.2, false)', name='dact3', num_data=1, num_params=0, num_outputs=1)

    d4 = mx.sym.TorchModule(dact3, lua_string='nn.SpatialConvolution(64 * 4, 64 * 8, 4, 4, 2, 2, 1, 1)', name='d4', num_data=1, num_params=2, num_outputs=1)
    dbn4 = mx.sym.TorchModule(data_0=d4, weight=mx.sym.Variable('dbn4_gamma'), bias=mx.sym.Variable('dbn4_beta'), lua_string='nn.SpatialBatchNormalization(64 * 8)', name='dbn4', num_data=1, num_params=2, num_outputs=1)
    dact4 = mx.sym.TorchModule(dbn4, lua_string='nn.LeakyReLU(0.2, false)', name='dact4', num_data=1, num_params=0, num_outputs=1)

    d5 = mx.sym.TorchModule(dact4, lua_string='nn.SpatialConvolution(64 * 8, 1, 4, 4)', name='d5', num_data=1, num_params=2, num_outputs=1)
    dact5 = d5#mx.sym.TorchModule(d5, lua_string='nn.Sigmoid()', name='dact5', num_data=1, num_params=0, num_outputs=1)
    dact5 = mx.sym.Flatten(dact5)

    #dloss = mx.sym.TorchCriterion(data=dact5, label=label, lua_string='nn.BCECriterion()', name='dloss')
    dloss = mx.sym.LogisticRegressionOutput(data=dact5, label=label, name='dloss')

    return gout, dloss

def make_dcgan_sym(ngf, ndf, nc):
    rand = mx.sym.Variable('rand')

    g1 = mx.sym.Deconvolution(rand, name='g1', kernel=(4,4), num_filter=ngf*8, no_bias=True)
    gbn1 = mx.sym.BatchNorm(g1, name='gbn1', fix_gamma=False)
    gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')

    g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*4, no_bias=True)
    gbn2 = mx.sym.BatchNorm(g2, name='gbn2', fix_gamma=False)
    gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')

    g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*2, no_bias=True)
    gbn3 = mx.sym.BatchNorm(g3, name='gbn3', fix_gamma=False)
    gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')

    g4 = mx.sym.Deconvolution(gact3, name='g4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf, no_bias=True)
    gbn4 = mx.sym.BatchNorm(g4, name='gbn4', fix_gamma=False)
    gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')

    g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=nc, no_bias=True)
    gout = mx.sym.Activation(g5, name='gact5', act_type='tanh')

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    d1 = mx.sym.Convolution(data, name='d1', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf, no_bias=True)
    dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

    d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*2, no_bias=True)
    dbn2 = mx.sym.BatchNorm(d2, name='dbn2', fix_gamma=False)
    dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

    d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*4, no_bias=True)
    dbn3 = mx.sym.BatchNorm(d3, name='dbn3', fix_gamma=False)
    dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

    d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*8, no_bias=True)
    dbn4 = mx.sym.BatchNorm(d4, name='dbn4', fix_gamma=False)
    dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

    d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4,4), num_filter=1, no_bias=True)
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
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]

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


@register
class Adam(Optimizer):
    """Adam optimizer as described in [King2014]_.

    .. [King2014] Diederik Kingma, Jimmy Ba,
       *Adam: A Method for Stochastic Optimization*,
       http://arxiv.org/abs/1412.6980

    the code in this class was adapted from
    https://github.com/mila-udem/blocks/blob/master/blocks/algorithms/__init__.py#L765

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.002.
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
        Default value is set to 0.9.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
        Default value is set to 0.999.
    epsilon : float, optional
        Default value is set to 1e-8.
    decay_factor : float, optional
        Default value is set to 1 - 1e-8.

    wd : float, optional
        L2 regularization coefficient add to all the weights
    rescale_grad : float, optional
        rescaling factor of gradient.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 decay_factor=(1 - 1e-8), **kwargs):
        super(Adam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay_factor = decay_factor

    def create_state(self, index, weight):
        """Create additional optimizer state: mean, variance

        Parameters
        ----------
        weight : NDArray
            The weight data

        """
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        lr = self._get_lr(index)
        self._update_count(index)

        t = self._index_update_count[index]
        mean, variance = state

        grad *= self.rescale_grad
        if self.clip_gradient is not None:
            clip(grad, -self.clip_gradient, self.clip_gradient, out=grad)

        mean[:] = self.beta1 * mean + (1. - self.beta1) * grad
        variance[:] = self.beta2 * variance + (1. - self.beta2) * grad * grad

        coef1 = 1. - self.beta1**t
        coef2 = 1. - self.beta2**t
        lr *= math.sqrt(coef2)/coef1

        weight[:] -= lr*mean/(sqrt(variance) + self.epsilon)

        wd = self._get_wd(index)
        if wd > 0.:
            weight[:] -= (lr * wd) * weight


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
    ctx = mx.gpu(0)
    #symG, symD = make_torch_sym()
    symG, symD = make_dcgan_sym(ngf, ndf, nc)
    #mx.viz.plot_network(symG, shape={'rand': (batch_size, 100, 1, 1)}).view()
    #mx.viz.plot_network(symD, shape={'data': (batch_size, nc, 64, 64)}).view()

    # ==============data==============
    X_train, X_test = get_mnist()
    train_iter = mx.io.NDArrayIter(X_train, batch_size=batch_size)
    rand_iter = RandIter(batch_size, Z)
    label = mx.nd.zeros((batch_size,), ctx=ctx)

    # =============module G=============
    modG = mx.mod.Module(symbol=symG, data_names=('rand',), label_names=None, context=ctx)
    modG.bind(data_shapes=rand_iter.provide_data)
    modG.init_params(initializer=mx.init.Normal(0.02))
    modG.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
        })

    # =============module D=============
    modD = mx.mod.Module(symbol=symD, data_names=('data',), label_names=('label',), context=ctx)
    modD.bind(data_shapes=train_iter.provide_data,
              label_shapes=[('label', (batch_size,))],
              inputs_need_grad=True)
    modD.init_params(initializer=mx.init.Normal(0.02))
    modD.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
        })

    mods = [modG, modD]

    # ============printing==============
    def norm_stat(d):
        return mx.nd.norm(d)/np.sqrt(d.size)
    mon = mx.mon.Monitor(10, norm_stat, pattern=".*output|d1_backward_data", sort=True)
    mon = None
    if mon is not None:
        for mod in mods:
            pass

    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()

    def fentropy(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return -(label*np.log(pred+1e-12) + (1.-label)*np.log(1.-pred+1e-12)).mean()

    mG = mx.metric.CustomMetric(fentropy)
    mD = mx.metric.CustomMetric(fentropy)
    mACC = mx.metric.CustomMetric(facc)

    print 'Training...'

    # =============train===============
    for epoch in range(100):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            rbatch = rand_iter.next()

            if mon is not None:
                mon.tic()


            modG.forward(rbatch, is_train=True)
            outG = modG.get_outputs()

            # update discriminator on fake
            label[:] = 0
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            #modD.update()
            gradD = [[grad.copyto(grad.context) for grad in grads] for grads in modD._exec_group.grad_arrays]

            modD.update_metric(mD, [label])
            modD.update_metric(mACC, [label])

            # update discriminator on real
            label[:] = 1
            batch.label = [label]
            modD.forward(batch, is_train=True)
            modD.backward()
            for gradsr, gradsf in zip(modD._exec_group.grad_arrays, gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr += gradf
            modD.update()

            modD.update_metric(mD, [label])
            modD.update_metric(mACC, [label])

            # update generator
            label[:] = 1
            modD.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            modD.backward()
            diffD = modD.get_input_grads()
            modG.backward(diffD)
            modG.update()

            modD.update_metric(mG, [label])


            if mon is not None:
                mon.toc_print()

            t += 1
            if t % 10 == 0:
                print 'epoch:', epoch, 'iter:', t, 'metric:', mACC.get(), mG.get(), mD.get()
                mACC.reset()
                mG.reset()
                mD.reset()

                visual('gout', outG[0].asnumpy())
                diff = diffD[0].asnumpy()
                diff = (diff - diff.mean())/diff.std()
                visual('diff', diff)



