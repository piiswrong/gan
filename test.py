def make_torch_sym():
    rand = mx.sym.Variable('rand')

    g1 = mx.sym.TorchModule(rand, lua_string='nn.SpatialFullConvolution(100, 64 * 8, 4, 4)', name='g1', num_data=1, num_params=2, num_outputs=1)
    gbn1 = mx.sym.TorchModule(data_0=g1, weight=mx.sym.Variable('gbn1_gamma'), bias=mx.sym.Variable('gbn1_beta'), lua_infershape_string='nn.SpatialBatchNormalization(64 * 8)', lua_string='cudnn.SpatialBatchNormalization(64 * 8)', name='gbn1', num_data=1, num_params=2, num_outputs=1)
    gact1 = mx.sym.TorchModule(gbn1, lua_string='nn.ReLU(false)', name='gact1', num_data=1, num_params=0, num_outputs=1)

    g2 = mx.sym.TorchModule(gact1, lua_string='nn.SpatialFullConvolution(64 * 8, 64 * 4, 4, 4, 2, 2, 1, 1)', name='g2', num_data=1, num_params=2, num_outputs=1)
    gbn2 = mx.sym.TorchModule(data_0=g2, weight=mx.sym.Variable('gbn2_gamma'), bias=mx.sym.Variable('gbn2_beta'), lua_infershape_string='nn.SpatialBatchNormalization(64 * 4)', lua_string='cudnn.SpatialBatchNormalization(64 * 4)', name='gbn2', num_data=1, num_params=2, num_outputs=1)
    gact2 = mx.sym.TorchModule(gbn2, lua_string='nn.ReLU(false)', name='gact2', num_data=1, num_params=0, num_outputs=1)

    g3 = mx.sym.TorchModule(gact2, lua_string='nn.SpatialFullConvolution(64 * 4, 64 * 2, 4, 4, 2, 2, 1, 1)', name='g3', num_data=1, num_params=2, num_outputs=1)
    gbn3 = mx.sym.TorchModule(data_0=g3, weight=mx.sym.Variable('gbn3_gamma'), bias=mx.sym.Variable('gbn3_beta'), lua_infershape_string='nn.SpatialBatchNormalization(64 * 2)', lua_string='cudnn.SpatialBatchNormalization(64 * 2)', name='gbn3', num_data=1, num_params=2, num_outputs=1)
    gact3 = mx.sym.TorchModule(gbn3, lua_string='nn.ReLU(false)', name='gact3', num_data=1, num_params=0, num_outputs=1)

    g4 = mx.sym.TorchModule(gact3, lua_string='nn.SpatialFullConvolution(64 * 2, 64, 4, 4, 2, 2, 1, 1)', name='g4', num_data=1, num_params=2, num_outputs=1)
    gbn4 = mx.sym.TorchModule(data_0=g4, weight=mx.sym.Variable('gbn4_gamma'), bias=mx.sym.Variable('gbn4_beta'), lua_infershape_string='nn.SpatialBatchNormalization(64)', lua_string='cudnn.SpatialBatchNormalization(64)', name='gbn4', num_data=1, num_params=2, num_outputs=1)
    gact4 = mx.sym.TorchModule(gbn4, lua_string='nn.ReLU(false)', name='gact4', num_data=1, num_params=0, num_outputs=1)

    g5 = mx.sym.TorchModule(gact4, lua_string='nn.SpatialFullConvolution(64, 3, 4, 4, 2, 2, 1, 1)', name='g5', num_data=1, num_params=2, num_outputs=1)
    gout = mx.sym.TorchModule(g5, lua_string='nn.Tanh()', name='gact5', num_data=1, num_params=0, num_outputs=1)

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    d1 = mx.sym.TorchModule(data, lua_string='nn.SpatialConvolution(3, 64, 4, 4, 2, 2, 1, 1)', name='d1', num_data=1, num_params=2, num_outputs=1)
    dact1 = mx.sym.TorchModule(d1, lua_string='nn.LeakyReLU(0.2, false)', name='dact1', num_data=1, num_params=0, num_outputs=1)

    d2 = mx.sym.TorchModule(dact1, lua_string='nn.SpatialConvolution(64, 64 * 2, 4, 4, 2, 2, 1, 1)', name='d2', num_data=1, num_params=2, num_outputs=1)
    dbn2 = mx.sym.TorchModule(data_0=d2, weight=mx.sym.Variable('dbn2_gamma'), bias=mx.sym.Variable('dbn2_beta'), lua_infershape_string='nn.SpatialBatchNormalization(64 * 2)', lua_string='cudnn.SpatialBatchNormalization(64 * 2)', name='dbn2', num_data=1, num_params=2, num_outputs=1)
    dact2 = mx.sym.TorchModule(dbn2, lua_string='nn.LeakyReLU(0.2, false)', num_data=1, num_params=0, num_outputs=1)

    d3 = mx.sym.TorchModule(dact2, lua_string='nn.SpatialConvolution(64 * 2, 64 * 4, 4, 4, 2, 2, 1, 1)', name='d3', num_data=1, num_params=2, num_outputs=1)
    dbn3 = mx.sym.TorchModule(data_0=d3, weight=mx.sym.Variable('dbn3_gamma'), bias=mx.sym.Variable('dbn3_beta'), lua_infershape_string='nn.SpatialBatchNormalization(64 * 4)', lua_string='cudnn.SpatialBatchNormalization(64 * 4)', name='dbn3', num_data=1, num_params=2, num_outputs=1)
    dact3 = mx.sym.TorchModule(dbn3, lua_string='nn.LeakyReLU(0.2, false)', name='dact3', num_data=1, num_params=0, num_outputs=1)

    d4 = mx.sym.TorchModule(dact3, lua_string='nn.SpatialConvolution(64 * 4, 64 * 8, 4, 4, 2, 2, 1, 1)', name='d4', num_data=1, num_params=2, num_outputs=1)
    dbn4 = mx.sym.TorchModule(data_0=d4, weight=mx.sym.Variable('dbn4_gamma'), bias=mx.sym.Variable('dbn4_beta'), lua_infershape_string='nn.SpatialBatchNormalization(64 * 8)', lua_string='cudnn.SpatialBatchNormalization(64 * 8)', name='dbn4', num_data=1, num_params=2, num_outputs=1)
    dact4 = mx.sym.TorchModule(dbn4, lua_string='nn.LeakyReLU(0.2, false)', name='dact4', num_data=1, num_params=0, num_outputs=1)

    d5 = mx.sym.TorchModule(dact4, lua_string='nn.SpatialConvolution(64 * 8, 1, 4, 4)', name='d5', num_data=1, num_params=2, num_outputs=1)
    dact5 = mx.sym.TorchModule(d5, lua_string='nn.Sigmoid()', name='dact5', num_data=1, num_params=0, num_outputs=1)
    dact5 = mx.sym.Flatten(dact5)

    #dloss = dact5
    dloss = mx.sym.TorchCriterion(data=dact5, label=label, lua_string='nn.BCECriterion()', name='dloss')
    #dloss = mx.sym.LogisticRegressionOutput(data=dact5, label=label, name='dloss')

    return gout, dloss


def check():
    def try_allclose(*args, **kwargs):
        from numpy.testing import assert_allclose
        try:
            assert_allclose(*args, **kwargs)
        except AssertionError as e:
            print e

    ndf = 64
    ngf = 64
    nc = 3
    batch_size = 3
    Z = 100
    lr = 0.0002
    beta1 = 0.5
    ctx = mx.gpu(3)

    rand_shape = (batch_size, Z, 1, 1)
    data_shape = (batch_size, nc, 64, 64)

    symG, symD = make_dcgan_sym(ngf, ndf, nc, no_bias=False)

    modG = mx.mod.Module(symbol=symG, data_names=('rand',), label_names=None, context=ctx)
    modG.bind(data_shapes=[('rand', rand_shape)])
    modG.init_params(initializer=mx.init.Normal(0.02))
    modG.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
        })

    modD = mx.mod.Module(symbol=symD, data_names=('data',), label_names=('label',), context=ctx)
    modD.bind(data_shapes=[('data', data_shape)],
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

    symG, symD = make_torch_sym()

    torchG = mx.mod.Module(symbol=symG, data_names=('rand',), label_names=None, context=ctx)
    torchG.bind(data_shapes=[('rand', rand_shape)])
    arg_params, aux_params = modG.get_params()
    torchG.init_params(arg_params=arg_params, aux_params=aux_params)
    torchG.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
        })

    torchD = mx.mod.Module(symbol=symD, data_names=('data',), label_names=('label',), context=ctx)
    torchD.bind(data_shapes=[('data', data_shape)],
              label_shapes=[('label', (batch_size,))],
              inputs_need_grad=True)
    arg_params, aux_params = modD.get_params()
    torchD.init_params(arg_params=arg_params, aux_params=aux_params)
    torchD.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
        })

    rand = mx.random.normal(0, 1.0, shape=rand_shape)
    data = mx.random.uniform(-1., 1., shape=data_shape)
    label = mx.nd.ones(shape=(batch_size,))

    modG.forward(mx.io.DataBatch([rand], []), is_train=True)
    torchG.forward(mx.io.DataBatch([rand], []), is_train=True)

    try_allclose(modG.get_outputs()[0].asnumpy(), torchG.get_outputs()[0].asnumpy(), rtol=1e-5)

    modD.forward(mx.io.DataBatch(modG.get_outputs(), [label]), is_train=True)
    torchD.forward(mx.io.DataBatch(torchG.get_outputs(), [label]), is_train=True)

    print -np.log(modD.get_outputs()[0].asnumpy()).mean()
    print torchD.get_outputs()[0].asnumpy().mean()

    modD.backward()
    torchD.backward()

    # for name, mxgrads, thgrads in zip(modD._exec_group.param_names, modD._exec_group.grad_arrays, torchD._exec_group.grad_arrays):
    #     for mxgrad, thgrad in zip(mxgrads, thgrads):
    #         print name
    #         try_allclose(mxgrad.asnumpy(), thgrad.asnumpy(), rtol=1e-5)

    modG.backward(modD.get_input_grads())
    torchG.backward(torchD.get_input_grads())

    for name, mxgrads, thgrads in zip(modG._exec_group.param_names, modG._exec_group.grad_arrays, torchG._exec_group.grad_arrays):
        for mxgrad, thgrad in zip(mxgrads, thgrads):
            print name
            try_allclose(mxgrad.asnumpy(), thgrad.asnumpy(), rtol=1e-3)

