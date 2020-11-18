import tensorflow as tf

def store(dsts, srcs):
    assert len(dsts) == len(srcs)
    return tf.group([d.assign(s) for (d,s) in zip(dsts,srcs)])

def mkname(*args):
    out = []
    for arg in args:
        if type(arg) is str:
            out.append(arg)
        elif hasattr(arg, 'name'):
            # Variables
            out.append(arg.name.rsplit(':', 1)[0])
        else:
            assert False
    return '/'.join(out)

class RK2(object):
    def __init__(self, g_params, d_params, g_loss, d_loss, lr=0.02, reg=0.002, name='ODE_GAN'):
        assert type(g_params) == list
        assert type(d_params) == list
        g_grad = tf.gradients(g_loss, g_params)
        d_grad = tf.gradients(d_loss, d_params)

        g_grad_magnitude = sum([tf.reduce_sum(tf.square(g)) for g in g_grad])
        d_penalty = tf.gradients(g_grad_magnitude, d_params)

        # ODE Step requires a few stages due to the fact that we need
        # to combine gradients at different locations, and tf models
        # generally store their weights in mutable variables. Here we
        # use RK2 (Heun's method), which is second order and averages
        # two gradients.

        # First, we compute the gradients, and the penalty gradient at
        # the starting point, and store them for later.
        d_penalty1 = [tf.Variable(tf.zeros_like(v), name=mkname(name, v, 'penalty1'), trainable=False) for v in d_params]
        g_grad1 = [tf.Variable(tf.zeros_like(v), name=mkname(name, v, 'grad1'), trainable=False) for v in g_grad]
        d_grad1 = [tf.Variable(tf.zeros_like(v), name=mkname(name, v, 'grad1'), trainable=False) for v in d_grad]
        with tf.control_dependencies([store(d_penalty1, d_penalty),
                                      store(g_grad1, g_grad),
                                      store(d_grad1, d_grad)]):
            # Then we step to x - lr*g1
            self.step1 = tf.group([param.assign_sub(lr * grad) for (param, grad) in zip(g_params, g_grad)] +
                                  [param.assign_sub(lr * grad) for (param, grad) in zip(d_params, d_grad)])


        # At x - lr*g1, we compute gradients again, and move to the
        # destination:
        #
        # x - (lr/2)(g1+g2) = x - lr*g1 + (lr/2)(g1-g2)
        #
        # Discriminator gets the penalty term '-reg * lr * penalty' to
        # prevent generator gradient from exploding.
        with tf.control_dependencies(g_grad + d_grad):
            self.step2 = tf.group([param.assign_add(0.5 * lr * (g1-g2))
                                   for (param, g1, g2) in zip(g_params, g_grad1, g_grad)] +
                                  [param.assign_add(0.5 * lr * (g1-g2) - reg * lr * penalty)
                                   for (param, g1, g2, penalty) in zip(d_params, d_grad1, d_grad, d_penalty1)])

        # Overall cost: 1*g_params + 2*d_params memory, 2 gradient evaluations, 1 second order gradient evaluation

    def step(self, sess, **kwargs):
        sess.run(self.step1, **kwargs)
        sess.run(self.step2, **kwargs)

class RK4(object):
    def __init__(self, g_params, d_params, g_loss, d_loss, lr=0.02, reg=0.002, name='ODE_GAN'):
        assert type(g_params) == list
        assert type(d_params) == list
        g_grad = tf.gradients(g_loss, g_params)
        d_grad = tf.gradients(d_loss, d_params)

        g_grad_magnitude = sum([tf.reduce_sum(tf.square(g)) for g in g_grad])
        d_penalty = tf.gradients(g_grad_magnitude, d_params)

        # For RK4 we take several steps.

        # First, we compute the gradients, and the penalty gradient at
        # the starting point, and store them for later.
        d_penalty1 = [tf.Variable(tf.zeros_like(v), name=mkname(name, v, 'penalty1'), trainable=False) for v in d_params]
        g_grad1 = [tf.Variable(tf.zeros_like(v), name=mkname(name, v, 'grad1'), trainable=False) for v in g_params]
        d_grad1 = [tf.Variable(tf.zeros_like(v), name=mkname(name, v, 'grad1'), trainable=False) for v in d_params]
        with tf.control_dependencies([store(d_penalty1, d_penalty),
                                      store(g_grad1, g_grad),
                                      store(d_grad1, d_grad)]):
            # Then we step to x - (lr/2)*g1
            self.step1 = tf.group([param.assign_sub(0.5 * lr * g1) for (param, g1) in zip(g_params, g_grad)] +
                                  [param.assign_sub(0.5 * lr * g1) for (param, g1) in zip(d_params, d_grad)])

        # Store gradients at x - (lr/2)*g1
        g_grad2 = [tf.Variable(tf.zeros_like(v), name=mkname(name, v, 'grad2'), trainable=False) for v in g_params]
        d_grad2 = [tf.Variable(tf.zeros_like(v), name=mkname(name, v, 'grad2'), trainable=False) for v in d_params]
        with tf.control_dependencies([store(g_grad2, g_grad),
                                      store(d_grad2, d_grad)]):
            # Then we step to x - (lr/2)*g2
            self.step2 = tf.group([param.assign_sub(0.5 * lr * (g2-g1)) for (param, g1, g2) in zip(g_params, g_grad1, g_grad)] +
                                  [param.assign_sub(0.5 * lr * (g2-g1)) for (param, g1, g2) in zip(d_params, d_grad1, d_grad)])

        # Store gradients at x - (lr/2)*g2
        g_grad3 = [tf.Variable(tf.zeros_like(v), name=mkname(name, v, 'grad3'), trainable=False) for v in g_params]
        d_grad3 = [tf.Variable(tf.zeros_like(v), name=mkname(name, v, 'grad3'), trainable=False) for v in d_params]
        with tf.control_dependencies([store(g_grad3, g_grad),
                                      store(d_grad3, d_grad)]):
            # Then we step to x - lr*g3
            self.step3 = tf.group([param.assign_sub(lr * g3 - 0.5 * lr * g2) for (param, g2, g3) in zip(g_params, g_grad2, g_grad)] +
                                  [param.assign_sub(lr * g3 - 0.5 * lr * g2) for (param, g2, g3) in zip(d_params, d_grad2, d_grad)])

        # At x - lr*g3, we compute gradients again, and move to the
        # destination:
        #
        # x - (h/6)(g1 + 2 g2 + 2 g3 + g4)
        #
        # Discriminator gets the penalty term '-reg * lr * penalty' to
        # prevent generator gradient from exploding.
        with tf.control_dependencies(g_grad + d_grad):
            self.step4 = tf.group([param.assign_sub((lr/6.0) * (g1 + 2.0*g2 + 2.0*g3 + g4) - lr * g3)
                                   for (param, g1, g2, g3, g4) in zip(g_params, g_grad1, g_grad2, g_grad3, g_grad)] +
                                  [param.assign_sub((lr/6.0) * (g1 + 2.0*g2 + 2.0*g3 + g4) - lr * g3 + reg * lr * penalty)
                                   for (param, g1, g2, g3, g4, penalty) in zip(d_params, d_grad1, d_grad2, d_grad3, d_grad, d_penalty1)])


    def step(self, sess, **kwargs):
        sess.run(self.step1, **kwargs)
        sess.run(self.step2, **kwargs)
        sess.run(self.step3, **kwargs)
        sess.run(self.step4, **kwargs)
