import tensorflow as tf
from ode_gan import GANOptimizer

def generator(z):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        # z : [b, 32]
        w1 = tf.get_variable('w1', [32, 25])
        b1 = tf.get_variable('b1', [25])
        x = tf.matmul(z, w1) + b1
        x = tf.nn.relu(x)
        w2 = tf.get_variable('w2', [25, 25])
        b2 = tf.get_variable('b2', [25])
        x = tf.matmul(x, w2) + b2
        x = tf.nn.relu(x)
        w3 = tf.get_variable('w3', [25, 2])
        b3 = tf.get_variable('b3', [2])
        x = tf.matmul(x, w3) + b3
        return x

def discriminator(x):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        # x : [b, 2]
        w1 = tf.get_variable('w1', [2, 25])
        b1 = tf.get_variable('b1', [25])
        x = tf.matmul(x, w1) + b1
        x = tf.nn.relu(x)
        w2 = tf.get_variable('w2', [25, 25])
        b2 = tf.get_variable('b2', [25])
        x = tf.matmul(x, w2) + b2
        x = tf.nn.relu(x)
        w3 = tf.get_variable('w3', [25, 1])
        b3 = tf.get_variable('b3', [1])
        x = tf.matmul(x, w3) + b3
        return x

if __name__ == '__main__':
    with tf.Session() as sess:
        x0 = tf.math.floor(tf.random.uniform([512, 2], minval=0, maxval=4))
        real_data = tf.random.normal([512, 2], stddev=0.2) + x0
        z = tf.random.normal([512, 32])
        fake_data = generator(z)

        real_score = discriminator(real_data)
        fake_score = discriminator(fake_data)

        d_loss = (tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_score), logits=real_score) +
                               tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_score), logits=fake_score)))/512.0
        g_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_score), logits=fake_score))/512.0


        g_params = [v for v in tf.trainable_variables() if 'generator' in v.name]
        d_params = [v for v in tf.trainable_variables() if 'discriminator' in v.name]

        opt = GANOptimizer(g_params=g_params, d_params=d_params, g_loss=g_loss, d_loss=d_loss)

        sess.run(tf.global_variables_initializer())

        counter = 1
        while True:
            opt.step(sess)
            print(counter, sess.run([g_loss, d_loss]))
            if counter % 1000 == 0:
                with open('output.real.data', 'w') as fp:
                    for (x, y) in sess.run(real_data):
                        fp.write(f'{x} {y}\n')
                with open('output.fake.data', 'w') as fp:
                    for (x, y) in sess.run(fake_data):
                        fp.write(f'{x} {y}\n')
            counter += 1
