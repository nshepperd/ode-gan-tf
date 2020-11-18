Tensorflow 1.* implementation of [Training Generative Adversarial Networks
by Solving Ordinary Differential
Equations](https://arxiv.org/abs/2010.15040).

Usage
-----

Just drop `ode_gan.py` in your project directory, import ode_gan, and
instantiate your choice of optimizer (`RK2` [Heun's method] or the
slower and more accurate `RK4`), passing the generator and
discriminator parameters and losses.

Call the `step()` method in your training loop to execute a single
training step.
