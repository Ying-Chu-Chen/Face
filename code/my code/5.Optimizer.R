# Optimizer

my_optimizer <- mx.opt.create(name = "adam", learning.rate = 0.00002, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08, wd = 0.0001, rescale.grad = 1)

# my_optimizer <- mx.opt.create(name = "adam", learning.rate = 2e-4, beta1 = 0.5, beta2 = 0.999, epsilon = 1e-08, wd = 0)

#my_optimizer = mx.opt.create(name = "sgd", learning.rate = 0.005, momentum = 0.9, wd = 0.001)

#second_optimizer <- mx.opt.create(name = "adam", learning.rate = 2e-5, beta1 = 0.5, beta2 = 0.999, epsilon = 1e-08, wd = 0)