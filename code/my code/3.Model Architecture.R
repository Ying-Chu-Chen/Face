

feature_symbol <- function() {
  
  data <- mx.symbol.Variable(name = 'data')
  
  dis_conv1 <- mx.symbol.Convolution(data = data, kernel = c(3, 3), num_filter = 24, no.bias = TRUE, name = 'dis_conv1')
  dis_bn1 <- mx.symbol.BatchNorm(data = dis_conv1, fix_gamma = FALSE, name = 'dis_bn1')
  dis_relu1 <- mx.symbol.LeakyReLU(data = dis_bn1, act_type = "leaky", slope = 0.2, name = "dis_relu1")
  dis_pool1 <- mx.symbol.Pooling(data = dis_relu1, pool_type = "avg", kernel = c(2, 2), stride = c(2, 2), name = 'dis_pool1')
  
  dis_conv2 <- mx.symbol.Convolution(data = dis_pool1, kernel = c(3, 3), stride = c(2, 2), num_filter = 32, no.bias = TRUE, name = 'dis_conv2')
  dis_bn2 <- mx.symbol.BatchNorm(data = dis_conv2, fix_gamma = TRUE, name = 'dis_bn2')
  dis_relu2 <- mx.symbol.LeakyReLU(data = dis_bn2, act_type = "leaky", slope = 0.2, name = "dis_relu2")
  dis_pool2 <- mx.symbol.Pooling(data = dis_relu2, pool_type = "avg", kernel = c(2, 2), stride = c(2, 2), name = 'dis_pool2')
  
  dis_conv3 <- mx.symbol.Convolution(data = dis_pool2, kernel = c(3, 3), num_filter = 64, no.bias = TRUE, name = 'dis_conv3')
  dis_bn3 <- mx.symbol.BatchNorm(data = dis_conv3, fix_gamma = FALSE, name = 'dis_bn3')
  dis_relu3 <- mx.symbol.LeakyReLU(data = dis_bn3, act_type = "leaky", slope = 0.2, name = "dis_relu3")
  
  dis_conv4 <- mx.symbol.Convolution(data = dis_relu3, kernel = c(4, 4), num_filter = 64, no.bias = TRUE, name = 'dis_conv4')
  dis_bn4 <- mx.symbol.BatchNorm(data = dis_conv4, fix_gamma = FALSE, name = 'dis_bn4')
  dis_relu4 <- mx.symbol.LeakyReLU(data = dis_bn4, act_type = "leaky", slope = 0.2, name = "dis_relu4")
  
  high_feature <- mx.symbol.Convolution(data = dis_relu4, kernel = c(2, 2), num_filter = 128, name = 'high_feature')
  
  # high_feature <- mx.symbol.Convolution(data = relu1, no_bias = TRUE, name = 'high_feature',
  #                                         kernel = c(7, 7), stride = c(1, 1), num_filter = 512)

  return(high_feature)
}

data <- mx.symbol.Variable(name = 'data')

dis_conv1 <- mx.symbol.Convolution(data = data, kernel = c(3, 3), num_filter = 24, no.bias = TRUE, name = 'dis_conv1')
dis_bn1 <- mx.symbol.BatchNorm(data = dis_conv1, fix_gamma = FALSE, name = 'dis_bn1')
dis_relu1 <- mx.symbol.LeakyReLU(data = dis_bn1, act_type = "leaky", slope = 0.2, name = "dis_relu1")
dis_pool1 <- mx.symbol.Pooling(data = dis_relu1, pool_type = "avg", kernel = c(2, 2), stride = c(2, 2), name = 'dis_pool1')

dis_conv2 <- mx.symbol.Convolution(data = dis_pool1, kernel = c(3, 3), stride = c(2, 2), num_filter = 32, no.bias = TRUE, name = 'dis_conv2')
dis_bn2 <- mx.symbol.BatchNorm(data = dis_conv2, fix_gamma = TRUE, name = 'dis_bn2')
dis_relu2 <- mx.symbol.LeakyReLU(data = dis_bn2, act_type = "leaky", slope = 0.2, name = "dis_relu2")
dis_pool2 <- mx.symbol.Pooling(data = dis_relu2, pool_type = "avg", kernel = c(2, 2), stride = c(2, 2), name = 'dis_pool2')

dis_conv3 <- mx.symbol.Convolution(data = dis_pool2, kernel = c(3, 3), num_filter = 64, no.bias = TRUE, name = 'dis_conv3')
dis_bn3 <- mx.symbol.BatchNorm(data = dis_conv3, fix_gamma = FALSE, name = 'dis_bn3')
dis_relu3 <- mx.symbol.LeakyReLU(data = dis_bn3, act_type = "leaky", slope = 0.2, name = "dis_relu3")

dis_conv4 <- mx.symbol.Convolution(data = dis_relu3, kernel = c(4, 4), num_filter = 64, no.bias = TRUE, name = 'dis_conv4')
dis_bn4 <- mx.symbol.BatchNorm(data = dis_conv4, fix_gamma = FALSE, name = 'dis_bn4')
dis_relu4 <- mx.symbol.LeakyReLU(data = dis_bn4, act_type = "leaky", slope = 0.2, name = "dis_relu4")

high_feature <- mx.symbol.Convolution(data = dis_relu4, kernel = c(2, 2), num_filter = 128, name = 'high_feature')


mx.symbol.infer.shape(high_feature, data = c(64, 64, 3, batch_size))$out.shapes
