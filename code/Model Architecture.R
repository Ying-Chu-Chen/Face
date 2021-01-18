feature_symbol <- function(name = '') {
  
  data <- mx.symbol.Variable(paste0('data', name))
  bn_data_p1 <- mx.symbol.BatchNorm(data = data, eps = "2e-05", name = 'bn_data_p1')
  
  conv0_p1 <- mx.symbol.Convolution(data = bn_data_p1, no_bias = TRUE, name = 'conv0_p1',
                                    kernel = c(7, 7), pad = c(3, 3), stride = c(2, 2), num_filter = 64)
  bn0_p1 <- mx.symbol.BatchNorm(data = conv0_p1, fix_gamma = FALSE, eps = "2e-05", name = 'bn0_p1')
  relu0_p1 <- mx.symbol.Activation(data = bn0_p1, act_type = "relu", name = 'relu0_p1')
  
  # stage1_unit1
  
  pooling0_p1 <- mx.symbol.Pooling(data = relu0_p1, pool_type = "max", name = 'pooling0_p1',
                                   kernel = c(3, 3), pad = c(1, 1), stride = c(2, 2))
  stage1_unit1_bn1_p1 <- mx.symbol.BatchNorm(data = pooling0_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage1_unit1_bn1_p1')
  stage1_unit1_relu1_p1 <- mx.symbol.Activation(data = stage1_unit1_bn1_p1, act_type = "relu", name = 'stage1_unit1_relu1_p1')
  stage1_unit1_conv1_p1 <- mx.symbol.Convolution(data = stage1_unit1_relu1_p1, no_bias = TRUE, name = 'stage1_unit1_conv1_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 64)
  stage1_unit1_bn2_p1 <- mx.symbol.BatchNorm(data = stage1_unit1_conv1_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage1_unit1_bn2_p1')
  stage1_unit1_relu2_p1 <- mx.symbol.Activation(data = stage1_unit1_bn2_p1, act_type = "relu", name = 'stage1_unit1_relu2_p1')
  stage1_unit1_conv2_p1 <- mx.symbol.Convolution(data = stage1_unit1_relu2_p1, no_bias = TRUE, name = 'stage1_unit1_conv2_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 64)
  
  stage1_unit1_sc_p1 <- mx.symbol.Convolution(data = stage1_unit1_relu1_p1, no_bias = TRUE, name = 'stage1_unit1_sc_p1',
                                              kernel = c(1, 1), pad = c(0, 0), stride = c(1, 1), num_filter = 64)
  
  elemwise_add_plus0_p1 <- mx.symbol.broadcast_plus(lhs = stage1_unit1_conv2_p1, rhs = stage1_unit1_sc_p1, name = 'elemwise_add_plus0_p1')
  
  # stage1_unit2
  
  stage1_unit2_bn1_p1 <- mx.symbol.BatchNorm(data = elemwise_add_plus0_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage1_unit2_bn1_p1')
  stage1_unit2_relu1_p1 <- mx.symbol.Activation(data = stage1_unit2_bn1_p1, act_type = "relu", name = 'stage1_unit2_relu1_p1')
  stage1_unit2_conv1_p1 <- mx.symbol.Convolution(data = stage1_unit2_relu1_p1, no_bias = TRUE, name = 'stage1_unit2_conv1_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 64)
  stage1_unit2_bn2_p1 <- mx.symbol.BatchNorm(data = stage1_unit2_conv1_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage1_unit2_bn2_p1')
  stage1_unit2_relu2_p1 <- mx.symbol.Activation(data = stage1_unit2_bn2_p1, act_type = "relu", name = 'stage1_unit2_relu2_p1')
  stage1_unit2_conv2_p1 <- mx.symbol.Convolution(data = stage1_unit2_relu2_p1, no_bias = TRUE, name = 'stage1_unit2_conv2_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 64)
  
  elemwise_add_plus1_p1 <- mx.symbol.broadcast_plus(lhs = stage1_unit2_conv2_p1, rhs = elemwise_add_plus0_p1, name = 'elemwise_add_plus1_p1')
  
  # stage2_unit1
  
  stage2_unit1_bn1_p1 <- mx.symbol.BatchNorm(data = elemwise_add_plus1_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage2_unit1_bn1_p1')
  stage2_unit1_relu1_p1 <- mx.symbol.Activation(data = stage2_unit1_bn1_p1, act_type = "relu", name = 'stage2_unit1_relu1_p1')
  stage2_unit1_conv1_p1 <- mx.symbol.Convolution(data = stage2_unit1_relu1_p1, no_bias = TRUE, name = 'stage2_unit1_conv1_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(2, 2), num_filter = 128)
  stage2_unit1_bn2_p1 <- mx.symbol.BatchNorm(data = stage2_unit1_conv1_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage2_unit1_bn2_p1')
  stage2_unit1_relu2_p1 <- mx.symbol.Activation(data = stage2_unit1_bn2_p1, act_type = "relu", name = 'stage2_unit1_relu2_p1')
  stage2_unit1_conv2_p1 <- mx.symbol.Convolution(data = stage2_unit1_relu2_p1, no_bias = TRUE, name = 'stage2_unit1_conv2_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 128)
  
  stage2_unit1_sc_p1 <- mx.symbol.Convolution(data = stage2_unit1_relu1_p1, no_bias = TRUE, name = 'stage2_unit1_sc_p1',
                                              kernel = c(1, 1), pad = c(0, 0), stride = c(2, 2), num_filter = 128)
  
  elemwise_add_plus2_p1 <- mx.symbol.broadcast_plus(lhs = stage2_unit1_conv2_p1, rhs = stage2_unit1_sc_p1, name = 'elemwise_add_plus2_p1')
  
  # stage2_unit2
  
  stage2_unit2_bn1_p1 <- mx.symbol.BatchNorm(data = elemwise_add_plus2_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage2_unit2_bn1_p1')
  stage2_unit2_relu1_p1 <- mx.symbol.Activation(data = stage2_unit2_bn1_p1, act_type = "relu", name = 'stage2_unit2_relu1_p1')
  stage2_unit2_conv1_p1 <- mx.symbol.Convolution(data = stage2_unit2_relu1_p1, no_bias = TRUE, name = 'stage2_unit2_conv1_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 128)
  stage2_unit2_bn2_p1 <- mx.symbol.BatchNorm(data = stage2_unit2_conv1_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage2_unit2_bn2_p1')
  stage2_unit2_relu2_p1 <- mx.symbol.Activation(data = stage2_unit2_bn2_p1, act_type = "relu", name = 'stage2_unit2_relu2_p1')
  stage2_unit2_conv2_p1 <- mx.symbol.Convolution(data = stage2_unit2_relu2_p1, no_bias = TRUE, name = 'stage2_unit2_conv2_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 128)
  
  elemwise_add_plus3_p1 <- mx.symbol.broadcast_plus(lhs = stage2_unit2_conv2_p1, rhs = elemwise_add_plus2_p1, name = 'elemwise_add_plus3_p1')
  
  # stage3_unit1
  
  stage3_unit1_bn1_p1 <- mx.symbol.BatchNorm(data = elemwise_add_plus3_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage3_unit1_bn1_p1')
  stage3_unit1_relu1_p1 <- mx.symbol.Activation(data = stage3_unit1_bn1_p1, act_type = "relu", name = 'stage3_unit1_relu1_p1')
  stage3_unit1_conv1_p1 <- mx.symbol.Convolution(data = stage3_unit1_relu1_p1, no_bias = TRUE, name = 'stage3_unit1_conv1_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(2, 2), num_filter = 256)
  stage3_unit1_bn2_p1 <- mx.symbol.BatchNorm(data = stage3_unit1_conv1_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage3_unit1_bn2_p1')
  stage3_unit1_relu2_p1 <- mx.symbol.Activation(data = stage3_unit1_bn2_p1, act_type = "relu", name = 'stage3_unit1_relu2_p1')
  stage3_unit1_conv2_p1 <- mx.symbol.Convolution(data = stage3_unit1_relu2_p1, no_bias = TRUE, name = 'stage3_unit1_conv2_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 256)
  
  stage3_unit1_sc_p1 <- mx.symbol.Convolution(data = stage3_unit1_relu1_p1, no_bias = TRUE, name = 'stage3_unit1_sc_p1',
                                              kernel = c(1, 1), pad = c(0, 0), stride = c(2, 2), num_filter = 256)
  
  elemwise_add_plus4_p1 <- mx.symbol.broadcast_plus(lhs = stage3_unit1_conv2_p1, rhs = stage3_unit1_sc_p1, name = 'elemwise_add_plus4_p1')
  
  # stage3_unit2
  
  stage3_unit2_bn1_p1 <- mx.symbol.BatchNorm(data = elemwise_add_plus4_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage3_unit2_bn1_p1')
  stage3_unit2_relu1_p1 <- mx.symbol.Activation(data = stage3_unit2_bn1_p1, act_type = "relu", name = 'stage3_unit2_relu1_p1')
  stage3_unit2_conv1_p1 <- mx.symbol.Convolution(data = stage3_unit2_relu1_p1, no_bias = TRUE, name = 'stage3_unit2_conv1_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 256)
  stage3_unit2_bn2_p1 <- mx.symbol.BatchNorm(data = stage3_unit2_conv1_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage3_unit2_bn2_p1')
  stage3_unit2_relu2_p1 <- mx.symbol.Activation(data = stage3_unit2_bn2_p1, act_type = "relu", name = 'stage3_unit2_relu2_p1')
  stage3_unit2_conv2_p1 <- mx.symbol.Convolution(data = stage3_unit2_relu2_p1, no_bias = TRUE, name = 'stage3_unit2_conv2_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 256)
  
  elemwise_add_plus5_p1 <- mx.symbol.broadcast_plus(lhs = stage3_unit2_conv2_p1, rhs = elemwise_add_plus4_p1, name = 'elemwise_add_plus5_p1')
  
  # stage4_unit1
  
  stage4_unit1_bn1_p1 <- mx.symbol.BatchNorm(data = elemwise_add_plus5_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage4_unit1_bn1_p1')
  stage4_unit1_relu1_p1 <- mx.symbol.Activation(data = stage4_unit1_bn1_p1, act_type = "relu", name = 'stage4_unit1_relu1_p1')
  stage4_unit1_conv1_p1 <- mx.symbol.Convolution(data = stage4_unit1_relu1_p1, no_bias = TRUE, name = 'stage4_unit1_conv1_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(2, 2), num_filter = 512)
  stage4_unit1_bn2_p1 <- mx.symbol.BatchNorm(data = stage4_unit1_conv1_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage4_unit1_bn2_p1')
  stage4_unit1_relu2_p1 <- mx.symbol.Activation(data = stage4_unit1_bn2_p1, act_type = "relu", name = 'stage4_unit1_relu2_p1')
  stage4_unit1_conv2_p1 <- mx.symbol.Convolution(data = stage4_unit1_relu2_p1, no_bias = TRUE, name = 'stage4_unit1_conv2_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 512)
  
  stage4_unit1_sc_p1 <- mx.symbol.Convolution(data = stage4_unit1_relu1_p1, no_bias = TRUE, name = 'stage4_unit1_sc_p1',
                                              kernel = c(1, 1), pad = c(0, 0), stride = c(2, 2), num_filter = 512)
  
  elemwise_add_plus6_p1 <- mx.symbol.broadcast_plus(lhs = stage4_unit1_conv2_p1, rhs = stage4_unit1_sc_p1, name = 'elemwise_add_plus6_p1')
  
  # stage4_unit2
  
  stage4_unit2_bn1_p1 <- mx.symbol.BatchNorm(data = elemwise_add_plus6_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage4_unit2_bn1_p1')
  stage4_unit2_relu1_p1 <- mx.symbol.Activation(data = stage4_unit2_bn1_p1, act_type = "relu", name = 'stage4_unit2_relu1_p1')
  stage4_unit2_conv1_p1 <- mx.symbol.Convolution(data = stage4_unit2_relu1_p1, no_bias = TRUE, name = 'stage4_unit2_conv1_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 512)
  stage4_unit2_bn2_p1 <- mx.symbol.BatchNorm(data = stage4_unit2_conv1_p1, fix_gamma = FALSE, eps = "2e-05", name = 'stage4_unit2_bn2_p1')
  stage4_unit2_relu2_p1 <- mx.symbol.Activation(data = stage4_unit2_bn2_p1, act_type = "relu", name = 'stage4_unit2_relu2_p1')
  stage4_unit2_conv2_p1 <- mx.symbol.Convolution(data = stage4_unit2_relu2_p1, no_bias = TRUE, name = 'stage4_unit2_conv2_p1',
                                                 kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 512)
  
  elemwise_add_plus7_p1 <- mx.symbol.broadcast_plus(lhs = stage4_unit2_conv2_p1, rhs = elemwise_add_plus6_p1, name = 'elemwise_add_plus7_p1')
  
  # Final
  
  bn1_p1 <- mx.symbol.BatchNorm(data = elemwise_add_plus7_p1, fix_gamma = FALSE, eps = "2e-05", name = 'bn1_p1')
  relu1_p1 <- mx.symbol.Activation(data = bn1_p1, act_type = "relu", name = 'relu1_p1')
  high_feature <- mx.symbol.Convolution(data = relu1_p1, no_bias = TRUE, name = 'high_feature',
                                          kernel = c(8, 8), stride = c(1, 1), num_filter = 512)
  
  return(high_feature)
}
