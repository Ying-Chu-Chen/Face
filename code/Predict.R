# 輸出每個人的特徵向量，並計算兩兩向量距離
# 以ROC Curve去選擇最佳切點

library(mxnet)
library(magrittr)
library(OpenImageR)

dis_model = mx.model.load("train model/img256_banch size 32/train v20", 20)
dis_sym = mx.symbol.load("train model/img256_banch size 32/train v20-symbol.json")

img_size <- 256
batch_size <- 32

load(paste0('data/test_list_', img_size, '.RData'))

#imageShow(test_list[[1]][[531]])

test_Y.array <- array(NA, dim = c(1, 1, 1, length(test_list[[1]])))

#奇數同人label 0，偶數不同人 label 1

for (i in 1:(length(test_list[[1]])/2)) {
  test_Y.array[,,,2*i-1] <- rep(0)
  test_Y.array[,,,2*i] <- rep(1)
}

#Get symbol

# all_layers <- dis_model$symbol$get.internals()
# high_feature_1_output <- which(all_layers$outputs == 'high_feature_1_output') %>% all_layers$get.output()

# all_layers = res_sym$get.internals()
# tail(all_layers$outputs, 30)

# my_model <- dis_model
# my_model$symbol <- high_feature_1_output
# my_model$arg.params <- my_model$arg.params[names(my_model$arg.params) %in% names(mx.symbol.infer.shape(high_feature_1_output, data1 = c(img_size, img_size, 3, batch_size))$arg.shapes)]
# my_model$aux.params <- my_model$aux.params[names(my_model$aux.params) %in% names(mx.symbol.infer.shape(high_feature_1_output, data1 = c(img_size, img_size, 3, batch_size))$aux.shapes)]


############################################

#2. build a dis exec
dis_layers <- dis_model$symbol$get.internals()
dis_model_symbol <- which(dis_layers$outputs == 'high_feature_1_output') %>% dis_layers$get.output()
arg_lst <- list(symbol = dis_model_symbol, ctx = mx.gpu(1), grad.req = 'null', data1 = c(img_size, img_size, 3, batch_size))
dis_pred_exc <- do.call(mx.simple.bind, arg_lst)

mx.exec.update.arg.arrays(dis_pred_exc, dis_model$arg.params, match.name = TRUE)
mx.exec.update.aux.arrays(dis_pred_exc, dis_model$aux_params, match.name = TRUE)

#3. predict

total_sample_n <- length(test_list[[1]])
total_len <- ceiling(total_sample_n/batch_size)

X1 <- array(data = NA, dim = c(dim(test_list[[1]][[1]]),total_len*batch_size))
X2 <- array(data = NA, dim = c(dim(test_list[[2]][[1]]),total_len*batch_size))

batch_dis_record <- array(data = NA, dim = total_len*batch_size)
dis_philentropy <- array(data = NA, dim = total_len*batch_size)

for (k in 1:total_len) {
  
  idx <- (k - 1) * batch_size + 1:batch_size
  idx[idx > total_sample_n] <- 1
  
  for (j in idx[1]:idx[batch_size]) {
    X1[,,,j] <- test_list[[1]][[j]]
    X2[,,,j] <- test_list[[2]][[j]]
  }
  
  batch_SEQ_ARRAY_1 <- array(X1[,,,idx], dim = c(dim(X1)[1:3], batch_size))
  mx.exec.update.arg.arrays(dis_pred_exc, arg.arrays = list(data1 = mx.nd.array(batch_SEQ_ARRAY_1, ctx = mx.gpu(1))), match.name = TRUE)
  mx.exec.forward(dis_pred_exc, is.train = FALSE)
  X1_batch_predict_out <- dis_pred_exc$outputs[[1]]
  #X1_batch_predict_out <- as.array(X1_batch_predict_out)
  
  batch_SEQ_ARRAY_2 <- array(X2[,,,idx], dim = c(dim(X2)[1:3], batch_size))
  mx.exec.update.arg.arrays(dis_pred_exc, arg.arrays = list(data1 = mx.nd.array(batch_SEQ_ARRAY_2, ctx = mx.gpu(1))), match.name = TRUE)
  mx.exec.forward(dis_pred_exc, is.train = FALSE)
  X2_batch_predict_out <- dis_pred_exc$outputs[[1]]
  #X2_batch_predict_out <- as.array(X2_batch_predict_out)
  
  # dis
  
  batch_dis <- as.array(X1_batch_predict_out - X2_batch_predict_out)^2 %>% colSums(., dims = 3) %>% sqrt(.)
  batch_dis_record[idx] <- batch_dis
  
  #奇數同人label 0，偶數不同人 label 1
  
  for (i in 1:32) {
    
    dis_data <- array(data = NA, dim = c(2,512))
    X1_batch_predict_out <- as.array(X1_batch_predict_out)
    X2_batch_predict_out <- as.array(X2_batch_predict_out)
    
    dis_data[1,] <- X1_batch_predict_out[,,,i]
    dis_data[2,] <- X2_batch_predict_out[,,,i]
    
    dis_philentropy[idx[i]] <- philentropy::distance(dis_data, method="euclidean")
    #dis_record[i] <- philentropy::distance(dis_data[1:2,], method="cosine")
  }

}


out_list <- list(batch_dis_record = batch_dis_record, LABEL = test_Y.array, dis_philentropy = dis_philentropy)



