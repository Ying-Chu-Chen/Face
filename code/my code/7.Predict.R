# 輸出每個人的特徵向量，並計算兩兩向量距離
# 以ROC Curve去選擇最佳切點

library(mxnet)
library(magrittr)
library(OpenImageR)
library(pROC)

# predict function

dis_predict <- function(img_size = 64, ctx = mx.gpu(1), batch_size = 50) {
  
  dis_model = mx.model.load("train model/train v1", 1)
  dis_sym = mx.symbol.load("train model/train v1-symbol.json")
  
  #2. build a dis exec
  dis_layers <- dis_model$symbol$get.internals()
  dis_model_symbol <- which(dis_layers$outputs == 'high_feature_output') %>% dis_layers$get.output()
  arg_lst <- list(symbol = dis_model_symbol, ctx = ctx, grad.req = 'null', data = c(img_size, img_size, 3, batch_size))
  dis_pred_exc <- do.call(mx.simple.bind, arg_lst)
  
  mx.exec.update.arg.arrays(dis_pred_exc, dis_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(dis_pred_exc, dis_model$aux_params, match.name = TRUE)
  
  #3. predict
  
  total_sample_n <- length(valid_list[[1]])
  total_len <- ceiling(total_sample_n/batch_size)
  
  X1 <- array(data = NA, dim = c(img_size, img_size, 3, total_len*batch_size))
  X2 <- array(data = NA, dim = c(img_size, img_size, 3, total_len*batch_size))
  
  batch_dis_record <- array(data = NA, dim = 200)
  dis_philentropy <- array(data = NA, dim = total_len*batch_size)
  
  for (k in 1:total_len) {
    
    idx <- (k - 1) * batch_size + 1:batch_size
    idx[idx > total_sample_n] <- 1
    
    for (j in idx[1]:idx[batch_size]) {
      X1[,,,j] <- valid_list[[1]][[j]]
      X2[,,,j] <- valid_list[[2]][[j]]
    }
    
    batch_SEQ_ARRAY_1 <- array(X1[,,,idx], dim = c(dim(X1)[1:3], batch_size))
    mx.exec.update.arg.arrays(dis_pred_exc, arg.arrays = list(data = mx.nd.array(batch_SEQ_ARRAY_1, ctx = ctx)), match.name = TRUE)
    mx.exec.forward(dis_pred_exc, is.train = FALSE)
    X1_batch_predict_out <- dis_pred_exc$outputs[[1]]
    #X1_batch_predict_out <- as.array(X1_batch_predict_out)
    
    batch_SEQ_ARRAY_2 <- array(X2[,,,idx], dim = c(dim(X2)[1:3], batch_size))
    mx.exec.update.arg.arrays(dis_pred_exc, arg.arrays = list(data = mx.nd.array(batch_SEQ_ARRAY_2, ctx = mx.gpu(1))), match.name = TRUE)
    mx.exec.forward(dis_pred_exc, is.train = FALSE)
    X2_batch_predict_out <- dis_pred_exc$outputs[[1]]
    #X2_batch_predict_out <- as.array(X2_batch_predict_out)
    
    # dis
    batch_dis <- as.array(X1_batch_predict_out - X2_batch_predict_out)^2 %>% colSums(., dims = 3) %>% sqrt(.)
    batch_dis_record[idx] <- batch_dis
    
  }
  
  out_list <- list(batch_dis_record = batch_dis_record, LABEL = Valid_Y.array)
  
}

roc_evalu <- function(response = Valid_Y.array,predictor = predict_list$batch_dis_record) {
  
  roc_result <- roc(response = response, predictor = predictor)
  return(list(roc_result = roc_result))
  
}



# philentropy packages

dis_data <- array(data = NA, dim = c(2,total_len*batch_size))
X1_batch_predict_out <- as.array(X1_batch_predict_out)
X2_batch_predict_out <- as.array(X2_batch_predict_out)

for (i in 1:32) {
  
  dis_data[1,i] <- X1_batch_predict_out[,,,i]
  dis_data[2,i] <- X2_batch_predict_out[,,,i]
  
  dis_philentropy[idx[i]] <- philentropy::distance(dis_data, method="euclidean")
  #dis_record[i] <- philentropy::distance(dis_data[1:2,], method="cosine")
}
