library(magrittr)
library(imager)
library(OpenImageR)
library(jpeg)
library(pROC)

# Load data

img_size <- 72
batch_size <- 32
crop_img_size <- 64

load(paste0('data/train_list_', img_size, '.RData'))
load(paste0('data/train_Y_', img_size, '.RData'))
load(paste0('data/valid_list_', img_size, '.RData'))
load(paste0('data/valid_Y_', img_size, '.RData'))

# iterator function

random_crop <- function(Train_img.array = Train_img.array1){
  
  random_row = sample(0:8, 1) + 1:64   #random select rows
  random_col = sample(0:8, 1) + 1:64   #random select cols
  Train_img.array <- Train_img.array[random_row,random_col,,] #cropping
  
}

# Iterator

library(mxnet)

# Iterator function

my_iterator_core <- function (batch_size) {
  
  batch <-  0
  batch_per_epoch <- floor(length(train_list[[2]])/batch_size)
  ids.1 <- rep(1:length(train_list[[1]]), ceiling(length(train_list[[2]])/length(train_list[[1]])))
  ids.1 <- ids.1[1:length(train_list[[2]])]
  ids.2 <- 1:length(train_list[[2]])
  
  reset <- function() {batch <<- 0}
  
  iter.next <- function() {
    
    batch <<- batch + 1
    if (batch > batch_per_epoch) {return(FALSE)} else {return(TRUE)}
    
  }
  
  value <- function() {
    
    idx <- 1:batch_size + (batch - 1) * batch_size
    
    idx.1 <- ids.1[idx]
    idx.2 <- ids.2[idx]
    
    Train_img.array1 <- array(0, dim = c(img_size, img_size, 3, batch_size))
    Train_img.array2 <- array(0, dim = c(img_size, img_size, 3, batch_size))
    
    for (i in 1:batch_size) {
      
      Train_img.array1[,,,i] <- train_list[[1]][[idx[i]]]
      Train_img.array2[,,,i] <- train_list[[2]][[idx[i]]]
      
    }
    
    Train_img.array1 <- random_crop(Train_img.array = Train_img.array1)
    Train_img.array2 <- random_crop(Train_img.array = Train_img.array2)
    
    Train_img.array1 <- mx.nd.array(Train_img.array1)
    Train_img.array2 <- mx.nd.array(Train_img.array2)
    
    Train_Y.array <- array(data = Train_Y.array[idx], dim = c(1, 1, 1, 32))
    label = mx.nd.array(Train_Y.array)
    
    return(list(Train_img.array1 = Train_img.array1, Train_img.array2 = Train_img.array2, label = label))
    
  }
  
  return(list(reset = reset, iter.next = iter.next, value = value, batch_size = batch_size, batch = batch))
  
}

my_iterator_func <- setRefClass("Custom_Iter",
                                fields = c("iter", "batch_size"),
                                contains = "Rcpp_MXArrayDataIter",
                                methods = list(
                                  initialize = function(iter, batch_size = 16){
                                    .self$iter <- my_iterator_core(batch_size = batch_size)
                                    .self
                                  },
                                  value = function(){
                                    .self$iter$value()
                                  },
                                  iter.next = function(){
                                    .self$iter$iter.next()
                                  },
                                  reset = function(){
                                    .self$iter$reset()
                                  },
                                  finalize=function(){
                                  }
                                )
)

# Build an iterator

my_iter <- my_iterator_func(iter = NULL, batch_size = 32)


my_iter$reset()
my_iter$iter.next()
#imageShow(train_list[[2]][[2000]])

# Predict function

crop_dis_predict <- function(indata = valid_list, LABEL = Valid_Y.array, dis_model = dis_model, 
                        dis_sym = dis_sym, crop_img_size = 64, img_size = 72, ctx = mx.gpu(1), batch_size = 50) {
  
  #2. build a dis exec
  dis_layers <- dis_model$symbol$get.internals()
  dis_model_symbol <- which(dis_layers$outputs == 'high_feature_output') %>% dis_layers$get.output()
  arg_lst <- list(symbol = dis_model_symbol, ctx = ctx, grad.req = 'null', data = c(crop_img_size, crop_img_size, 3, batch_size))
  dis_pred_exc <- do.call(mx.simple.bind, arg_lst)
  
  mx.exec.update.arg.arrays(dis_pred_exc, dis_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(dis_pred_exc, dis_model$aux.params, match.name = TRUE)
  
  #3. predict function
  
  total_sample_n <- length(indata[[1]])
  total_len <- ceiling(total_sample_n/batch_size)
  
  X1 <- array(data = NA, dim = c(img_size, img_size, 3, total_len*batch_size))
  X2 <- array(data = NA, dim = c(img_size, img_size, 3, total_len*batch_size))
  
  batch_dis_record <- array(data = NA, dim = total_sample_n)

  sub_X1_list <- list()
  sub_X2_list <- list()
  batch_dis <- list()

  for (k in 1:total_len) {
    
    idx <- (k - 1) * batch_size + 1:batch_size
    idx[idx > total_sample_n] <- 1
    
    for (j in idx[1]:idx[batch_size]) {
      
      X1[,,,j] <- indata[[1]][[j]]
      X2[,,,j] <- indata[[2]][[j]]
      
    }
    
    sub_X1_list[[1]] = X1[1:64,1:64,,idx] #lefttop
    sub_X1_list[[2]] = X1[9:72,1:64,,idx] #righttop
    sub_X1_list[[3]] = X1[1:64,9:72,,idx] #leftbottom
    sub_X1_list[[4]] = X1[9:72,9:72,,idx] #rightbottom
    sub_X1_list[[5]] = X1[5:68,5:68,,idx] #center
    
    sub_X2_list[[1]] = X2[1:64,1:64,,idx] #lefttop
    sub_X2_list[[2]] = X2[9:72,1:64,,idx] #righttop
    sub_X2_list[[3]] = X2[1:64,9:72,,idx] #leftbottom
    sub_X2_list[[4]] = X2[9:72,9:72,,idx] #rightbottom
    sub_X2_list[[5]] = X2[5:68,5:68,,idx] #center
    
    for (m in 1:5) {
      
      #aaa <<- dis_pred_exc
      
      batch_SEQ_ARRAY_1 <- array(sub_X1_list[[m]], dim = c(crop_img_size, crop_img_size, 3, batch_size))
      mx.exec.update.arg.arrays(dis_pred_exc, arg.arrays = list(data = mx.nd.array(batch_SEQ_ARRAY_1)), match.name = TRUE)
      mx.exec.forward(dis_pred_exc, is.train = FALSE)
      X1_batch_predict_out <- as.array(dis_pred_exc$ref.outputs[[1]])
      
      batch_SEQ_ARRAY_2 <- array(sub_X2_list[[m]], dim = c(crop_img_size, crop_img_size, 3, batch_size))
      mx.exec.update.arg.arrays(dis_pred_exc, arg.arrays = list(data = mx.nd.array(batch_SEQ_ARRAY_2)), match.name = TRUE)
      mx.exec.forward(dis_pred_exc, is.train = FALSE)
      X2_batch_predict_out <- as.array(dis_pred_exc$ref.outputs[[1]])
      
      batch_dis[[m]] <- (X1_batch_predict_out - X2_batch_predict_out)^2 %>% colSums(., dims = 3) %>% sqrt(.)
      
    }
    
    batch_dis_mean <- (batch_dis[[1]] + batch_dis[[2]] + batch_dis[[3]] + batch_dis[[4]] + batch_dis[[5]]) / 5
    batch_dis_record[idx] <- batch_dis_mean
    
  }
  
  out_list <- list(batch_dis_record = batch_dis_record, LABEL = LABEL)
  
}

roc_evalu <- function(response = Valid_Y.array, predictor = predict_list$batch_dis_record) {
  
  roc_result <- roc(response = response, predictor = predictor)
  return(list(roc_result = roc_result))
  
}

dis_predict <- function(indata = valid_list, LABEL = Valid_Y.array, dis_model = dis_model, 
                        dis_sym = dis_sym, img_size = 72, ctx = mx.gpu(1), batch_size = 50) {
  
  #2. build a dis exec
  dis_layers <- dis_model$symbol$get.internals()
  dis_model_symbol <- which(dis_layers$outputs == 'high_feature_output') %>% dis_layers$get.output()
  arg_lst <- list(symbol = dis_model_symbol, ctx = ctx, grad.req = 'null', data = c(img_size, img_size, 3, batch_size))
  dis_pred_exc <- do.call(mx.simple.bind, arg_lst)
  
  mx.exec.update.arg.arrays(dis_pred_exc, dis_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(dis_pred_exc, dis_model$aux.params, match.name = TRUE)

  #3. predict
  
  total_sample_n <- length(indata[[1]])
  total_len <- ceiling(total_sample_n/batch_size)
  
  X1 <- array(data = NA, dim = c(img_size, img_size, 3, total_len*batch_size))
  X2 <- array(data = NA, dim = c(img_size, img_size, 3, total_len*batch_size))
  
  batch_dis_record <- array(data = NA, dim = total_sample_n)
  
  for (k in 1:total_len) {
    
    idx <- (k - 1) * batch_size + 1:batch_size
    idx[idx > total_sample_n] <- 1
    
    for (j in idx[1]:idx[batch_size]) {
      X1[,,,j] <- indata[[1]][[j]]
      X2[,,,j] <- indata[[2]][[j]]
    }

    batch_SEQ_ARRAY_1 <- array(X1[,,,idx], dim = c(dim(X1)[1:3], batch_size))
    mx.exec.update.arg.arrays(dis_pred_exc, arg.arrays = list(data = mx.nd.array(batch_SEQ_ARRAY_1, ctx = ctx)), match.name = TRUE)
    mx.exec.forward(dis_pred_exc, is.train = FALSE)
    X1_batch_predict_out <<- dis_pred_exc$outputs[[1]]
    #X1_batch_predict_out <- as.array(X1_batch_predict_out)
    
    batch_SEQ_ARRAY_2 <- array(X2[,,,idx], dim = c(dim(X2)[1:3], batch_size))
    mx.exec.update.arg.arrays(dis_pred_exc, arg.arrays = list(data = mx.nd.array(batch_SEQ_ARRAY_2, ctx = mx.gpu(1))), match.name = TRUE)
    mx.exec.forward(dis_pred_exc, is.train = FALSE)
    X2_batch_predict_out <<- dis_pred_exc$outputs[[1]]
    #X2_batch_predict_out <- as.array(X2_batch_predict_out)
    
    # dis
    batch_dis <- as.array(X1_batch_predict_out - X2_batch_predict_out)^2 %>% colSums(., dims = 3) %>% sqrt(.)
    batch_dis_record[idx] <- batch_dis
    
  }
  
  out_list <- list(batch_dis_record = batch_dis_record, LABEL = LABEL)
  
}

