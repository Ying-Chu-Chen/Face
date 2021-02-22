library(magrittr)
library(imager)
library(OpenImageR)
library(jpeg)

# Load data

img_size <- 64
batch_size <- 32

load(paste0('data/train_list_', img_size, '.RData'))
load(paste0('data/train_Y_', img_size, '.RData'))
load(paste0('data/valid_list_', img_size, '.RData'))
load(paste0('data/valid_Y_', img_size, '.RData'))


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
    
    Train_img.array1 <- mx.nd.array(Train_img.array1)
    Train_img.array2 <- mx.nd.array(Train_img.array2)
    
    label = mx.nd.array(Train_Y.array[idx])
    
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
dis_predict <- function(indata = valid_list, LABEL = Valid_Y.array, dis_model = dis_model, dis_sym = dis_sym, img_size = 64, ctx = mx.gpu(1), batch_size = 50) {
  
  #2. build a dis exec
  dis_layers <- dis_model$symbol$get.internals()
  dis_model_symbol <- which(dis_layers$outputs == 'high_feature_output') %>% dis_layers$get.output()
  arg_lst <- list(symbol = dis_model_symbol, ctx = ctx, grad.req = 'null', data = c(img_size, img_size, 3, batch_size))
  dis_pred_exc <- do.call(mx.simple.bind, arg_lst)
  
  mx.exec.update.arg.arrays(dis_pred_exc, dis_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(dis_pred_exc, dis_model$aux_params, match.name = TRUE)
  
  #3. predict
  
  total_sample_n <- length(indata[[1]])
  total_len <- ceiling(total_sample_n/batch_size)
  
  X1 <- array(data = NA, dim = c(img_size, img_size, 3, total_len*batch_size))
  X2 <- array(data = NA, dim = c(img_size, img_size, 3, total_len*batch_size))
  
  batch_dis_record <- array(data = NA, dim = 200)
  
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
  
  out_list <- list(batch_dis_record = batch_dis_record, LABEL = LABEL)
  
}

roc_evalu <- function(response = Valid_Y.array,predictor = predict_list$batch_dis_record) {
  
  roc_result <- roc(response = response, predictor = predictor)
  return(list(roc_result = roc_result))
  
}
