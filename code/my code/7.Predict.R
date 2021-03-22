
library(mxnet)
library(magrittr)
library(OpenImageR)
library(pROC)
library(imager)
library(jpeg)

img_size = 72
load(paste0('data/valid_list_', img_size, '.RData'))
load(paste0('data/valid_Y_', img_size, '.RData'))

epoch=68
dis_model <- mx.model.load(paste0("train model/train v", epoch), epoch)
dis_sym = mx.symbol.load(paste0("train model/train v", epoch, "-symbol.json"))


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

# Predict

predict_list <- crop_dis_predict(indata = valid_list, LABEL = Valid_Y.array, dis_model = dis_model, 
                                 dis_sym = dis_sym, crop_img_size = 64, img_size = 72, ctx = mx.gpu(1), batch_size = 50)

roc_list <- roc_evalu(response = Valid_Y.array[1,1,1,], predictor = predict_list$batch_dis_record)



# ROC cut point

ROC1 = roc(Valid_Y.array[1,1,1,], predict_list$batch_dis_record)
plot(ROC1, col = "red")
pos = which.max(ROC1$sensitivities + ROC1$specificities)
cutpoint = ROC1$thresholds[pos] #0.6707954

# plot(ROC1, col = "red")
# points(ROC1$specificities[pos], ROC1$sensitivities[pos], pch = 19, cex = 1)
# 
# description = paste0("cut of point: ", formatC(cutpoint, 2, format = "f"),
#                      " (Sens = ", formatC(ROC1$sensitivities[pos], 3, format = "f"),
#                      " ;Spec = ", formatC(ROC1$specificities[pos], 3, format = "f"), ")")
# 
# text(ROC1$specificities[pos], ROC1$sensitivities[pos], description, pos = 1)

# example image input


lin <- readJPEG("data/example image/lin/1.JPG")
lin <- resizeImage(lin, img_size, img_size, method = 'bilinear')
lin_2 <- readJPEG("data/example image/lin/2.JPG")
lin_2 <- resizeImage(lin_2, img_size, img_size, method = 'bilinear')

chen <- readJPEG("data/example image/chen/1.jpg")
chen_2 <- readJPEG("data/example image/chen/2.jpg")
chen <- resizeImage(chen, img_size, img_size, method = 'bilinear')
chen_2 <- resizeImage(chen_2, img_size, img_size, method = 'bilinear')

huang <- readJPEG("data/example image/huang/1.jpg")
huang <- resizeImage(huang, img_size, img_size, method = 'bilinear')


# app predict function

app_predict <- function(indata_1 = chen, indata_2 = huang, img_size = 64, epoch = 68, 
                        dis_cutpoint = cutpoint, ctx = mx.gpu(1), indata_1_name = "chen") {
  
  dis_model = mx.model.load(paste0("train model/train v", epoch), epoch)
  dis_sym = mx.symbol.load(paste0("train model/train v", epoch, "-symbol.json"))
  
  #2. build a dis exec
  dis_layers <- dis_model$symbol$get.internals()
  dis_model_symbol <- which(dis_layers$outputs == 'high_feature_output') %>% dis_layers$get.output()
  arg_lst <- list(symbol = dis_model_symbol, ctx = ctx, grad.req = 'null', data = c(img_size, img_size, 3, 1))
  dis_pred_exc <- do.call(mx.simple.bind, arg_lst)
  
  mx.exec.update.arg.arrays(dis_pred_exc, dis_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(dis_pred_exc, dis_model$aux.params, match.name = TRUE)
  
  #3. predict
  
  SEQ_ARRAY_1 <- array(data = indata_1, dim = c(img_size, img_size, 3, 1))
  SEQ_ARRAY_2 <- array(data = indata_2, dim = c(img_size, img_size, 3, 1))
  
  mx.exec.update.arg.arrays(dis_pred_exc, arg.arrays = list(data = mx.nd.array(SEQ_ARRAY_1, ctx = ctx)), match.name = TRUE)
  mx.exec.forward(dis_pred_exc, is.train = FALSE)
  X1_batch_predict_out <<- dis_pred_exc$outputs[[1]]
  
  mx.exec.update.arg.arrays(dis_pred_exc, arg.arrays = list(data = mx.nd.array(SEQ_ARRAY_2, ctx = mx.gpu(1))), match.name = TRUE)
  mx.exec.forward(dis_pred_exc, is.train = FALSE)
  X2_batch_predict_out <<- dis_pred_exc$outputs[[1]]
  
  # dis
  batch_dis <- as.array(X1_batch_predict_out - X2_batch_predict_out)^2 %>% colSums(., dims = 3) %>% sqrt(.)
  
  if(batch_dis > cutpoint) {
    message(paste0("indata_1 與 indata_2 距離", round(batch_dis, 4), " ,兩者不同人"))
  } else {
    message(paste0("indata_1 與 indata_2 距離", round(batch_dis, 4), " ,都是" , indata_1_name))
  }
  
}

app_crop_predict <- function(indata_1 = chen, indata_2 = huang, img_size = 72, crop_img_size = 64,
                             epoch = 68, dis_cutpoint = cutpoint, ctx = mx.gpu(1), indata_1_name = "chen", 
                             batch_size = 1) {
  
  dis_model = mx.model.load(paste0("train model/train v", epoch), epoch)
  dis_sym = mx.symbol.load(paste0("train model/train v", epoch, "-symbol.json"))
  
  #2. build a dis exec
  dis_layers <- dis_model$symbol$get.internals()
  dis_model_symbol <- which(dis_layers$outputs == 'high_feature_output') %>% dis_layers$get.output()
  arg_lst <- list(symbol = dis_model_symbol, ctx = ctx, grad.req = 'null', data = c(crop_img_size, crop_img_size, 3, batch_size))
  dis_pred_exc <- do.call(mx.simple.bind, arg_lst)
  
  mx.exec.update.arg.arrays(dis_pred_exc, dis_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(dis_pred_exc, dis_model$aux.params, match.name = TRUE)
  
  #3. predict
  
  X1 <- array(data = indata_1, dim = c(img_size, img_size, 3, batch_size))
  X2 <- array(data = indata_2, dim = c(img_size, img_size, 3, batch_size))
  
  batch_dis <- list()
  sub_X1_list <- list()
  sub_X2_list <- list()
  
  sub_X1_list[[1]] = X1[1:64,1:64,,1] #lefttop
  sub_X1_list[[2]] = X1[9:72,1:64,,1] #righttop
  sub_X1_list[[3]] = X1[1:64,9:72,,1] #leftbottom
  sub_X1_list[[4]] = X1[9:72,9:72,,1] #rightbottom
  sub_X1_list[[5]] = X1[5:68,5:68,,1] #center
  
  sub_X2_list[[1]] = X2[1:64,1:64,,1] #lefttop
  sub_X2_list[[2]] = X2[9:72,1:64,,1] #righttop
  sub_X2_list[[3]] = X2[1:64,9:72,,1] #leftbottom
  sub_X2_list[[4]] = X2[9:72,9:72,,1] #rightbottom
  sub_X2_list[[5]] = X2[5:68,5:68,,1] #center
  
  for (m in 1:5) {
    
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
  
  
  if(batch_dis_mean > cutpoint) {
    message(paste0("indata_1 與 indata_2 距離", round(batch_dis_mean, 4), " ,兩者不同人"))
  } else {
    message(paste0("indata_1 與 indata_2 距離", round(batch_dis_mean, 4), " ,都是" , indata_1_name))
  }
  
}

outcome <- app_crop_predict(indata_1 = lin, indata_2 = huang, img_size = 72, crop_img_size = 64,
                            epoch = 60, dis_cutpoint = cutpoint, ctx = mx.gpu(1), indata_1_name = "lin", 
                            batch_size = 1)


 