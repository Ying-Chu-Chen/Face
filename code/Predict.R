library(mxnet)

res_model = mx.model.load("train model/img256_banch size 32/train v20", 20)
res_sym = mx.symbol.load("train model/img256_banch size 32/train v20-symbol.json")

img_size <- 256
batch_size <- 32

load(paste0('data/test_list_', img_size, '.RData'))

#Get symbol

all_layers <- res_model$symbol$get.internals()
high_feature_1_output <- which(all_layers$outputs == 'high_feature_1_output') %>% all_layers$get.output()

# all_layers = res_sym$get.internals()
# tail(all_layers$outputs, 30)

my_model <- res_model
my_model$symbol <- high_feature_1_output
my_model$arg.params <- my_model$arg.params[names(my_model$arg.params) %in% names(mx.symbol.infer.shape(high_feature_1_output, data1 = c(img_size, img_size, 3, batch_size))$arg.shapes)]
my_model$aux.params <- my_model$aux.params[names(my_model$aux.params) %in% names(mx.symbol.infer.shape(high_feature_1_output, data1 = c(img_size, img_size, 3, batch_size))$aux.shapes)]

high_feature_1_output <- predict(my_model, X = test_list[[1]][[1]], ctx = mx.gpu())
dim(high_feature_1_output)



dis_predict <- function(x1 = NULL, x2 = NULL, 
                        dis_model, dis_out_layer_name,
                        batch_size = 50, my_ctx, 
                        crop_len, crop_stride, scale_input,
                        sample_neg_size = 50, sample_pos_size = 500) {
  
  if (is.null(x1) + is.null(x2) == 1){stop("You have to assign both x1 and x2")}
  if (is.null(x1) & is.null(x2)) {
    
    message("Process valid data")
    
    filename <- valid_data$filename
    cno <- valid_data$cno
    uni_cno <- unique(cno)
    
    if (length(uni_cno) < sample_pos_size){ sample_pos_size <- length(uni_cno)}
    if (length(uni_cno) < sample_neg_size){ sample_neg_size <- length(uni_cno)}
    
    sample_pos_cno <- base:::sample(x = uni_cno, size = sample_pos_size, replace = FALSE) 
    sample_neg_cno <- base:::sample(x = uni_cno, size = sample_neg_size, replace = FALSE) 
    
    seq_list_1 <- vector(mode = "list", length = (sample_pos_size + sample_neg_size))
    seq_list_2 <- vector(mode = "list", length = (sample_pos_size + sample_neg_size))
    label <- array(data = NA, dim = c(1, (sample_pos_size + sample_neg_size)))
    
    # To get identity comparsion
    
    for (i in 1:sample_pos_size) {
      pos_combn <- filename[cno == sample_pos_cno[i]] %>% combn(x = ., m = 2, simplify = FALSE) 
      ind_list <- base:::sample(x = length(pos_combn), size = 1, replace = FALSE) 
      
      load(paste0(ecg_dir, pos_combn[[ind_list]][1]))
      seq_list_1[[i]] <- do.call('cbind', ecg_list)
      
      load(paste0(ecg_dir, pos_combn[[ind_list]][2]))
      seq_list_2[[i]] <- do.call('cbind', ecg_list)
      
      label[, i] <- 0
    }
    
    #####################################
    # To get different comparsion
    
    for (i in 1:sample_neg_size) {
      
      neg_path <- filename[cno == sample_neg_cno[i]] %>% base:::sample(x = ., size = 1, replace = FALSE) 
      load(paste0(ecg_dir, neg_path))
      seq_list_1[[(sample_pos_size+i)]] <- do.call('cbind', ecg_list)
      
      other_cno <- uni_cno[uni_cno != sample_neg_cno[i]] %>% base:::sample(x = ., size = 1, replace = FALSE) 
      other_path <- filename[cno %in% other_cno] %>% base:::sample(x = ., size = 1, replace = FALSE) 
      load(paste0(ecg_dir, other_path))
      seq_list_2[[(sample_pos_size+i)]] <- do.call('cbind', ecg_list)
      
      label[, (sample_pos_size+i)] <- 1
    }
    
    X1 <- abind(seq_list_1, along = 3)
    X2 <- abind(seq_list_2, along = 3)
    
    X1 <- seq_crop(X1, full_len = 5000, crop_len = crop_len, crop_stride = crop_stride)
    X2 <- seq_crop(X2, full_len = 5000, crop_len = crop_len, crop_stride = crop_stride)    
    
    X1 <- array(X1, dim = c(dim(X1)[1:2], 1, dim(X1)[3])) * scale_input
    X2 <- array(X2, dim = c(dim(X2)[1:2], 1, dim(X2)[3])) * scale_input
    
    LABEL <- label
    message("Process valid data done.")
    
  } else {
    X1 <- x1
    X2 <- x2
    LABEL <- NULL
  }
  
  #2. build a dis exec
  dis_layers <- dis_model$symbol$get.internals()
  dis_model_symbol <- which(grepl(dis_out_layer_name, dis_layers$outputs)) %>% dis_layers$get.output()
  arg_lst <- list(symbol = dis_model_symbol, ctx = my_ctx, grad.req = 'null', indiv = c(dim(X1)[1:3], batch_size))
  dis_pred_exc <- do.call(mx.simple.bind, arg_lst)
  
  mx.exec.update.arg.arrays(dis_pred_exc, dis_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(dis_pred_exc, dis_model$aux_params, match.name = TRUE)
  
  #3. predict
  
  total_sample_n <- dim(X1)[length(dim(X1))]
  total_len <- ceiling(total_sample_n/batch_size)
  
  predict_norml2 <- NULL
  
  pb <- txtProgressBar(max = total_len, style = 3)
  t0 <- Sys.time()
  
  for (k in 1:total_len) {
    
    idx <- (k - 1) * batch_size + 1:batch_size
    idx[idx > total_sample_n] <- 1
    
    batch_SEQ_ARRAY_1 <- array(X1[,,,idx], dim = c(dim(X1)[1:3], batch_size))
    mx.exec.update.arg.arrays(dis_pred_exc, arg.arrays = list(indiv = mx.nd.array(batch_SEQ_ARRAY_1, ctx = my_ctx)), match.name = TRUE)
    mx.exec.forward(dis_pred_exc, is.train = FALSE)
    X1_batch_predict_out <- dis_pred_exc$outputs[[1]]
    
    batch_SEQ_ARRAY_2 <- array(X2[,,,idx], dim = c(dim(X2)[1:3], batch_size))
    mx.exec.update.arg.arrays(dis_pred_exc, arg.arrays = list(indiv = mx.nd.array(batch_SEQ_ARRAY_2, ctx = my_ctx)), match.name = TRUE)
    mx.exec.forward(dis_pred_exc, is.train = FALSE)
    X2_batch_predict_out <- dis_pred_exc$outputs[[1]]
    
    # norm
    
    batch_norm <- as.array(X1_batch_predict_out - X2_batch_predict_out)^2 %>% colSums(., dims = 1) %>% sqrt(.)
    
    predict_norml2[idx] <- batch_norm
    
    setTxtProgressBar(pb, k)
  }
  
  close(pb)
  
  time <- as.double(difftime(Sys.time(), t0,  units = "secs"))
  speed <- total_len * batch_size / time
  
  message(paste0("Total sample = ", total_len * batch_size, "  Speed = ", formatC(speed, 3, format = "f"), " samples/sec"))
  
  out_list <- list(predict_norml2 = predict_norml2, LABEL = LABEL)
  
  return(out_list)
}


predict_list <- dis_predict(model = my_model, batch_size = 50, my_ctx = my_ctx, min_individual = min_individual,
                            sample_neg_size = evalu_neg_size, sample_pos_size = evalu_pos_size)
