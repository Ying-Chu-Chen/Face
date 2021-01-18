source('~/Face/code/Iterator.R')
source('~/Face/code/Model Architecture.R')
source('~/Face/code/Loss Function.R')
source('~/Face/code/Optimizer.R')

# Executor & Train

#Input shape

my_iter$reset()
my_iter$iter.next()
train_data <- my_iter$value()
input_dim <- sapply(train_data, dim)
batch_size <- tail(input_dim[[1]], 1)

# Ouput shape

my_values <- my_iter$value()
ouput_dim = mx.symbol.infer.shape(feature_symbol_1, data1 = c(img_size, img_size, 3, batch_size))$out.shapes

get_output_dim <- function(input_dim, loss_name){
  
  # output shape decide by input shape
  
  ouput_dim = mx.symbol.infer.shape(feature_symbol_1, data1 = c(img_size, img_size, 3, batch_size))$out.shapes
  
  output_shape <- list()
  for (i in 1:length(loss_name)){
    if (loss_name[[i]] == "label"){
      output_shape[[i]] <- dim(my_values[['label']])
      names(output_shape)[i] <- 'label'
    } else {
      output_shape[[i]] <- ouput_dim
      names(output_shape)[i] <- loss_name[[i]]
    }
  }
  
  return(output_shape)
}

ouput_dim <- get_output_dim(input_dim = input_dim, loss_name = loss_name)

# Feature Extracter Executor

exec_list <- list(symbol = feature_symbol_1, fixed.param = NULL, ctx = mx.gpu(1), grad.req = "write")
exec_list <- append(exec_list, list(data1 = c(img_size, img_size, 3, batch_size)))
my_executor <- do.call(mx.simple.bind, exec_list)

# Loss Executor

loss_exec_list <- list(symbol = dis_loss, ctx = mx.gpu(1), grad.req = "write")
loss_exec_list <- append(loss_exec_list, list(person_1 = c(1, 1, 512, batch_size), person_2 = c(1, 1, 512, batch_size), label = batch_size))
#loss_exec_list <- append(loss_exec_list, ouput_dim)
loss_executor <- do.call(mx.simple.bind, loss_exec_list)

# Initial params

my_arg <- list()
my_arg$arg.params <- list()
my_arg$aux.params <- list()

init_list <- list(symbol = feature_symbol_1, ctx = mx.gpu(1), input.shape = list(data1 = c(img_size, img_size, 3, batch_size)), output.shape = NULL)
init_list <- append(init_list, list(initializer = mxnet:::mx.init.uniform(0.01)))
my_arg <- do.call(mx.model.init.params, init_list)

# Update parameters

mx.exec.update.arg.arrays(my_executor, my_arg$arg.params, match.name = TRUE)
mx.exec.update.aux.arrays(my_executor, my_arg$aux.params, match.name = TRUE)

#message(paste0('The number of total parameters = ', sum(sapply(my_arg$arg.params, length))))

my_updater <- mx.opt.get.updater(optimizer = my_optimizer, weights = my_executor$ref.arg.arrays)

# Forward/Backward

epoch <- 1
end_epoch <- 20
loss_report <- matrix(data = NA, nrow = end_epoch, ncol = 1)

for (epoch in 1:end_epoch) {
  
  # Training
  
  my_iter$reset()
  batch_loss <- list()
  batch_t0 <- Sys.time()
  
  while (my_iter$iter.next()) {
    
    my_values <- my_iter$value()
    
    # Person 1 forward
    
    mx.exec.update.arg.arrays(my_executor, arg.arrays = list(data1 = my_values[['Train_img.array1']]), match.name = TRUE)
    mx.exec.forward(my_executor, is.train = TRUE)
    person_1_ouput <- my_executor$outputs[[1]]
    #person_1_ouput <- as.array(my_executor$ref.outputs[[1]])
    #person_w_ouput <- my_executor$outputs[[1]]
    
    # as.array(person_1_ouput)[,,1,1]
    # as.array(person_w_ouput)[,,1,1]
    
    # Person 2 forward
    
    mx.exec.update.arg.arrays(my_executor, arg.arrays = list(data1 = my_values[['Train_img.array2']]), match.name = TRUE)
    mx.exec.forward(my_executor, is.train = TRUE)
    person_2_ouput <- my_executor$outputs[[1]]
    #person_2_ouput <- as.array(my_executor$ref.outputs[[1]])
    
    
    # prepare arg from person 1 and person 2
    
    arg_list <- vector("list", length(ouput_dim))
    names(arg_list) <- names(ouput_dim)
    arg_list[['person_1']] <- person_1_ouput
    arg_list[['person_2']] <- person_2_ouput
    # arg_list[['person_1']] <- mx.nd.array(person_1_ouput, ctx = mx.gpu(0))
    # arg_list[['person_2']] <- mx.nd.array(person_2_ouput, ctx = mx.gpu(0))
    arg_list[['label']] <- my_values[['label']]
    
    # Compute loss and save loss from separate network
    
    mx.exec.update.arg.arrays(loss_executor, arg.arrays = arg_list, match.name = TRUE)      
    mx.exec.forward(loss_executor, is.train = TRUE)
    mx.exec.backward(loss_executor)
    gradient_1 <- loss_executor$grad.arrays$person_1
    gradient_2 <- loss_executor$grad.arrays$person_2
    # gradient_1 <- as.array(loss_executor$ref.grad.arrays$person_1)
    # gradient_2 <- as.array(loss_executor$ref.grad.arrays$person_2)
    
    # Foward person_1 input and get grad_array_1
    
    mx.exec.update.arg.arrays(my_executor, arg.arrays = list(data1 = my_values[['Train_img.array1']]), match.name = TRUE)
    mx.exec.forward(my_executor, is.train = TRUE)
    mx.exec.backward(my_executor, out_grads = gradient_1)
    #mx.exec.backward(my_executor, out_grads = mx.nd.array(gradient_1, ctx = mx.gpu(0)))
    grad_array_1 <- my_executor$grad.arrays
    #grad_array_1 <- my_executor$ref.grad.arrays
    
    # Foward person_2 input and get grad_array_2
    
    mx.exec.update.arg.arrays(my_executor, arg.arrays = list(data1 = my_values[['Train_img.array2']]), match.name = TRUE)
    mx.exec.forward(my_executor, is.train = TRUE)
    mx.exec.backward(my_executor, out_grads = gradient_2)
    # mx.exec.backward(my_executor, out_grads = mx.nd.array(gradient_2, ctx = mx.gpu(0)))
    grad_array_2 <- my_executor$grad.arrays
    #grad_array_2 <- my_executor$ref.grad.arrays
    
    # Sum gradient
    
    sum_grad <- list()
    for (k in 1:length(grad_array_1)){
      # sum_grad[[k]] <- grad_array_1[[k]] + grad_array_2[[k]]
      sum_grad[[k]] <- as.array(grad_array_1[[k]]) + as.array(grad_array_2[[k]])
      sum_grad[[k]] <- mx.nd.array(sum_grad[[k]], ctx = mx.gpu(1))
      names(sum_grad)[k] <- names(grad_array_1[k])
    }
    
    # sum_grad <- sum_grad[-which(names(sum_grad) == 'data1')]
    
    update_args <- my_updater(weight = my_executor$arg.arrays, grad = sum_grad)
    #update_args <- my_updater(weight = my_executor$ref.arg.arrays, grad = sum_grad)
    mx.exec.update.arg.arrays(my_executor, update_args, skip.null = TRUE)
    
    # Logging 
    
    batch_loss[[length(batch_loss) + 1]] <- as.array(loss_executor$ref.outputs[[1]])
    
    batch_time <- as.double(difftime(Sys.time(), batch_t0, units = "secs"))
    batch_speed <- length(batch_loss) * tail(input_dim[[1]], 1) / batch_time
    batch_result <- mean(unlist(batch_loss))
    message(paste0("Batch [", length(batch_loss),
                   "] Speed: ", formatC(batch_speed, 2, format = "f"),
                   " samples/sec Train-loss=", formatC(batch_result, 4, format = "f")))
    
  }
  
  message(paste0("Epoch [", epoch, "] Train-loss = ", formatC(mean(unlist(batch_loss)), format = "f", 4)))
  loss_report[epoch, 1] <- paste0("Epoch [", epoch, "] Train-loss = ", formatC(mean(unlist(batch_loss)), format = "f", 4))
  
  #Get model
  
  #my_model <- mxnet:::mx.model.extract.model(symbol = feature_symbol_1, train.execs = list(my_executor))
  #my_model[['arg.params']] <- append(my_model[['arg.params']], my_arg[['arg.params']][fixed_params])
  #my_model[['arg.params']] <- my_model[['arg.params']][!names(my_model[['arg.params']]) %in% "data1"]
  #mx.model.save(model = my_model, prefix = paste0('train model/banch size 32/train v', epoch), iteration = epoch)
  
  #return(my_model)
  
}
