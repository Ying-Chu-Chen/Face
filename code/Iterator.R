library(magrittr)
library(imager)
library(OpenImageR)
library(jpeg)

# Load data

img_size <- 256

load(paste0('data/train_list_', img_size, '.RData'))

Train_Y.array <- array(NA, dim = c(1, 1, 1, nrow(train.data)))

#奇數同人label 0，偶數不同人 label 1
for (i in 1:1100) {
  Train_Y.array[,,,2*i-1] <- rep(0)
  Train_Y.array[,,,2*i] <- rep(1)
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
    
    Train_img.array1 <- mx.nd.array(Train_img.array1)
    Train_img.array2 <- mx.nd.array(Train_img.array2)
    
    label = mx.nd.array(Train_Y.array[,,,idx])
    
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
#imageShow(train_list[[2]][[2199]])