library(magrittr)
library(imager)
library(OpenImageR)
library(jpeg)

# Read csv data

train.data <- read.csv('data/csv/pairsDevTrain.csv', header = TRUE)
class(train.data)

# Read image

Train_img.array1 <- array(0, dim = c(128, 128, 3, 2200))
Train_img.array2 <- array(0, dim = c(128, 128, 3, 2200))

# person1 img in Train_img.array1

for (i in 1:2200) {
  
  if (train.data[i,2] < 10) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', train.data[i,1], '/', train.data[i,1], '_000', train.data[i,2], '.jpg'))
  }
  
  if (10 <= train.data[i,2] & train.data[i,2] < 100) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', train.data[i,1], '/', train.data[i,1], '_00', train.data[i,2], '.jpg'))
  }
  
  if (100 <= train.data[i,2]) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', train.data[i,1], '/', train.data[i,1], '_0', train.data[i,2], '.jpg'))
  }
  
  #Train_img.array1[,,,i] <- img
  Train_img.array1[,,,i] <- resizeImage(img, 128, 128, method = 'bilinear')
}


# person2 img in Train_img.array2

for (i in 1:2200) {
  
  if (train.data[i,4] < 10) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', train.data[i,3], '/', train.data[i,3], '_000', train.data[i,4], '.jpg'))
  }
  
  if (10 <= train.data[i,4] & train.data[i,4] < 100) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', train.data[i,3], '/', train.data[i,3], '_00', train.data[i,4], '.jpg'))
  }
  
  if (100 <= train.data[i,4]) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', train.data[i,3], '/', train.data[i,3], '_0', train.data[i,4], '.jpg'))
  }
  
  #Train_img.array2[,,,i] <- img
  Train_img.array2[,,,i] <- resizeImage(img, 128, 128, method = 'bilinear')
}


# train list

train_list <- list()
train_list[[1]] <- list()
train_list[[2]] <- list()

pb <- txtProgressBar(max = 1100, style = 3)

for (i in 1:1100) {
  
  train_list[[1]][[2*i-1]] <- Train_img.array1[,,,i]
  train_list[[1]][[2*i]] <- Train_img.array1[,,,1100+i]
  train_list[[2]][[2*i-1]] <- Train_img.array2[,,,i]
  train_list[[2]][[2*i]] <- Train_img.array2[,,,1100+i]
  
  setTxtProgressBar(pb, i)
}

close(pb)

#imageShow(train_list[[2]][[4]])

names(train_list) <- c('person_1', 'person_2')

save(train_list, file = "data/train_list_128.RData")