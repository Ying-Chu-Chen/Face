library(magrittr)
library(imager)
library(OpenImageR)
library(jpeg)

# Train RData
# Read csv data

train.data <- read.csv('data/csv/pairsDevTrain_old.csv', header = TRUE)

# Read image

img_size <- 256

Train_img.array1 <- array(0, dim = c(img_size, img_size, 3, nrow(train.data)))
Train_img.array2 <- array(0, dim = c(img_size, img_size, 3, nrow(train.data)))

# person1 img in Train_img.array1

for (i in 1:nrow(train.data)) {
  
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
  Train_img.array1[,,,i] <- resizeImage(img, img_size, img_size, method = 'bilinear')
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
  Train_img.array2[,,,i] <- resizeImage(img, img_size, img_size, method = 'bilinear')
}


# train list

train_list <- list()
train_list[[1]] <- list()
train_list[[2]] <- list()

#奇數同人，偶數不同人

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

save(train_list, file = paste0('data/train_list_', img_size, '.RData'))


#############################################

# Test RData

# Read csv data

# test.match_data <- read.csv('data/csv/matchpairsDevTest.csv', header = TRUE)
# test.match_data <- cbind(test.match_data[,1:2], test.match_data[,c(1,3)])
# colnames(test.match_data) <- c('person1', 'image_number1', 'person2', 'image_number2')
# 
# test.mismatch_data <- read.csv('data/csv/mismatchpairsDevTest.csv', header = TRUE)
# colnames(test.mismatch_data) <- c('person1', 'image_number1', 'person2', 'image_number2')
# 
# test_data <- rbind(test.match_data, test.mismatch_data)
# write.csv(test_data, file = 'data/csv/pairsDevTest.csv')

test_data <- read.csv('data/csv/pairsDevTest.csv', header = TRUE)
test_data <- test_data[,-1]

# Read image

img_size <- 64

Test_img.array1 <- array(0, dim = c(img_size, img_size, 3, nrow(test_data)))
Test_img.array2 <- array(0, dim = c(img_size, img_size, 3, nrow(test_data)))

# person1 img in Test_img.array1

for (i in 1:nrow(test_data)) {
  
  if (test_data[i,2] < 10) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', test_data[i,1], '/', test_data[i,1], '_000', test_data[i,2], '.jpg'))
  }
  
  if (10 <= test_data[i,2] & test_data[i,2] < 100) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', test_data[i,1], '/', test_data[i,1], '_00', test_data[i,2], '.jpg'))
  }
  
  if (100 <= test_data[i,2]) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', test_data[i,1], '/', test_data[i,1], '_0', test_data[i,2], '.jpg'))
  }
  
  #Test_img.array1[,,,i] <- img
  Test_img.array1[,,,i] <- resizeImage(img, img_size, img_size, method = 'bilinear')
}


# person2 img in Test_img.array2

for (i in 1:nrow(test_data)) {
  
  if (test_data[i,4] < 10) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', test_data[i,3], '/', test_data[i,3], '_000', test_data[i,4], '.jpg'))
  }
  
  if (10 <= test_data[i,4] & test_data[i,4] < 100) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', test_data[i,3], '/', test_data[i,3], '_00', test_data[i,4], '.jpg'))
  }
  
  if (100 <= test_data[i,4]) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', test_data[i,3], '/', test_data[i,3], '_0', test_data[i,4], '.jpg'))
  }
  
  #Test_img.array2[,,,i] <- img
  Test_img.array2[,,,i] <- resizeImage(img, img_size, img_size, method = 'bilinear')
}


# test list

test_list <- list()
test_list[[1]] <- list()
test_list[[2]] <- list()

#奇數同人，偶數不同人

pb <- txtProgressBar(max = nrow(test_data)/2, style = 3)

for (i in 1:(nrow(test_data)/2)) {
  
  test_list[[1]][[2*i-1]] <- Test_img.array1[,,,i]
  test_list[[1]][[2*i]] <- Test_img.array1[,,,nrow(test_data)/2+i]
  test_list[[2]][[2*i-1]] <- Test_img.array2[,,,i]
  test_list[[2]][[2*i]] <- Test_img.array2[,,,nrow(test_data)/2+i]
  
  setTxtProgressBar(pb, i)
}

close(pb)

#imageShow(test_list[[1]][[501]])

names(test_list) <- c('person_1', 'person_2')

save(test_list, file = paste0('data/test_list_', img_size, '.RData'))
