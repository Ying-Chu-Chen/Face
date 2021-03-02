library(magrittr)
library(imager)
library(OpenImageR)
library(jpeg)

# Train RData
# Read csv data

train_pair <- read.csv('data/csv/pairsDevTrain_old.csv', header = TRUE)

# Read image

img_size <- 72

train_img.array1 <- array(0, dim = c(img_size, img_size, 3, nrow(train_pair)))
train_img.array2 <- array(0, dim = c(img_size, img_size, 3, nrow(train_pair)))

# person1 img in train_img.array1

for (i in 1:nrow(train_pair)) {
  
  if (train_pair[i,2] < 10) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', train_pair[i,1], '/', train_pair[i,1], '_000', train_pair[i,2], '.jpg'))
  }
  
  if (10 <= train_pair[i,2] & train_pair[i,2] < 100) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', train_pair[i,1], '/', train_pair[i,1], '_00', train_pair[i,2], '.jpg'))
  }
  
  if (100 <= train_pair[i,2]) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', train_pair[i,1], '/', train_pair[i,1], '_0', train_pair[i,2], '.jpg'))
  }
  
  #train_img.array1[,,,i] <- img
  train_img.array1[,,,i] <- resizeImage(img, img_size, img_size, method = 'bilinear')
}


# person2 img in train_img.array2

for (i in 1:nrow(train_pair)) {
  
  if (train_pair[i,4] < 10) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', train_pair[i,3], '/', train_pair[i,3], '_000', train_pair[i,4], '.jpg'))
  }
  
  if (10 <= train_pair[i,4] & train_pair[i,4] < 100) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', train_pair[i,3], '/', train_pair[i,3], '_00', train_pair[i,4], '.jpg'))
  }
  
  if (100 <= train_pair[i,4]) {
    img <- readJPEG(paste0('data/lfw-deepfunneled', '/', train_pair[i,3], '/', train_pair[i,3], '_0', train_pair[i,4], '.jpg'))
  }
  
  #train_img.array2[,,,i] <- img
  train_img.array2[,,,i] <- resizeImage(img, img_size, img_size, method = 'bilinear')
}

#check image

i=1100

imageShow(train_img.array1[,,,i])
imageShow(train_img.array2[,,,i])

# data normalization

for (i in 1:nrow(train_pair)) {
  train_img.array1[,,,i] = (train_img.array1[,,,i] - mean(train_img.array1[,,,i]))/sd(train_img.array1[,,,i])
  train_img.array2[,,,i] = (train_img.array2[,,,i] - mean(train_img.array2[,,,i]))/sd(train_img.array2[,,,i])
}

# train data label
# 同人label 1，不同人 label 0

train_Y.array <- array(NA, dim = c(1, 1, 1, nrow(train_pair)))

train_Y.array[,,,1:(nrow(train_pair)/2)] <- rep(1)
train_Y.array[,,,(nrow(train_pair)/2+1):nrow(train_pair)] <- rep(0)

# separate train data and validation data
# 同人label 1 奇數，不同人 label 0 偶數

set.seed(1234)
train.seq_pos = sample(1:(nrow(train_pair)/2), 1000)
train.seq_neg = sample(1101:nrow(train_pair), 1000)

train.seq <- 0

for (i in 1:1000) {
  
  train.seq[i*2-1] = train.seq_pos[i]
  train.seq[i*2] = train.seq_neg[i]
  
}

Train_img.array1 = train_img.array1[,,,train.seq]
Train_img.array2 = train_img.array2[,,,train.seq]
Train_Y.array = train_Y.array[,,,train.seq]

Valid_img.array1 = train_img.array1[,,,-train.seq]
Valid_img.array2 = train_img.array2[,,,-train.seq]
Valid_Y.array = train_Y.array[,,,-train.seq]

#check image
# 同人label 1，不同人 label 0

i=122
Valid_Y.array[i]
imageShow(Valid_img.array1[,,,i])
imageShow(Valid_img.array2[,,,i])

# train list

train_list <- list()
train_list[[1]] <- list()
train_list[[2]] <- list()

pb <- txtProgressBar(max = 2000, style = 3)

for (i in 1:2000) {
  
  train_list[[1]][[i]] <- Train_img.array1[,,,i]
  train_list[[2]][[i]] <- Train_img.array2[,,,i]
  
  setTxtProgressBar(pb, i)
}

close(pb)

names(train_list) <- c('person_1', 'person_2')

# check image
i=7
Train_Y.array[i]
imageShow(train_list[[1]][[i]])
imageShow(train_list[[2]][[i]])

# validation list

valid_list <- list()
valid_list[[1]] <- list()
valid_list[[2]] <- list()

pb <- txtProgressBar(max = length(Valid_Y.array), style = 3)

for (i in 1:length(Valid_Y.array)) {
  
  valid_list[[1]][[i]] <- Valid_img.array1[,,,i]
  valid_list[[2]][[i]] <- Valid_img.array2[,,,i]
  
  setTxtProgressBar(pb, i)
}

close(pb)

# check image
i=200
Valid_Y.array[i]
imageShow(valid_list[[1]][[i]])
imageShow(valid_list[[2]][[i]])

names(valid_list) <- c('person_1', 'person_2')

# save data

save(train_list, file = paste0('data/train_list_', img_size, '.RData'))
save(Train_Y.array, file = paste0('data/train_Y_', img_size, '.RData'))

save(valid_list, file = paste0('data/valid_list_', img_size, '.RData'))
save(Valid_Y.array, file = paste0('data/valid_Y_', img_size, '.RData'))


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
