library(magrittr)
library(imager)
library(OpenImageR)
library(jpeg)

# pairing

ifwnames <- read.csv('data/csv/lfw-names.csv', header = TRUE)

row_3up = which(ifwnames[,2] > 2)
row_equal2 = which(ifwnames[,2] == 2)

img_size = 72

# train_pos_pair, same identity, label 0

 # row_3up, person's number of image more than 3

train_pos_pair = list()
train_pos_pair[[1]] = list()
train_pos_pair[[2]] = list()

pb <- txtProgressBar(max = length(row_3up), style = 3)

for (i in 1:length(row_3up)) {
  
  file_names <- list.files(paste0("/home/joanna/Face/data/lfw-deepfunneled/", ifwnames[row_3up[i],1]))
  choose2 <- combn(1:ifwnames[row_3up[i],2],2)
  
  if(length(choose2[1,]) <= 20) {
    
    random_col <- sample(ncol(choose2), 20, replace = TRUE)
    choose2 <- choose2[,random_col]
    
    for (j in 1:length(choose2[1,])) {
      
      img <- readJPEG(paste0('data/lfw-deepfunneled/', ifwnames[row_3up[i],1], "/", file_names[choose2[1,j]]))
      img <- resizeImage(img, img_size, img_size, method = 'bilinear')
      train_pos_pair[[1]][[length(train_pos_pair[[1]])+1]] <- img
      
      img <- readJPEG(paste0('data/lfw-deepfunneled/', ifwnames[row_3up[i],1], "/", file_names[choose2[2,j]]))
      img <- resizeImage(img, img_size, img_size, method = 'bilinear')
      train_pos_pair[[2]][[length(train_pos_pair[[2]])+1]] <- img
      
    }
    
  }
  
  if(length(choose2[1,]) > 20) {
    
    sam_group <- sample(1:length(choose2[1,]), 20, replace = FALSE)
    choose2 <- choose2[,sam_group]
    
    for (j in 1:length(choose2[1,])) {
      
      img <- readJPEG(paste0('data/lfw-deepfunneled/', ifwnames[row_3up[i],1], "/", file_names[choose2[1,j]]))
      img <- resizeImage(img, img_size, img_size, method = 'bilinear')
      train_pos_pair[[1]][[length(train_pos_pair[[1]])+1]] <- img
      
      img <- readJPEG(paste0('data/lfw-deepfunneled/', ifwnames[row_3up[i],1], "/", file_names[choose2[2,j]]))
      img <- resizeImage(img, img_size, img_size, method = 'bilinear')
      train_pos_pair[[2]][[length(train_pos_pair[[2]])+1]] <- img
      
    }
    
  }
  
  setTxtProgressBar(pb, i)
  
}

close(pb)

i = 10033
imageShow(train_pos_pair[[1]][[i]])
imageShow(train_pos_pair[[2]][[i]])

# train_neg_pair, different identity, label 1

 # row_3up, person's number of image more than 3

train_neg_pair = list()
train_neg_pair[[1]] = list()
train_neg_pair[[2]] = list()

 # sample 2 person in row_3up
sam_group <- sample(length(combn(row_3up, 2)[1,]), length(train_pos_pair[[1]]), replace = FALSE)
choose2 <- combn(row_3up, 2)[,sam_group]

pb <- txtProgressBar(max = length(choose2[1,]), style = 3)

for (i in 1:length(train_pos_pair[[1]])) {
  
  # sample 1 image in person 1
  file_names <- list.files(paste0("/home/joanna/Face/data/lfw-deepfunneled/", ifwnames[choose2[1,i],1]))
  img <- readJPEG(paste0('data/lfw-deepfunneled/', ifwnames[choose2[1,i],1], "/", file_names[sample(1:length(file_names), 1)]))
  img <- resizeImage(img, img_size, img_size, method = 'bilinear')
  train_neg_pair[[1]][[length(train_neg_pair[[1]])+1]] <- img
  
  # sample 1 image in person 2
  file_names <- list.files(paste0("/home/joanna/Face/data/lfw-deepfunneled/", ifwnames[choose2[2,i],1]))
  img <- readJPEG(paste0('data/lfw-deepfunneled/', ifwnames[choose2[2,i],1], "/", file_names[sample(1:length(file_names), 1)]))
  img <- resizeImage(img, img_size, img_size, method = 'bilinear')
  train_neg_pair[[2]][[length(train_neg_pair[[2]])+1]] <- img
  
  setTxtProgressBar(pb, i)
}

close(pb)

i = 3939
imageShow(train_neg_pair[[1]][[i]])
imageShow(train_neg_pair[[2]][[i]])


# label train data, same identity label 0, different identity label 1

train_list = list()
train_list[[1]] = list()
train_list[[2]] = list()

Train_Y.array <- array(NA, dim = c(1, 1, 1, length(train_pos_pair[[1]])*2))

for (i in 1:length(train_pos_pair[[1]])) {
  
  train_list[[1]][[2*i-1]] <- train_pos_pair[[1]][[i]]
  train_list[[1]][[2*i]] <- train_neg_pair[[1]][[i]]
  
  train_list[[2]][[2*i-1]] <- train_pos_pair[[2]][[i]]
  train_list[[2]][[2*i]] <- train_neg_pair[[2]][[i]]
  
  Train_Y.array[,,,2*i-1] <- 0
  Train_Y.array[,,,2*i] <- 1
  
}


i=11111
imageShow(train_list[[1]][[i]])
imageShow(train_list[[2]][[i]])
Train_Y.array[,,,i]

names(train_list) <- c('person_1', 'person_2')

# save data

save(train_list, file = paste0('data/train_list_', img_size, '.RData'))
save(Train_Y.array, file = paste0('data/train_Y_', img_size, '.RData'))

#####


# valid_pos_pair, same identity, label 0

 # row_equal2, person's number of image equal 2

valid_pos_pair = list()
valid_pos_pair[[1]] = list()
valid_pos_pair[[2]] = list()

pb <- txtProgressBar(max = length(row_equal2), style = 3)

for (i in 1:length(row_equal2)) {
  
  file_names <- list.files(paste0("/home/joanna/Face/data/lfw-deepfunneled/", ifwnames[row_equal2[i],1]))
  choose2 <- combn(1:ifwnames[row_equal2[i],2],2)
    
    img <- readJPEG(paste0('data/lfw-deepfunneled/', ifwnames[row_equal2[i],1], "/", file_names[choose2[1,1]]))
    img <- resizeImage(img, img_size, img_size, method = 'bilinear')
    valid_pos_pair[[1]][[length(valid_pos_pair[[1]])+1]] <- img
    
    img <- readJPEG(paste0('data/lfw-deepfunneled/', ifwnames[row_equal2[i],1], "/", file_names[choose2[2,1]]))
    img <- resizeImage(img, img_size, img_size, method = 'bilinear')
    valid_pos_pair[[2]][[length(valid_pos_pair[[2]])+1]] <- img
  
  setTxtProgressBar(pb, i)
  
}

close(pb)

i=777
imageShow(valid_pos_pair[[1]][[i]])
imageShow(valid_pos_pair[[2]][[i]])


# valid_neg_pair, different identity, label 1

# row_equal2, person's number of image equal 2

valid_neg_pair = list()
valid_neg_pair[[1]] = list()
valid_neg_pair[[2]] = list()

pb <- txtProgressBar(max = length(row_equal2), style = 3)

# sample 2 person in row_equal2
sam_group <- sample(length(combn(row_equal2, 2)[1,]), length(valid_pos_pair[[1]]), replace = FALSE)
choose2 <- combn(row_equal2, 2)[,sam_group]

for (i in 1:length(choose2[1,])) {
  
  # sample 1 image in person 1
  file_names <- list.files(paste0("/home/joanna/Face/data/lfw-deepfunneled/", ifwnames[choose2[1,i],1]))
  img <- readJPEG(paste0('data/lfw-deepfunneled/', ifwnames[choose2[1,i],1], "/", file_names[sample(1:length(file_names), 1)]))
  img <- resizeImage(img, img_size, img_size, method = 'bilinear')
  valid_neg_pair[[1]][[length(valid_neg_pair[[1]])+1]] <- img
  
  # sample 1 image in person 2
  file_names <- list.files(paste0("/home/joanna/Face/data/lfw-deepfunneled/", ifwnames[choose2[2,i],1]))
  img <- readJPEG(paste0('data/lfw-deepfunneled/', ifwnames[choose2[2,i],1], "/", file_names[sample(1:length(file_names), 1)]))
  img <- resizeImage(img, img_size, img_size, method = 'bilinear')
  valid_neg_pair[[2]][[length(valid_neg_pair[[2]])+1]] <- img
  
  setTxtProgressBar(pb, i)
}

close(pb)

i=66
imageShow(valid_neg_pair[[1]][[i]])
imageShow(valid_neg_pair[[2]][[i]])


# label validation data, same identity label 0, different identity label 1

valid_list = list()
valid_list[[1]] = list()
valid_list[[2]] = list()

Valid_Y.array <- array(NA, dim = c(1, 1, 1, length(valid_pos_pair[[1]])*2))

for (i in 1:length(valid_pos_pair[[1]])) {
  
  valid_list[[1]][[2*i-1]] <- valid_pos_pair[[1]][[i]]
  valid_list[[1]][[2*i]] <- valid_neg_pair[[1]][[i]]
  
  valid_list[[2]][[2*i-1]] <- valid_pos_pair[[2]][[i]]
  valid_list[[2]][[2*i]] <- valid_neg_pair[[2]][[i]]
  
  Valid_Y.array[,,,2*i-1] <- 0
  Valid_Y.array[,,,2*i] <- 1
  
}


i=1111
imageShow(valid_list[[1]][[i]])
imageShow(valid_list[[2]][[i]])
Valid_Y.array[,,,i]

names(valid_list) <- c('person_1', 'person_2')

# save data

save(valid_list, file = paste0('data/valid_list_', img_size, '.RData'))
save(Valid_Y.array, file = paste0('data/valid_Y_', img_size, '.RData'))
