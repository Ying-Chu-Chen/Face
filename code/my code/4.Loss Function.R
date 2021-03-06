#Loss Function

# fi = identity 1, fj = identity 2, yij = same or different identities, m = parameter to control distance of fi to fj

verification_loss_2 <- function (person_1, person_2, label, m = 1) {
  
  verif.fi_fj <- mx.symbol.broadcast_minus(lhs = person_1, rhs = person_2, name = "verif.fi_fj")
  verif.square_fi_fj <- mx.symbol.square(data = verif.fi_fj, name = "verif.square_fi_fj")
  verif.sum_square <- mx.symbol.sum(data = verif.square_fi_fj, axis = c(3,2,1), keepdims = TRUE, name = "verif.sum_square")       
  verif.distance <- mx.symbol.sqrt(data = verif.sum_square, name = "verif.distance")
  
  # Same identities
  
  verif.square_distance <- mx.symbol.square(data = verif.distance, name = "verif.square_distance")
  verif.loss_sameid <- mx.symbol.broadcast_mul(lhs = verif.square_distance, rhs = 1 - label, name = "verif.loss_sameid")
  verif.loss_sameid.half <- verif.loss_sameid * 0.5
  
  # Different identities
  
  verif.m_minus_distance <- m - verif.distance
  verif.max_differentid <- mx.symbol.relu(data = verif.m_minus_distance, name = "verif.max_differentid")
  verif.square_max_differentid <- mx.symbol.square(data = verif.max_differentid, name = "verif.square_max_differentid")
  verif.loss_differentid <- mx.symbol.broadcast_mul(lhs = verif.square_max_differentid, rhs = label, name = "verif_loss_differentid")
  verif.loss_differentid.half <- verif.loss_differentid * 0.5
  
  #Sum & mean
  
  verif.sum_loss <- verif.loss_sameid.half + verif.loss_differentid.half
  verif.mean_loss <- mx.symbol.mean(data = verif.sum_loss, axis = c(2,1,0), keepdims = FALSE,name = 'verif.mean_loss')
  
  verification_loss <- mx.symbol.MakeLoss(data = verif.mean_loss, name = "verification_loss")
  
  return(verification_loss)
}

verification_loss <- function (person_1, person_2, label, 
                               pos_lambda = 2, m_para = 2.5, lambda = 1) {
  
  # According to DeepID2, verification loss function
  
  diff_indiv <- mx.symbol.broadcast_minus(lhs = person_1, rhs = person_2, name = "verif_diff_indiv")
  square_diff <- mx.symbol.square(data = diff_indiv, name = "verif_square_diff")
  sum_square <- mx.symbol.sum(data = square_diff, axis = 1, keepdims = TRUE,
                              name = "verif_sum_square")       
  norm_indiv <- mx.symbol.sqrt(data = sum_square, name = "verif_norm_indiv")
  
  # Postive
  #pos_m_norm <- 1e-8 - norm_indiv
  #max_pos <- mx.symbol.relu(data = pos_m_norm, name = "verif_max_pos")
  
  square_max_pos <- mx.symbol.square(data = norm_indiv, name = "verif_square_max_pos")
  loss_pos <- mx.symbol.broadcast_mul(lhs = square_max_pos, rhs = 1 - label, name = "verif_loss_pos")
  loss_pos_half <- loss_pos * 0.5
  
  # Negative
  neg_m_norm <- m_para - norm_indiv
  max_neg <- mx.symbol.relu(data = neg_m_norm, name = "verif_max_neg")
  
  square_max_neg <- mx.symbol.square(data = max_neg, name = "verif_square_max_neg")
  loss_neg <- mx.symbol.broadcast_mul(lhs = square_max_neg, rhs = label, name = "verif_loss_neg")
  loss_neg_half <- loss_neg * 0.5
  
  sum_loss <- loss_pos_half * pos_lambda + loss_neg_half
  
  loss_mean <- mx.symbol.mean(data = sum_loss, axis = 0, keepdims = FALSE,
                              name = 'verif_loss_mean')
  
  weighted_loss_mean <- loss_mean * lambda
  loss_out <- mx.symbol.MakeLoss(data = weighted_loss_mean, name = "verif_loss")
  
  return(loss_out)
}

#################

feature_symbol_1 <- featrue_out

loss_name <- list(person_1_name = 'person_1', person_2_name = 'person_2', label_name = 'label')

person_1 <- mx.symbol.Variable(loss_name$person_1_name)
person_2 <- mx.symbol.Variable(loss_name$person_2_name)
label <- mx.symbol.Variable(loss_name$label_name)

#dis_loss <- verification_loss(person_1 = person_1, person_2 = person_2, label = label, pos_lambda = 2, m_para = 2.5, lambda = 1)
dis_loss <- verification_loss_2(person_1 = person_1, person_2 = person_2, label = label)

mx.symbol.infer.shape(dis_loss, person_1 = c(2,2,320,32), person_2 = c(2,2,320,32), label = c(1,1,1,32))$out.shapes
#return(list(dis_loss = dis_loss, loss_name = loss_name))

###########################

#check

#mx.symbol.infer.shape(label1, data = c(128, 7), label = c(1, 7))
#mx.symbol.infer.shape(label1, data = c(128, 7), label = c(1, 7))
#mx.symbol.infer.shape(loss, data = c(1, 7), label = c(1, 7))