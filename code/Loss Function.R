#Loss Function

# fi = identity 1, fj = identity 2, yij = same or different identities, m = parameter to control distance of fi to fj

verification_loss <- function (person_1, person_2, label, m = 1) {
  
  verif.fi_fj <- mx.symbol.broadcast_minus(lhs = person_1, rhs = person_2, name = "verif.fi_fj")
  verif.square_fi_fj <- mx.symbol.square(data = verif.fi_fj, name = "verif.square_fi_fj")
  verif.sum_square <- mx.symbol.sum(data = verif.square_fi_fj, axis = 1, keepdims = TRUE, name = "verif.sum_square")       
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
  verif.mean_loss <- mx.symbol.mean(data = verif.sum_loss, axis = 0, keepdims = FALSE,name = 'verif.mean_loss')
  
  verification_loss <- mx.symbol.MakeLoss(data = verif.mean_loss, name = "verification_loss")
  
  return(verification_loss)
}

#################

feature_symbol_1 <- feature_symbol(name = '1')

loss_name <- list(person_1_name = 'person_1', person_2_name = 'person_2', label_name = 'label')

person_1 <- mx.symbol.Variable(loss_name$person_1_name)
person_2 <- mx.symbol.Variable(loss_name$person_2_name)
label <- mx.symbol.Variable(loss_name$label_name)

dis_loss <- verification_loss(person_1 = person_1, person_2 = person_2, label = label, m = 1)

#return(list(dis_loss = dis_loss, loss_name = loss_name))

###################

# f = DeepID2 vector, label = target class, weight = softmax layer parameters 

# Ident <- function (f, label, weight, eps = 1e-8) {
#   
#   label1 <- mx.symbol.broadcast_mul(lhs = mx.symbol.log(data = f + eps), rhs = label, name = 'label1')
#   label2 <- mx.symbol.broadcast_mul(lhs = mx.symbol.log(data = 1 - f + eps), rhs = 1 - label, name = 'label2')
#   
#   average1 <- mx.symbol.mean(data = label1, axis = 0, keepdims = FALSE, name = 'average1')
#   average2 <- mx.symbol.mean(data = label2, axis = 0, keepdims = FALSE, name = 'average2')
#   
#   loss <- (0 - average1 - average2) 
#   CE_loss <- mx.symbol.MakeLoss(data = loss, name = 'CE_loss')
#   
#   return(CE_loss)
# }

###########################

#check

#mx.symbol.infer.shape(label1, data = c(128, 7), label = c(1, 7))
#mx.symbol.infer.shape(label1, data = c(128, 7), label = c(1, 7))
#mx.symbol.infer.shape(loss, data = c(1, 7), label = c(1, 7))