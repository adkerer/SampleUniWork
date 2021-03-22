learn <- function(hist){
  
  #   bayesian_network = list(pneumonia_matrix, visited_tb_matrix, smokes_matrix, 
  #                        temperature_distribution_matrix, tuberculosis_matrix, lung_cancer_matrix, 
  #                       bronchitis_matrix, xray_matrix, dyspnea_matrix)
  bayesian_network = create_start_matrices() #see order of matrices in comment above, due to faster to create
  bayesian_network = fill_last_matrices(bayesian_network)
  
  #returns bayesian network in topological order as in hist
  topological_bayesian_network = list(bayesian_network[[1]],bayesian_network[[4]],bayesian_network[[2]],bayesian_network[[5]],
                                      bayesian_network[[3]],bayesian_network[[6]],bayesian_network[[7]],
                                      bayesian_network[[8]],bayesian_network[[9]])
  return(topological_bayesian_network)
}

create_start_matrices <- function(){
  
  pneumonia_matrix = matrix(0,1,2)
  visited_tb_matrix = matrix(0,1,2)
  smokes_matrix = matrix(0,1,2)
  temperature_distribution_matrix = matrix(0,2,3)
  tuberculosis_matrix = matrix(0,2,3)
  lung_cancer_matrix = matrix(0,2,3)
  bronchitis_matrix = matrix(0,2,3)
  xray_matrix = matrix(0,8,5)
  dyspnea_matrix = matrix(0,4,4)
  
  pneumonia_matrix[1,2] = mean(hist[,1]) 
  visited_tb_matrix[1,2] = mean(hist[,3])
  smokes_matrix[1,2] = mean(hist[,5])
  
  
  bayesian_network = list(pneumonia_matrix, visited_tb_matrix, smokes_matrix, 
                          temperature_distribution_matrix, tuberculosis_matrix, lung_cancer_matrix, 
                          bronchitis_matrix, xray_matrix, dyspnea_matrix)
  counter = 1
  for (i in bayesian_network){
    if(counter <4){
      bayesian_network[[counter]][1,1] = 1-i[1,2] #prob for first 3 diseases based on earlier results
    }
    else if (counter<8){
      bayesian_network[[counter]][2,1] = 1 #populates w. 1 for upper sickness/symptom
    }
    else if (counter<9){ #populates xray_matrix 1 for upper sickness/symptom
      for(j in (1:nrow(bayesian_network[[counter]]))){ 
        if (j==2 | j == 4 | j == 6 | j ==8){bayesian_network[[counter]][j,1] = 1}
        if (j ==3 | j == 4 | j>=7){bayesian_network[[counter]][j,2] = 1}
        if (j >= 5){bayesian_network[[counter]][j,3] = 1}
      }
    }
    else{ #populates dyspnea_matrix w. 1 for upper sickness/symptom
      for(j in (1:nrow(bayesian_network[[counter]]))){
        if (j==2 | j == 4){bayesian_network[[counter]][j,1] = 1}
        if (j ==3 | j == 4 ){bayesian_network[[counter]][j,2] = 1}
      }
    }
    counter = counter +1
  }
  
  return (bayesian_network)
  #print(bayesian_network)
}

fill_last_matrices <- function(bayesian_network){
  for (i in (4:9)){
    if (i == 4){ #temperature_matrix
      #calculates mean & sd of the normally distributed temperature
      bayesian_network[[i]][1,2] = mean(hist[which(hist[,1]==0),2])
      bayesian_network[[i]][1,3] = sd(hist[which(hist[,1]==0),2])
      bayesian_network[[i]][2,2] = mean(hist[which(hist[,1]==1),2])
      bayesian_network[[i]][2,3] = sd(hist[which(hist[,1]==1),2])
    }#temperature
    else if (i == 5){
      bayesian_network[[i]][1,3] = mean(hist[which(hist[,3]==0),4])
      bayesian_network[[i]][2,3] = mean(hist[which(hist[,3]==1),4])
      bayesian_network[[i]][1,2] = 1 - bayesian_network[[i]][1,3]
      bayesian_network[[i]][2,2] = 1 - bayesian_network[[i]][2,3]
    }#tuberculosis
    else if (i == 6){
      bayesian_network[[i]][1,3] = mean(hist[which(hist[,5]==0),6])
      bayesian_network[[i]][2,3] = mean(hist[which(hist[,5]==1),6])
      bayesian_network[[i]][1,2] = 1 - bayesian_network[[i]][1,3]
      bayesian_network[[i]][2,2] = 1 - bayesian_network[[i]][2,3]
    }#lung_cancer
    else if (i == 7){
      bayesian_network[[i]][1,3] = mean(hist[which(hist[,5]==0),7])
      bayesian_network[[i]][2,3] = mean(hist[which(hist[,5]==1),7])
      bayesian_network[[i]][1,2] = 1 - bayesian_network[[i]][1,3]
      bayesian_network[[i]][2,2] = 1 - bayesian_network[[i]][2,3]
    }#bronchitis
    else if (i == 8){
      dirichlet_hist = hist
      dirichlet_hist[10001,] =c(1,bayesian_network[[i]][2,2],1,1,1,1,1,0,1)
      dirichlet_hist[10002,] =c(1,bayesian_network[[i]][2,2],1,1,1,1,1,1,1) 
      bayesian_network[[i]][1,5] = mean(dirichlet_hist[which(dirichlet_hist[,1]==0 & dirichlet_hist[,4]==0 & dirichlet_hist[,6]==0),8])
      bayesian_network[[i]][2,5] = mean(dirichlet_hist[which(dirichlet_hist[,1]==1 & dirichlet_hist[,4]==0 & dirichlet_hist[,6]==0),8])
      bayesian_network[[i]][3,5] = mean(dirichlet_hist[which(dirichlet_hist[,1]==0 & dirichlet_hist[,4]==1 & dirichlet_hist[,6]==0),8])
      bayesian_network[[i]][4,5] = mean(dirichlet_hist[which(dirichlet_hist[,1]==1 & dirichlet_hist[,4]==1 & dirichlet_hist[,6]==0),8])
      bayesian_network[[i]][5,5] = mean(dirichlet_hist[which(dirichlet_hist[,1]==0 & dirichlet_hist[,4]==0 & dirichlet_hist[,6]==1),8])
      bayesian_network[[i]][6,5] = mean(dirichlet_hist[which(dirichlet_hist[,1]==1 & dirichlet_hist[,4]==0 & dirichlet_hist[,6]==1),8])
      bayesian_network[[i]][7,5] = mean(dirichlet_hist[which(dirichlet_hist[,1]==0 & dirichlet_hist[,4]==1 & dirichlet_hist[,6]==1),8])
      bayesian_network[[i]][8,5] = mean(dirichlet_hist[which(dirichlet_hist[,1]==1 & dirichlet_hist[,4]==1 & dirichlet_hist[,6]==1),8])
      for (j in (1:nrow(bayesian_network[[i]]))){
        bayesian_network[[i]][j,4] = 1 - bayesian_network[[i]][j,5]
      }
    }#x-ray
    else if (i == 9){
      bayesian_network[[i]][1,4] = mean(hist[which(hist[,6]==0 & hist[,7]==0),9])
      bayesian_network[[i]][2,4] = mean(hist[which(hist[,6]==1 & hist[,7]==0),9])
      bayesian_network[[i]][3,4] = mean(hist[which(hist[,6]==0 & hist[,7]==1),9])
      bayesian_network[[i]][4,4] = mean(hist[which(hist[,6]==1 & hist[,7]==1),9])
      for (j in (1:nrow(bayesian_network[[i]]))){
        bayesian_network[[i]][j,3] = 1 - bayesian_network[[i]][j,4]
      }
    }#dyspnea
    
  }
  return (bayesian_network)
}

diagnose <- function(network, cases) {
  #inputs are cases and the network of probabilties
  #Pn, TB, LC and Br are seeked for 10 different case
  #For every case (1-10)
  # (for 1 to 1000)
  # First, randomly assign values to Pn, TB, LC and Br
  # Then, consecutively, for every of the unknwon values, use flip function to create a new U
  # Compare the old (un-flipped) and the new values, and create a probabilty of changing the sample to the new
  # if p_new > p_old -> use new state
  # if p_new < p_old -> use new state with a p_new / p_old probability (sample from uniform random)
  # After all values are set a sample is created
  # Use the previous sample as the start state and iterate 999 times
  #Given the 900 samples ( 100 first removed due to burn-in), 
  #find probabilities of Pn, TB and LC being 0 or 1 by tally
  #and decide the final state for current case (1-10)
  #QUESTION: should the final state probabilities also be calculated in topological order?
  #          The cases are conditionally independent from each other so it shouldn't matter. 
  #          But is TB chosen given Te, VTB, SM, XR, Dy AND Pn if Pn is already set?
  #Pn, TB, LC and Br are seeked for 10 different case
  diseases <- c('Pn', 'TB', 'LC', 'Br')
  predictions <- cases[diseases] #copy cases to predictions
  for (row in 1:nrow(predictions)){
    #10 cases
    initial_state <- cases[row,]
    initial_state$Pn <- round(runif(1,0,1))
    initial_state$TB <- round(runif(1,0,1))
    initial_state$LC <- round(runif(1,0,1))
    initial_state$Br <- round(runif(1,0,1)) #randomly assign values 0 or 1 to NA-values
    
    samples <- 2000
    samples_df <- data.frame('Pn'=c(),'Te'=c(),'VTB' = c(), 'TB' = c(), 'Sm' = c(), 'LC'=c(), 'Br'=c(),'XR'=c(),'Dy'=c())
    prev_sample <- initial_state #the first previous sample is the initial state. After this, the previous sample is the previous sample. 
    for (i in 1:samples){
      #1000 samples
      prev_state <- prev_sample
      for(dis in diseases){
        #flip every disease one by one
        new_state <- prev_state #create copy
        new_state[dis] <- flip(prev_state[dis]) #and flip one value
        #calculate probs
        prev_state <- which_state(new_state,prev_state,network) #save the new state as next run's previous state 
      }
      new_sample <- prev_state #the final state / new sample is the last state recorded after flipping for-loop
      samples_df <- rbind(samples_df,new_sample) #SAVE the sample, in dataframe
      prev_sample <- new_sample #new sample is next iterations previous sample
    }
    #after samples are recorded
    #get prediction (for every case 1-10)
    prediction <- get_prediction(samples_df,cases[row,])
    predictions[row, ] <- prediction
  }
  
  return(data.matrix(predictions))
}

flip <- function(int){
  if(int == 0){return(1)}
  else{return(0)}
}

which_state <- function(new_state,prev_state,network){
  p_new <- get_probability(new_state, network)
  p_prev <- get_probability(prev_state, network)
  if(p_new > p_prev){return(new_state)}
  else{
    prob_threshold <- p_new/p_prev 
    if(prob_threshold > 1){print('!')} #density value /dnorm check
    random_prob <- runif(1,0,1)
    if(random_prob <= prob_threshold){return(new_state)}#if the random number is under the threshhold
    else {return(prev_state)}
  }
}

get_prediction <- function(samples_df,initial_case){
  burn_in <- 0.1*nrow(samples_df) 
  burned_in_df <- samples_df[burn_in:nrow(samples_df),]
  diseases <- c('Pn', 'TB', 'LC', 'Br')
  prediction <- initial_case[diseases] #fulkod - behöver egentligen inte initial_case...
  
  for(dis in diseases){
    count_0 <- length(burned_in_df[dis][burned_in_df[dis] == 0])
    count_1 <- length(burned_in_df[dis][burned_in_df[dis] == 1])
    #prob_0 <- count_0/(count_0+count_1)
    prob_1 <- count_1/(count_0+count_1)#get probabilites
    prediction[dis] <- prob_1
  }
  return(prediction)
}

get_probability <- function(state, network){
  #given a state, calculate the probability of the state
  prob_BN = 1
  #Pn Te VTB TB Sm LC Br XR Dy
  for (i in 1:length(state)){
    matrix_probs <- network[[i]]
    if(length(matrix_probs) == 2){
      prob_iter <- matrix_probs[,state[[i]] + 1]
    }
    else if(length(matrix_probs) == 6){
      if(i == 2){
        #temperature case, given Pn
        row <- matrix_probs[state$Pn+1,]
        prob_iter <- dnorm(state$Te,row[[2]],row[[3]]) # row[[2]],row[[3]] = mean,sd
      }
      else if (i == 4){
        #Tuberculosis, given VTB
        row <- matrix_probs[state$VTB+1,] #+1 offset to get correct row (0 is at row 1, and 1 is at row 2)
        prob_iter <- row[state[[i]] + 2]
      }
      else{
        #Lung cancer, Bronchitis given Smokes
        row <- matrix_probs[state$Sm+1,]
        prob_iter <- row[state[[i]] + 2]
      }
    }
    else if(length(matrix_probs) == 16){
      #Dyspnea, given bronchitis and lung cancer
      row_index <- which(matrix_probs[,1]==state$LC & matrix_probs[,2]==state$Br)
      row <- matrix_probs[row_index,]
      prob_iter <- row[state[[i]] + 3] #3 offset 
    }
    else if(length(matrix_probs) == 40){
      #X-ray, given TB PN LC
      row_index <- which(matrix_probs[,1]==state$Pn & matrix_probs[,2]==state$TB & matrix_probs[,3]==state$LC)
      row <- matrix_probs[row_index,]
      prob_iter <- row[state[[i]] + 4] #offset
    }
    prob_BN = prob_BN * prob_iter 
  }
  return(prob_BN)
}

