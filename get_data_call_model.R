##########################################################
##########################################################
# MovieLens Recommendation system (HarvardX project submission - PH125.9x Data Science: Capstone)
# Author: Martin Haitzmann
# eMail: martin.c.haitzmann@gmail.com or m.haitzmann@unido.org

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################
##setwd if necessary
#setwd("./capstone_movielens")

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(chron)) install.packages("chron", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# download or use already downloaded and saved data 
DL <- FALSE

if (DL) {
  dl <- tempfile()
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
}

ratings <- fread(text = gsub("::", "\t", readLines("ml-10M100K/ratings.dat")),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines("ml-10M100K/movies.dat"), "\\::", 3)

colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
#                                            title = as.character(title),
#                                            genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##########################################################
# Code for data analysis is in capstone_movielens_report.Rmd
##########################################################

##########################################################
# Function for the models 
##########################################################
# !!!!!!!!!!!IMPORTANT!!!!!!!!!!
# (below the function is the call for the final model)
# only call if call_*_models_methods is set to TRUE
# for the report the function is called from Rmd File
##########################################################
call_prelim_models_methods <- FALSE
call_final_models_methods <- TRUE

# data models/methods for recommendation systen
# define as a function
# develop with edx only (split into train_set and test_set)
# for final reuslts apply to edx as train_set and validation as test_set
######################################
models_methods <- function(final = FALSE, final_lambda = 4.2) { 
  # it foreseen to compare models according to RMSE
  ####
  RMSE <- function(true_ratings, predicted_ratings){
    sqrt(mean((true_ratings - predicted_ratings)^2))
  }
  ####
  
  # final <- FALSE: Create different models on edx data set (split to test_set and train set)
  # final <- TRUE :  use the created models on edx as train_set and validation as test_set
  ####
  #final <- F
  if(final) cat("final run: edx is used as train_set, validation as test_set\n") else cat("model development: edx is split into train_set and test_set\n")
  ####
  if (final) {
    train_set <- edx
    test_set <- validation
  } else {
    set.seed(755, sample.kind = "Rounding")
    test_index <- createDataPartition(y = edx$rating, times = 1,
                                      p = 0.2, list = FALSE)
    train_set <- edx[-test_index,]
    test_set <- edx[test_index,]
    test_set <- test_set %>% 
      semi_join(train_set, by = "movieId") %>%
      semi_join(train_set, by = "userId")
  }
  
  ###########
  ###########
  # add variables to test and train set that might have an influence as shown in the explanatory analysis
  # calendar variables
  
  # calendar and daytime variables
  ##
  train_set <- train_set %>% mutate(
    r_y = year(as_datetime(timestamp)),
    r_m = month(as_datetime(timestamp)),
    r_q = quarter(as_datetime(timestamp)),
    r_h = chron::is.weekend(strftime(as_datetime(timestamp), format = "%m/%d/%y")),
    r_t = cut(x = lubridate::hour(as_datetime(timestamp)), 
              breaks = lubridate::hour(hm("00:00", "6:00", "12:00", "18:00", "23:59")), 
              labels = c("Night", "Morning", "Afternoon", "Evening"), 
              include.lowest=TRUE)
  )  
  test_set <- test_set %>% mutate(
    r_y = year(as_datetime(timestamp)),
    r_m = month(as_datetime(timestamp)),
    r_q = quarter(as_datetime(timestamp)),
    r_h = chron::is.weekend(strftime(as_datetime(timestamp), format = "%m/%d/%y")),
    r_t = cut(x = lubridate::hour(as_datetime(timestamp)), 
              breaks = lubridate::hour(hm("00:00", "6:00", "12:00", "18:00", "23:59")), 
              labels = c("Night", "Morning", "Afternoon", "Evening"), 
              include.lowest=TRUE)
  ) 
  # combine calender/daytime info with userId
  ##
  train_set$userId_Cal_Daytime <- paste(train_set$userId, train_set$r_t, train_set$r_q, train_set$r_h, sep = "|")
  test_set$userId_Cal_Daytime <- paste(test_set$userId, test_set$r_t, test_set$r_q, test_set$r_h,  sep = "|")
  
  # genre variable combining "Film-Noir|IMAX|Documentary" in one genre
  ##
  train_set$genre_FN_IMAX_DOC <- 0  
  train_set$genre_FN_IMAX_DOC[ grep("Film-Noir|IMAX|Documentary", train_set$genres)] <- 1 
  
  test_set$genre_FN_IMAX_DOC <- 0  
  test_set$genre_FN_IMAX_DOC[ grep("Film-Noir|IMAX|Documentary", test_set$genres)] <- 1 
  
  ###########
  ###########
  # create object to store the model resuts
  rmse_results <- tibble()
  ###
  # calculate average of train_set for further use in the modes
  mu <- mean(train_set$rating)
  
  #for final run do not apply all models only the last (with best performance)
  if (!final) {
    ###########
    # model_0:  average
    method_name <- "Just the average"
    method_name_short <- "model_0_rmse"
    
    predicted_ratings <- mu
    # calculate rsme
    assign(method_name_short, RMSE(predicted_ratings, test_set$rating))
    rmse_results <- rbind(rmse_results, tibble(method = method_name, 
                                               method_short = method_name_short, 
                                               RMSE = get(method_name_short))
    )
    cat("\nModel __", method_name, "__ processed.\n", sep = "")
    
    ##########
    # model_1: Movie Effect Model 
    
    method_name <- "Movie Effect Model"
    method_name_short <- "model_1_rmse"
    
    movie_avgs <- train_set %>% 
      group_by(movieId) %>% 
      summarize(b_i = mean(rating - mu))
    predicted_ratings <- mu + test_set %>% 
      left_join(movie_avgs, by='movieId') %>%
      .$b_i
    # calculate rsme
    assign(method_name_short, RMSE(predicted_ratings, test_set$rating))
    rmse_results <- rbind(rmse_results, tibble(method = method_name, 
                                               method_short = method_name_short, 
                                               RMSE = get(method_name_short))
    )
    cat("\nModel __", method_name, "__ processed.\n", sep = "")
    
    ##########
    # model_2: Movie + User Effects Model
    method_name <- "Movie + User Effects Model"
    method_name_short <- "model_2_rmse"
    
    user_avgs <- train_set %>% 
      left_join(movie_avgs, by='movieId') %>%
      group_by(userId) %>%
      summarize(b_u = mean(rating - mu - b_i))
    predicted_ratings <- test_set %>% 
      left_join(movie_avgs, by='movieId') %>%
      left_join(user_avgs, by='userId') %>%
      mutate(pred = mu + b_i + b_u) %>%
      .$pred
    # calculate rsme
    assign(method_name_short, RMSE(predicted_ratings, test_set$rating))
    rmse_results <- rbind(rmse_results, tibble(method = method_name, 
                                               method_short = method_name_short, 
                                               RMSE = get(method_name_short))
    )
    cat("\nModel __", method_name, "__ processed.\n", sep = "")
    
    
    ##########
    # model_2a: Movie + User + genre_FN_IMAX_DOC Effects Model
    method_name <- "Movie + User + genre_FN_IMAX_DOC Effects Model"
    method_name_short <- "model_2a_rmse"
    
    genre_avgs <- train_set %>% 
      group_by(genre_FN_IMAX_DOC) %>% 
      summarize(b_g = mean(rating - mu))
    
    predicted_ratings <- test_set %>% 
      left_join(movie_avgs, by='movieId') %>%
      left_join(user_avgs, by='userId') %>%
      left_join(genre_avgs, by = 'genre_FN_IMAX_DOC') %>%
      mutate(pred = mu + b_i + b_u + b_g) %>%
      .$pred
    # calculate rsme
    assign(method_name_short, RMSE(predicted_ratings, test_set$rating))
    rmse_results <- rbind(rmse_results, tibble(method = method_name, 
                                               method_short = method_name_short, 
                                               RMSE = get(method_name_short))
    )
    cat("\nModel __", method_name, "__ processed.\n", sep = "")
    
    
    ##########
    # model_2b: Movie + User/Cal_Daytime Effects Effects Model
    method_name <- "Movie + User/Cal_Daytime Effects Model"
    method_name_short <- "model_2b_rmse"
    
    user_avgs_cal <- train_set %>% 
      left_join(movie_avgs, by='movieId') %>%
      group_by(userId_Cal_Daytime) %>%
      summarize(b_ucal = mean(rating - mu - b_i))
    
    predicted_ratings <- test_set %>% 
      left_join(movie_avgs, by='movieId') %>%
      left_join(user_avgs, by='userId') %>%
      left_join(user_avgs_cal, by='userId_Cal_Daytime') %>%
      mutate(pred = ifelse(is.na(b_ucal),mu + b_i + b_u, mu + b_i + b_ucal)) %>%
      .$pred
    # calculate rsme
    assign(method_name_short, RMSE(predicted_ratings, test_set$rating))
    rmse_results <- rbind(rmse_results, data_frame(method = method_name, 
                                                   method_short = method_name_short, 
                                                   RMSE = get(method_name_short))
    )
    cat("\nModel __", method_name, "__ processed.\n", sep = "")
  }
  
  ##########
  # model_3: Regularized Movie + User/Cal_Daytime Effects Model
  method_name <- "Regularized Movie + User/Cal_Daytime Effects Model"
  method_name_short <- "model_3a_rmse"
  if (final) lambdas <- final_lambda else lambdas <- seq(4, 5, 0.1)
  rmses <- sapply(lambdas, function(l){
    #mu <- mean(train_set$rating) 
    #mu already set in the beginning
    b_i <- train_set %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+l))
    b_u <- train_set %>%
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+l))
    b_ucal <- train_set %>%
      left_join(b_i, by="movieId") %>%
      group_by(userId_Cal_Daytime) %>%
      summarize(b_ucal = sum(rating - b_i - mu)/(n()+l))
    predicted_ratings <-
      test_set %>%
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      left_join(b_ucal, by = "userId_Cal_Daytime") %>%
      mutate(pred = ifelse(is.na(b_ucal),mu + b_i + b_u, mu + b_i + b_ucal)) %>%
      pull(pred)
    return(RMSE(predicted_ratings, test_set$rating))
  })
  #qplot(lambdas, rmses)
  lambda <- lambdas[which.min(rmses)]
  
  # calculate rsme
  assign(method_name_short, min(rmses))
  rmse_results <- rbind(rmse_results, data_frame(method = method_name, 
                                                 method_short = method_name_short, 
                                                 RMSE = get(method_name_short))
  )
  cat("\nModel __", method_name, "__ processed.\n", sep = "")
  if (!final) cat("\nIteratively selected minimizing penalty term lambda: ", lambda)
  if (final) cat("\nPenalty term lambda was: ", lambda)
  ##################
  return(rmse_results)
  ##################
}


###FUNCTION CALL###

# preliminary models developed from edx split into test and train set
if (call_prelim_models_methods) {
  prelim_results <- models_methods(final = FALSE) 
  print(prelim_results)
}

# final model with edx as train and validation as test set
#be aware to set the lambda from the prelim call manually to the function call final
if (call_final_models_methods) {
  final_result <- models_methods(final = TRUE, final_lambda = 4.2) 
  print(final_result) 
}
