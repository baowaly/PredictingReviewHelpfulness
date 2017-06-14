#remove all variables
rm(list=ls())

#start time
start.time <- Sys.time()

cat("Loading Packages..")

# source("/home/baowaly/steam/c.R/featureRankingAll.R")
list.of.packages <- c("caret", "data.table","parallel","doMC", "gbm", "e1071", "xgboost", "ROCR", "pROC", "DMwR", "ROSE", "plyr", "binom")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

if(length(new.packages)) install.packages(new.packages, repos="http://cran.rstudio.com/")

library(data.table)
library(parallel)
library(caret)
library(doMC)
library(gbm)
library(e1071)
library(xgboost)
library(ROCR)
library(pROC)
library(DMwR)
library(ROSE)
library(plyr)
library(binom)

#Register cores
freeCores = max(1, detectCores(logical = FALSE) - 1)
registerDoMC(freeCores)

#Set path
path = "/home/baowaly/steam/"
#path = "/home/yipeitu/steam_helpful_review_20160701/"
setwd(path)

#Model Evaluation function
get_eval_score = function(model.fit, trainX, trainY, testX, testY, score, method){
  
  positiveClass <- ifelse(score %in% c(0.05, 0.10), "No", "Yes")
  
  #Evaluation on Training Data
  #train.pred <- predict(model.fit, trainX)
  train.pred <- predict(model.fit, type = "raw")
  train.ref <- trainY#dataTrain$helpful
  train.confMatrx <- confusionMatrix(data=train.pred, reference=train.ref, positive=positiveClass, mode = "everything")
  train.accuracy <- train.confMatrx$overall[["Accuracy"]]
  train.f1score <- train.confMatrx$byClass[["F1"]]
  
  #NULL MODEL
  #train.pred.null <- 
  
  
  #Training AUC
  train.pred.prob <- predict(model.fit, type = "prob")
  pred_vector <- train.pred.prob$Yes
  if(positiveClass == "No")
    pred_vector <- train.pred.prob$No
  ref_vector <- as.factor(ifelse(train.ref == positiveClass, 1, 0)) #make numeric
  auc.pred <- prediction(predictions = pred_vector, labels = ref_vector)
  auc.tmp <- performance(auc.pred,"auc");
  train.auc <- as.numeric(auc.tmp@y.values)
  
  #Evaluation on Test Data
  test.pred <- predict(model.fit, testX)
  test.ref <- testY#dataTest$helpful
  test.confMatrx <- confusionMatrix(data=test.pred, reference=test.ref, positive=positiveClass, mode = "everything")
  test.precision <- test.confMatrx$byClass[["Precision"]]
  test.recall <- test.confMatrx$byClass[["Recall"]]
  test.sensitivity <- test.confMatrx$byClass[["Sensitivity"]]
  test.specificity <- test.confMatrx$byClass[["Specificity"]]
  test.accuracy <- test.confMatrx$overall[["Accuracy"]]
  test.f1score <- test.confMatrx$byClass[["F1"]]
  cat("\n\tF1score: ", test.f1score)
  
  #Test AUC
  test.pred.prob <- predict(model.fit, testX, type = "prob")
  pred_vector <- test.pred.prob$Yes
  if(positiveClass == "No")
    pred_vector <- test.pred.prob$No
  ref_vector <- as.numeric(ifelse(test.ref == positiveClass, 1, 0)) #make numeric
  auc.pred <- prediction(predictions = pred_vector, labels = ref_vector)
  auc.tmp <- performance(auc.pred,"auc");
  test.auc <- as.numeric(auc.tmp@y.values)
  
  return(data.table(train.accuracy=train.accuracy, 
                    train.f1score=train.f1score, 
                    train.auc=train.auc, 
                    test.precision=test.precision, 
                    test.recall=test.recall, 
                    test.sensitivity=test.sensitivity, 
                    test.specificity=test.specificity, 
                    test.accuracy=test.accuracy, 
                    test.f1score=test.f1score, 
                    test.auc=test.auc, 
                    method=method))
}

##########Model Bulding and Evaluation###########

#######Select Some Parameters#########
genre <- "Racing"
vote.num <- 50

#Cross Validation Params
n.fold <- 10 
n.repeats <- 3
######################################

#load dataset by vote
dataFile <- paste0("Dataset/LDA_reviews_", genre ,"_50.Rdata")
if(!file.exists(dataFile) || file.size(dataFile) == 0){
  stop("Data file doesn't exist")
}
d.genre.vote <- get(load(dataFile))
#########Dataset Load End##################

#scores <- c(0.05, 0.10, 0.80, 0.85, 0.90, 0.95)
score <- 0.90

  ##Add a helpful column
  d.genre.vote$helpful <- ifelse(d.genre.vote$ws.score > score, "Yes", "No")
  d.genre.vote$helpful <- as.factor(d.genre.vote$helpful)
  dim.y <- "helpful"
  
  #load Feature File
  cat("\nLoading Feature File..")
  if(genre == "All"){
    featureFile <- paste0("Features/FT_", genre, "_V", vote.num, "_R", score,"_S10.Rdata")
  }else{
    featureFile <- paste0("Features/FT_", genre, "_V", vote.num, "_R", score,"_S25.Rdata")
  }
  
  if(!file.exists(featureFile)){
    stop("\nFeature File Not Exists!!")
  }
  load(featureFile)
  
  #Select dataset with max features
  max.dims <- NROW(feature.rank)
  dim.x <- feature.rank[order(feature.rank$total)[1:max.dims],]$name
  
  #Exclude wv features
  #dim.x <- grep("^wv.*?", dim.x, value = TRUE,  invert = TRUE)
  
  #topic features
  dim.topics <- grep("^topic.*?", names(d.genre.vote), value = TRUE)
  n_topics <- length(dim.topics)
  dg <- d.genre.vote[, c(dim.x, dim.topics, dim.y), with=F]
  
  #Delete all rows with NA
  dg <- na.omit(dg)
  n.rows <- NROW(dg)
  
  print(table(dg$helpful))
  
  ############Sample Size for Model Training###########
  cat("\nFinding Important Features..")
  n.samplePercent <- 100
  comb.score <- NULL
  feature.imp <- NULL
  
  #################################################

  n.sampleSize <- ceiling(n.rows * n.samplePercent/100)
  cat("\nModel Building and Evaluation: Genre:", genre, " Vote:", vote.num, " Score:", score, " Sample Size(",n.samplePercent,"%):", n.sampleSize)
  
  positiveClass <- ifelse(score %in% c(0.05, 0.10), "No", "Yes")
  
  #loop control
  n.loop <- seq(1)
  for(l.counter in n.loop){
    
    cat("\nResample: ", l.counter, " Score:", score, " Sample Size(",n.samplePercent,"%):", n.sampleSize)
    
    #Randomly shuffle the data
    sampleData <- dg[sample(n.sampleSize),] 
    
    #Partition sample dataset into training and testing
    split=0.80
    trainIndex <- as.vector(createDataPartition(y=sampleData$helpful, p=split, list=FALSE))
    
    cat("\n\tGBM: Get train and test data")
    dataTrain <- sampleData[trainIndex, ]
    dataTest <- sampleData[-trainIndex, ]
    
    dataTrain$pred <- dataTrain$helpful
    dataTrain$pred <- "No"
    
    # confusion matrix of test set
    cmTrain <- confusionMatrix(data=dataTrain$pred, reference=dataTrain$helpful, positive=positiveClass, mode = "everything")
    print(cmTrain)
    
    #split test data 
    dataTest$pred <- dataTest$helpful
    dataTest$pred <- "No"
    cmTest <- confusionMatrix(data=dataTest$pred, reference=dataTest$helpful, positive=positiveClass, mode = "everything")
    
    print(cmTest)
    
  
  }#end of loop
    

  
  #Save score
  #outputFile <- paste0("LDA-Evaluation/NULL_SCORE_", genre, "_V", vote.num, "_R", score,".Rdata")
  #if(!file.exists(outputFile)){
    #file.create(outputFile)  
  #} 
  #save(comb.score, file=outputFile)
  #cat("\nSaved file in: ", outputFile, "\n")

  


cat("\nModel Bulding and Evaluation: Done")
#########End of Model Bulding and Evaluation###########

end.time <- Sys.time()
time.taken <- end.time - start.time
print(time.taken)