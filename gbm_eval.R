#remove all variables
rm(list=ls())

#start time
start.time <- Sys.time()

cat("Loading Packages..")

list.of.packages <- c("caret", "data.table","parallel","doMC", "gbm", "e1071", "xgboost", "ROCR", "pROC", "ROSE", "plyr", "binom")
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
#library(DMwR)
library(ROSE)
library(plyr)
library(binom)

#Register cores
freeCores = max(1, detectCores(logical = FALSE) - 1)
registerDoMC(freeCores)

#Set path
path = "~/PredictingReviewHelpfulness/"
setwd(path)

#Model Evaluation function
get_eval_score = function(model.fit, trainX, trainY, testX, testY, score, method){
  
  positiveClass <- ifelse(score %in% c(0.05, 0.10), "No", "Yes")
  
  #Evaluation on Training Data
  #train.pred <- predict(model.fit, trainX)
  train.pred <- predict(model.fit, type = "raw")
  train.ref <- trainY#dataTrain$helpful
  train.confMatrx <- confusionMatrix(data=train.pred, reference=train.ref, positive=positiveClass, mode = "everything")
  train.precision <- train.confMatrx$byClass[["Precision"]]
  train.recall <- train.confMatrx$byClass[["Recall"]]
  train.sensitivity <- train.confMatrx$byClass[["Sensitivity"]]
  train.specificity <- train.confMatrx$byClass[["Specificity"]]
  train.accuracy <- train.confMatrx$overall[["Accuracy"]]
  train.f1score <- train.confMatrx$byClass[["F1"]]
  
  
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
  #cat("\n\tF1score: ", test.f1score)
  
  #Test AUC
  test.pred.prob <- predict(model.fit, testX, type = "prob")
  pred_vector <- test.pred.prob$Yes
  if(positiveClass == "No")
    pred_vector <- test.pred.prob$No
  ref_vector <- as.numeric(ifelse(test.ref == positiveClass, 1, 0)) #make numeric
  auc.pred <- prediction(predictions = pred_vector, labels = ref_vector)
  auc.tmp <- performance(auc.pred,"auc");
  test.auc <- as.numeric(auc.tmp@y.values)
  
  return(list(train.precision=train.precision, 
              train.recall=train.recall, 
              train.sensitivity=train.sensitivity, 
              train.specificity=train.specificity, 
              train.accuracy=train.accuracy, 
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
    
#Define some variables
genre <- "Racing"
vote_num <- 50

#load dataset 
dataFile <- paste0("Dataset/Reviews_", genre ,"_50.Rdata")
if(!file.exists(dataFile) || file.size(dataFile) == 0){
  stop("Data file doesn't exist")
}
load(dataFile)
original_dataset <- ds.reviews

# dimensions of dataset
dim(original_dataset)

##Add a helpful column
score_thrsld <- 0.90
original_dataset$helpful <- ifelse(original_dataset$ws.score > score_thrsld, "Yes", "No")
original_dataset$helpful <- as.factor(original_dataset$helpful)
dim.y <- "helpful"

# Load feature file
featureFile <- paste0("Features/FT_", genre, "_V", vote_num, "_R", score_thrsld,"_S25.Rdata")
if(!file.exists(featureFile)){
  stop("\nFeature File Not Exists!!")
}
load(featureFile)

# Total number of features
max.dims <- NROW(feature.rank)
print(max.dims)

# Total feature list
dim.x <- feature.rank[order(feature.rank$total)[1:max.dims],]$name

# Exclude tfidf features, we found there are not important at all
dim.x <- grep("^tfidf.*?", dim.x, value = TRUE,  invert = TRUE)
dataset <- original_dataset[, c(dim.x, dim.y), with=F]
# Peek at the Data   
head(dataset[,1:5], 5)

# Select number of features
gbm.dim <- 840
# Selec the best subset of the total features
gbm.dim.x <- dim.x[1:gbm.dim]

# Total number of rows in the dataset
n.rows <- NROW(dataset)
# Class Distribution
print(table(dataset$helpful))

#Sample size
n.samplePercent <- 100
n.sampleSize <- ceiling(n.rows * n.samplePercent/100)


comb.score <- data.frame()
feature.imp <- NULL

n.loop <- 10
for( iter in seq(n.loop) ){
  
  cat("\nResample: ", iter, " Score:", score_thrsld, " Sample Size(",n.samplePercent,"%):", n.sampleSize)
  
  # Balance data set with both over and under sampling
  balanced_data <- ovun.sample(helpful ~ ., data = dataset, method="both", p=0.5, N=n.sampleSize)$data
  print(table(balanced_data$helpful))
  
  # Split dataset
  split <- 0.80
  trainIndex <- as.vector(createDataPartition(y=balanced_data$helpful, p=split, list=FALSE))
  
  # Get train data
  gbm.dataTrain <- balanced_data[trainIndex, c(gbm.dim.x, dim.y), ]
  dim(gbm.dataTrain)
  
  # Get test data
  gbm.dataTest <- balanced_data[-trainIndex, c(gbm.dim.x,dim.y), ]
  dim(gbm.dataTest)
  
  # Split train data
  gbm.trainX <- gbm.dataTrain[, gbm.dim.x, ]
  gbm.trainY <- as.factor(gbm.dataTrain[, dim.y, ])
  
  # Split test data 
  gbm.testX <- gbm.dataTest[, gbm.dim.x, ]
  gbm.testY <- as.factor(gbm.dataTest[, dim.y, ])
  
  # Remove columns with near zero variance
  nearZV <- nearZeroVar(gbm.trainX)
  if(length(nearZV) > 0){
    gbm.trainX <- gbm.trainX[, -nearZV]
    gbm.testX <- gbm.testX[, -nearZV]
  }
  
  # Preprocess training Data
  preObj <- preProcess(gbm.trainX, method = c("center", "scale"))
  gbm.trainX <- predict(preObj, gbm.trainX)
  
  # Preprocess test Data
  preObj <- preProcess(gbm.testX, method = c("center", "scale"))
  gbm.testX <- predict(preObj, gbm.testX)
  
  # Control parameters 
  gbm.fitControl = trainControl(method="repeatedcv", #small size -> repeatedcv
                                number=10, #10-fold cv
                                repeats=3,
                                returnResamp = "final",
                                selectionFunction = "best",
                                classProbs=TRUE, 
                                summaryFunction=twoClassSummary,
                                allowParallel = TRUE)
  # Train the model
  gbmFit <- train(gbm.trainX, gbm.trainY, method="gbm", metric="ROC", trControl=gbm.fitControl, verbose=F)
  
  eval_score <- get_eval_score(gbmFit, gbm.trainX, gbm.trainY, gbm.testX, gbm.testY, score_thrsld, "GBM")
  #print(eval_score)
  comb.score <- rbind(comb.score, eval_score)
  
  # Importance features
  gbmImp <- varImp(gbmFit, scale = TRUE)
  impFeatures <- gbmImp$importance
  f_weight <- NULL
  for (f in gbm.dim.x) {
    f_weight <- c(f_weight, impFeatures[f, ])
  }
  f_weight[is.na(f_weight)] <- length(gbm.dim.x) * (-1)
  # Combine important features
  feature.imp <- cbind(feature.imp, weight=f_weight)

}#end of for loop

#Save score
outputFile <- paste0("Evaluation/SCORE_", genre, "_V", vote_num, "_R", score_thrsld,".Rdata")
if(!file.exists(outputFile)){
  file.create(outputFile)  
} 
save(comb.score, file=outputFile)

#print final score
final_eval_score <- sapply(Filter(is.numeric, comb.score), mean)
print(final_eval_score)

#########Save Feautre Importance#################################
feature.imp.do <- T
if(feature.imp.do){
  #make summation of weights
  feature.imp <- as.data.frame(feature.imp, stringsAsFactors=FALSE)
  n.cols <- NCOL(feature.imp)
  col_names <- sprintf("weight.%s",seq(1:n.cols))
  colnames(feature.imp) <- col_names
  
  feature.imp$total <- rowSums(subset(feature.imp, select=1:n.cols))
  feature.imp$mean <- rowMeans(subset(feature.imp, select=1:n.cols))
  
  #include feature column
  dim.x.mod <- gsub(".root", "", gbm.dim.x)
  dim.x.mod <- gsub(".log", "", dim.x.mod)
  dim.x.mod <- gsub(".reciprocal", "", dim.x.mod)
  feature.imp$feature <- dim.x.mod
  
  #sort
  feature.imp <- feature.imp[order(-feature.imp$total),]
  rownames(feature.imp) <- NULL
  
  outputFile <- paste0("ImpFeatures/Imp_FT_", genre, "_V", vote_num, "_R", score_thrsld,".Rdata")
  if(!file.exists(outputFile)){
    file.create(outputFile)  
  } 
  save(feature.imp, file=outputFile)
}


cat("\nModel Bulding and Evaluation: Done")
cat("\nSaved file in: ", outputFile, "\n")
#########End of Model Bulding and Evaluation###########

end.time <- Sys.time()
time.taken <- end.time - start.time
print(time.taken)