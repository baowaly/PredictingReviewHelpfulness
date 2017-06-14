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
path = "/home/baowaly/1052DataScience/"
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
dataFile <- paste0("Dataset/Reviews_", genre ,"_50.Rdata")
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
  featureFile <- paste0("Features/FT_", genre, "_V", vote.num, "_R", score,"_S25.Rdata")
  if(!file.exists(featureFile)){
    stop("\nFeature File Not Exists!!")
  }
  load(featureFile)
  
  #Select dataset with max features
  max.dims <- NROW(feature.rank)
  dim.x <- feature.rank[order(feature.rank$total)[1:max.dims],]$name
  
  #Exclude tfidf features
  dim.x <- grep("^tfidf.*?", dim.x, value = TRUE,  invert = TRUE)
  dg <- d.genre.vote[, c(dim.x, dim.y), with=F]
  
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
    
    #loop control
    n.loop <- seq(10)
    for(l.counter in n.loop){
      
      cat("\nResample: ", l.counter, " Score:", score, " Sample Size(",n.samplePercent,"%):", n.sampleSize)
      
      #Creates possibly balanced samples by random over-sampling minority examples, under-sampling majority examples or combination of over- and under-sampling.
      sampleData <- ovun.sample(helpful ~ ., data = dg, method="both", p=0.5, N=n.sampleSize)$data
      
      #Partition sample dataset into training and testing
      split=0.80
      trainIndex <- as.vector(createDataPartition(y=sampleData$helpful, p=split, list=FALSE))
      
      ############Features select############################
      gbm.dims <- c(seq(100, 380, 20))
      ########################################
      for(gbm.dim in gbm.dims){  
        ############################################### GBM START HERE ######################################################
        cat("\n\tGBM: Selecting Features: ", gbm.dim)
        
        #selecting features
        gbm.dim.x <- dim.x[1:gbm.dim] #feature.rank[order(feature.rank$total)[1:gbm.dim],]$name
        
        cat("\n\tGBM: Get train and test data")
        gbm.dataTrain <- sampleData[trainIndex, c(gbm.dim.x,dim.y), ]
        gbm.dataTest <- sampleData[-trainIndex, c(gbm.dim.x,dim.y), ]
        
        #split train data
        gbm.trainX <- gbm.dataTrain[, gbm.dim.x, ]
        gbm.trainY <- as.factor(gbm.dataTrain[, dim.y, ])
        
        #split test data 
        gbm.testX <- gbm.dataTest[, gbm.dim.x, ]
        gbm.testY <- as.factor(gbm.dataTest[, dim.y, ])
        
        gbm.fitControl = trainControl(method="cv", #small size -> repeatedcv
                                      number=n.fold, 
                                      repeats=n.repeats,
                                      returnResamp = "final",
                                      selectionFunction = "best",
                                      classProbs=TRUE, 
                                      summaryFunction=twoClassSummary,
                                      allowParallel = TRUE
                                      )
        
        cat("\n\tGBM: Remove columns with near zero variance")
        nearZV <- nearZeroVar(gbm.trainX)
        if(length(nearZV) > 0){
          gbm.trainX <- gbm.trainX[, -nearZV]
          gbm.testX <- gbm.testX[, -nearZV]
        }
        
        cat("\n\tGBM: Preprocess Data")
        #Preprocess Training Data
        preObj <- preProcess(gbm.trainX, method = c("center", "scale"))
        gbm.trainX <- predict(preObj, gbm.trainX)
        
        #Preprocess Testing Data
        preObj <- preProcess(gbm.testX, method = c("center", "scale"))
        gbm.testX <- predict(preObj, gbm.testX)
        
        cat("\n\tGBM: Learning Model")
        
        gbmFit <- train(gbm.trainX, gbm.trainY, method="gbm", metric="ROC", trControl=gbm.fitControl, verbose=F)
        
        gbmImp <- varImp(gbmFit, scale = TRUE)
        
        # combine importance features
        impFeatures <- gbmImp$importance
        f_weight <- NULL
        for (f in gbm.dim.x) {
          f_weight <- c(f_weight, impFeatures[f, ])
        }
        f_weight[is.na(f_weight)] <- length(gbm.dim.x) * (-1)
        feature.imp <- cbind(feature.imp, weight=f_weight)
        
        #cat("\n\tGBM: Get Evaluation Score")
        gbmScore <- cbind(genre=genre, vote=vote.num, ws.score=score, sample.size=n.sampleSize, features=gbm.dim, get_eval_score(gbmFit, gbm.trainX, gbm.trainY, gbm.testX, gbm.testY, score, "GBM"))
        #cat("\n\tGBM: Done\n") 
        ################################################ GBM END HERE #######################################################
        
        #combine score
        #print(gbmScore)
        comb.score <- rbind(comb.score, gbmScore)
        
      }#end of dimentions
      
    }#end of loop
    

  
  #Save score
  outputFile <- paste0("Evaluation/SCORE_", genre, "_V", vote.num, "_R", score,".Rdata")
  if(!file.exists(outputFile)){
    file.create(outputFile)  
  } 
  save(comb.score, file=outputFile)
  
  #########Feautre Importance#################################
  feature.imp.do <- FALSE
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
    
    outputFile <- paste0("ImpFeatures/Imp_FT_", genre, "_V", vote.num, "_R", score,".Rdata")
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