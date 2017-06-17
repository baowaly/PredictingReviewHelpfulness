#remove all variables
rm(list=ls())

#start time
start.time <- Sys.time()

cat("Loading Packages..")

# source("/home/baowaly/steam/c.R/featureRankingAll.R")
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
library(ROSE)
library(plyr)
library(binom)

#Register cores
freeCores = max(1, detectCores(logical = FALSE) - 1)
registerDoMC(freeCores)

#Set path
path = "~/PredictingReviewHelpfulness/"
setwd(path)

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

positiveClass <- ifelse(score_thrsld %in% c(0.05, 0.10), "No", "Yes")

  
cat("\nBuilding null model")

#Randomly shuffle the data
sampleData <- dataset[sample(n.sampleSize),] 

#Partition sample dataset into training and testing
split=0.80
trainIndex <- as.vector(createDataPartition(y=sampleData$helpful, p=split, list=FALSE))

dataTrain <- sampleData[trainIndex, ]
dataTest <- sampleData[-trainIndex, ]

dataTrain$pred <- dataTrain$helpful
dataTrain$pred <- "No"

# confusion matrix of test set
cmTrain <- confusionMatrix(data=dataTrain$pred, reference=dataTrain$helpful, positive=positiveClass, mode = "everything")
print(cmTrain)
train.accuracy <- cmTrain$overall[["Accuracy"]]
print(train.accuracy)

#split test data 
dataTest$pred <- dataTest$helpful
dataTest$pred <- "No"
cmTest <- confusionMatrix(data=dataTest$pred, reference=dataTest$helpful, positive=positiveClass, mode = "everything")
test.accuracy <- cmTest$overall[["Accuracy"]]
print(cmTest)
print(test.accuracy)

end.time <- Sys.time()
time.taken <- end.time - start.time
print(time.taken)