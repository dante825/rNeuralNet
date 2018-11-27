# Data Analytics
# Name: Ng Kang Wei
# Student ID: WQD170068

library(dplyr)

setwd('./UCI-HAR-Dataset/')

################ Loading the datasets ###############
# Load the features names
features <- read.delim('./features.txt', header = F, sep = '')
dim(features)
head(features)
# Select the features names from the loaded df
features <- features %>% select(V2)

# Load the training dataset
trainData <- read.delim('./train/X_train.txt', header = F, sep = '', dec = '.', col.names = features[,1])
dim(trainData)
head(trainData)

# Load the label for the training data
trainLabel <- read.delim('./train/y_train.txt', header = F, sep = '', col.names = 'activity')
dim(trainLabel)

# Add the label data into a column of the training data
trainData$activity <- trainLabel[,1]

# Activity labels:
# 1-walking
# 2-walking_upstairs
# 3-walking_downstairs
# 4-sitting
# 5-standing
# 6-laying

# Load the test data
testData <- read.delim('./test/X_test.txt', header = F, sep = '', dec = '.', col.names = features[,1])
dim(testData)
head(testData)

# Load the labels for test data
testLabel <- read.delim('./test/y_test.txt', header = F, sep = '', col.names = 'activity')
dim(testLabel)

# Add the test labels to the test data
testData$activity <- testLabel[,1]

########### Data preprocessing ######
sapply(trainData, function (x) sum(is.na(x)))
sum(is.na(trainData))

sapply(testData, function(x) sum(is.na(x)))
sum(is.na(testData))

unique(trainData$activity)
unique(testData$activity)

str(trainData, list.len=ncol(trainData))
str(testData, list.len=ncol(testData))

summary(trainData)
summary(testData)

# Change the label of the response variable
# table(trainData$activity)
# trainData$activity[trainData$activity==1] <- "walking"
# trainData$activity[trainData$activity==2] <- "walking_upstairs"
# trainData$activity[trainData$activity==3] <- "walking_downstairs"
# trainData$activity[trainData$activity==4] <- "sitting"
# trainData$activity[trainData$activity==5] <- "standing"
# trainData$activity[trainData$activity==6] <- "laying"
# 
# table(testData$activity)
# testData$activity[testData$activity==1] <- "walking"
# testData$activity[testData$activity==2] <- "walking_upstairs"
# testData$activity[testData$activity==3] <- "walking_downstairs"
# testData$activity[testData$activity==4] <- "sitting"
# testData$activity[testData$activity==5] <- "standing"
# testData$activity[testData$activity==6] <- "laying"

# Change the labels to factor
# trainData$activity <- factor(trainData$activity)
# testData$activity <- factor(testData$activity)
# 
# levels(trainData$activity)
# levels(testData$activity)

# Neuralnet package does not work with factor so the response variable is not change to factor
########### Classification #############
library(neuralnet)
library(caret)
n <- names(trainData)
f <- as.formula(paste('activity ~', paste( n[!n %in% 'activity'], collapse = '+')))
nn <- neuralnet(f, trainData, hidden = 4, linear.output = FALSE, threshold = 0.01)
plot(nn, rep = 'best')

# Remove the labels from the test data
testNolabel <- testData %>% select(-activity)

# Test the neural network model on the test data
nn.results <- compute(nn, testNolabel)

# Confusion Matrix
prediction <- round(nn.results$net.result)
table(prediction)
actual <- testData$activity
table(actual)

u <- union(prediction, actual)
t <- table(factor(prediction, u), factor(actual, u))
confusionMatrix(t)

##### PCA #########
pcaTrain <- trainData %>% select(-activity)
pcaTest <- testData %>% select(-activity)
pca <- prcomp(pcaTrain, scale = F)

# Variance
pcaVar <- (pca$sdev) ^2

# Proportion of variance
propVar <- pcaVar / sum(pcaVar)
propVar[1:100]

# Plot the scree plot for proportion of variance
plot(propVar, xlab = 'Principal Component', ylab = 'Proportion of variance explained', type = 'b',
     main = 'Proportion of Variance explained by Principal Components')

# Sanity check with cumulative scree plot
plot(cumsum(propVar), xlab = 'Principal component', ylab = 'Cumulative proportion of variance explained',
     type = 'b', main = 'Cumulative Proportion of Variance explained by Principal Component')

# From the scree plot, 100 features should be enough

# Create a new dataset from PCA result
tmpTrain <- data.frame(activity = trainData$activity, pca$x)
t <- as.data.frame(predict(pca, newdata = pcaTest))

train2 <- tmpTrain[,1:51]
test2 <- t[, 1:50]

# Build the neural network With the training data after PCA
library(neuralnet)
library(caret)
n <- names(train2)
f <- as.formula(paste('activity ~', paste(n[!n %in% 'activity'], collapse = '+')))
nn <- neuralnet(f, train2, hidden = 5, linear.output = FALSE, threshold = 0.01)

plot(nn, rep = 'best')

# Test the model
nn.results2 <- compute(nn, test2)

prediction <- round(nn.results2$net.result)
table(prediction)
actual <- testData$activity
table(actual)

u <- union(prediction, actual)
t <- table(factor(prediction, u), factor(actual, u))
confusionMatrix(t)