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

# Change the labels to factor
trainData$activity <- factor(trainData$activity)
testData$activity <- factor(testData$activity)

levels(trainData$activity)
levels(testData$activity)

# Remove some not needed variable to clear up space
rm(features, testLabel, trainLabel)

############### Neural Network ######################
library(nnet)
library(caret)

# Plot the neural network
# Find the precision and recall

# Construct the neural network model with the features from PCA
nn <- nnet(activity ~ ., data = trainData, size=4, decay=1.0e-5, maxit=50)

# Apply the neural network model on the testdata
prediction <- predict(nn, testData[-activity], type = 'class')

# Check the predicted output
table(prediction)

# Compare the actual output with the predicted
actual <- testData$activity
table(actual)

# Confusion matrix for better comparison
results <- data.frame(actual=actual, prediction=prediction)
t <- table(results)
confusionMatrix(t)

############# Principal Component Analysis #############
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

############## Neural Network after PCA #################
library(nnet)
library(caret)

# Construct the neural network model with the features from PCA
nn <- nnet(activity ~ ., data = train2, size=4, decay=1.0e-5, maxit=50)

# Apply the neural network model on the testdata
prediction <- predict(nn, test2, type = 'class')

# Check the predicted output
table(prediction)

# Compare the actual output with the predicted
actual <- testData$activity
table(actual)

# Confusion matrix for better comparison
results <- data.frame(actual=actual, prediction=prediction)
t <- table(results)
confusionMatrix(t)
