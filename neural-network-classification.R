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
features <- features[,2]

# Load the training dataset
trainData <- read.delim('./train/X_train.txt', header = F, sep = '', dec = '.', col.names = features)
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
testData <- read.delim('./test/X_test.txt', header = F, sep = '', dec = '.', col.names = features)
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
# library(nnet)
# library(caret)
# 
# # The number of features is too much, select a subset of the features from the dataset
# subtrain <- trainData[,1:200]
# subtrain$activity <- trainData$activity
# 
# subtest <- testData[,1:200]
# 
# # Construct the neural network model with the features from PCA
# nn <- nnet(activity ~ ., data = subtrain, size=4, decay=1.0e-5, maxit=50)
# 
# # Apply the neural network model on the testdata
# prediction <- predict(nn, subtest, type = 'class')
# 
# # Check the predicted output
# table(prediction)
# 
# # Compare the actual output with the predicted
# actual <- testData$activity
# table(actual)
# 
# # Confusion matrix for better comparison
# results <- data.frame(actual=actual, prediction=prediction)
# t <- table(results)
# confusionMatrix(t)

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

# From the scree plot, 50 features can explain 90% of the variance

# Create a new dataset from PCA result
tmpTrain <- data.frame(activity = trainData$activity, pca$x)
t <- as.data.frame(predict(pca, newdata = pcaTest))

train2 <- tmpTrain[,1:51]
test2 <- t[, 1:50]

############## Neural Network after PCA #################
library(nnet)
library(caret)

# Construct the neural network model with the features from PCA
nn <- nnet(activity ~ ., data = train2, size=10, decay=1.0e-5, maxit=50)

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

############# Deep learning ####################
library(h2o)
library(caret)

# Train an artificial neural network with h2o
h2o.init(nthreads = -1, max_mem_size = '2G')
# Create a clean slate just in case the cluster is already running
h2o.removeAll()
nnModel <-  h2o.deeplearning(y = 'activity',
                         x = setdiff(names(train2), 'activity'),
                         training_frame = as.h2o(train2),
                         activation = 'Rectifier',
                         hidden = c(100,100),
                         distribution = 'multinomial',
                         epochs = 100,
                         train_samples_per_iteration = -2,
                         variable_importances = T)

# Show some statistics of the model
summary(nnModel)
plot(nnModel)

# Use the model to predict the activity on the test data
pred <- h2o.predict(nnModel, newdata = as.h2o(test2))

# Extract the predicted label from the prediction
pred <- as.vector(pred$predict)
table(pred)
actual <- testData$activity
table(actual)

# Construct the confusion matrix
results <- data.frame(actual=actual, prediction=pred)
t <- table(results)
confusionMatrix(t)

# Shut down h2o after completing
h2o.shutdown(prompt = FALSE)
