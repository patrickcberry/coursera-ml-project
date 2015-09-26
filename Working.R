# Clean up environment

rm(list=ls())

# Load the data sets

raw_train <- read.csv("../project-data/pml-training.csv")
raw_test <- read.csv("../project-data/pml-testing.csv")


# #####################################################
# TRAINING DATA
#
# Variable to predict is the 'classe variable
# - Factor variable with 5 levels A ... E
#
# 160 VARIABLES
# 19622 OBSERVATIONS
#
# Contains may factor variables
# Contains many variables with NAs
# 
# Six users
# Timestamped

summary(raw_train)
str(raw_train)
str(raw_train$classe)

# #####################################################
# TEST DATA
#
# test$X and test$problem_id both contain the same value which is the id for submission 
#
# The test data dosn't contain the predicted variable so cannot be used as test data in developing
# the models

raw_test$X

# #####################################################
# Create data sets
#
# training 60%
# testing 20%
# validation 20%

library(caret)

set.seed(1234567)
inTrain <- createDataPartition( raw_train$classe, p = 0.6, list = FALSE )

training <- raw_train[ inTrain, ]
testAndValidation <- raw_train[ -inTrain, ]

inVal <- createDataPartition( testAndValidation$classe, p = 0.5, list = FALSE )
validation <- testAndValidation[ inVal, ]
testing <- testAndValidation[ -inVal, ]

rm(list=c("inTrain", "inVal", "test", "testAndValidation", "raw_train"))

# #####################################################
# Preprocessing

# Removing unnecessary varaibles
rem <- grepl("^X|user|timestamp|window", names(training))str
training <- training[, !rem]

# Too many covariates at 160
# Remove near zero covariates

nzv <- nearZeroVar( training )
training <- training[,-nzv]

# Remove covariates with high number of missing values (greater than 50%)

training <- training[, colSums(is.na(training)) < 0.5 * nrow(training)]

# #####################################################
# MODEL FITTING

# Since the outcome variable is a factor cannot use a linear model (unless the outcome is 
# converted to a continous variable)
#
# Tree or Random forest would be most appropiate

controlCV <- trainControl(method="cv")

modFitRF <- train( training$classe ~ .,
                   data = training,
                   method = "rf",
                   trControl=controlCV )




modFitRF <- train( training$classe ~ .,
                   data = training,
                   preProcess = "knnImpute", 
                   na.action = na.pass,
                   method = "rf",
                   trControl=controlCV )



# Linear Models
#
# GLM + PCA
#

modFitGLM <- train( training$classe ~ ., 
                    method = "glm", 
                    preProcess = c("center","scale","knnImpute", "pca"), 
                    na.action = na.pass,
                    data = training )











