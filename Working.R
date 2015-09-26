# Clean up environment

rm(list=ls())
setwd("C:/Users/patrick.berry/Desktop/Coursera/Machine Learning/coursera-ml-project")

# Load the data sets

raw_train <- read.csv("../project-data/pml-training.csv",na.strings=c("NA","NaN","#DIV/0!", ""))
raw_test <- read.csv("../project-data/pml-testing.csv",na.strings=c("NA","NaN","#DIV/0!", ""))


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
# Data Partitioning
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

rm(list=c("inTrain", "inVal", "testAndValidation", "raw_train"))

# #####################################################
# Preprocessing

# Removing unnecessary varaibles
rem <- grepl("^X|user|timestamp|window", names(training))
training <- training[, !rem]

# Too many covariates at 160
# Remove near zero covariates

nzv <- nearZeroVar( training )
training <- training[,-nzv]

rm(list=c("rem","nzv"))

# Remove covariates with high number of missing values (greater than 50%)

training <- training[, colSums(is.na(training)) < 0.5 * nrow(training)]

# #####################################################
# MODEL FITTING

# Since the outcome variable is a factor cannot use a linear model (unless the outcome is 
# converted to a continous variable)
#
# Tree or Random forest would be most appropiate

# RANDOM FORREST



ctrl <- trainControl(method="cv",
                     repeats = 1,
                     number = 5,
                     verboseIter = TRUE )

modFitRF <- train( training$classe ~ .,
                   data = training,
                   method = "rf",
                   trControl=ctrl )

# In sample error
predRF <- predict( modFitRF, newdata = training)
inSampleErrorRF <- sum(predRF != training$classe) / nrow(training)

# Out of sample error
predOOSRF <- predict( modFitRF, newdata = testing)
outSampleErrorRF <- sum(predOOSRF != testing$classe) / nrow(testing) * 100

# TREE RPART

modFitRP <- train( training$classe ~ .,
                   data = training,
                   method = "rpart",
                   trControl=ctrl )

# In sample error
predRP <- predict( modFitRP, newdata = training)
inSampleErrorRP <- sum(predRP != training$classe) / nrow(training)

# Out of sample error
predOOSRP <- predict( modFitRP, newdata = testing)
outSampleErrorRP <- sum(predOOSRP != testing$classe) / nrow(testing) * 100

# BOOSTING
# - Crashed system

modFitGBM <- train( training$classe ~ .,
                   data = training,
                   method = "lda",
                   trControl=ctrl )

# In sample error
predGBM <- predict( modFitGBM, newdata = training)
inSampleErrorGBM <- sum(predGBM != training$classe) / nrow(training)

# Out of sample error
predOOSGBM <- predict( modFitGBM, newdata = testing)
outSampleErrorGBM <- sum(predOOSGBM != testing$classe) / nrow(testing) * 100

# ######################################################
# ModelFit
#

confusionMatrix(predOOSRF,testing$classe)
varImp( modFitRF )

# ######################################################
# Create submission files
#

# Create submission vector

answers <- as.character(predict( modFitRF, raw_test ))

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(answers)


