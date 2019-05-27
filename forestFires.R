
# Use DALEX (by @pbiecek) to explain Black-Box Models
# featuring models produced with 'Caret'

setwd("~/R/DALEX")

# installation process

# devtools::install_github("MI2DataLab/factorMerger")
# devtools::install_github("pbiecek/breakDown") # dependencies
# 
# devtools::install_github("pbiecek/DALEX") # DALEX


# Taking A Brief Look at the Dataset ----

library(lattice)
library(caret)
library(tidyverse)
library(DALEX)

fires <- read_csv("forestfires.csv",
                  col_names = TRUE)

glimpse(fires)
map_dbl(fires, ~ length(unique(.))) # see the unique number of instances for each variable

fires <- fires %>%
  mutate_at(vars(X, Y, month, day, rain), list(as.factor)) # convert some predictors to categorical

map_dbl(fires[, -c(1,2,3,4,12)], ~ max(.)/min(.)) # heuristic way to detect skewness

histogram(fires$DMC,
          data = fires,
          col = "gray40") # slighlty right skewed

histogram(fires$DC,
          data = fires,
          col = "gray40") # left skewed

histogram(fires$wind,
          data = fires,
          col = "gray40")

# Using Log Transformation to resolve skewness

catVars <- fires %>%
   select_if(is.factor) # set the dataframe containing only categorical columns

numVars <- fires %>%
  select_if(is.numeric) # set the dataframe containing only numeric columns

numericVars <- setdiff(names(fires), names(catVars)) # set the names of numeric columns

fires[numericVars[1:7]] <- log(fires[numericVars[1:7]] + 1) 
fires

histogram(fires$DC,
           data = fires,
           col = "gray40",
           grid = TRUE) # histogram for DC column as a    lattice object


# Centering and Scaling the Dataset ----

# trsf <- preProcess(fires[, 1:12],
#                    method = c('center', 'scale'))
# 
# firesTrans <- predict(trsf, newdata = fires[, 1:12])
# firesTrans$area <- fires$area
# 
# histogram(firesTrans$DC,
#           data = firesTrans,
#           col = "gray40",
#           grid = TRUE)

# Split the Dataset ----

set.seed(198)
inTr <- createDataPartition(fires$area, p = .7,
                            list = FALSE) 

trainSet <- fires[inTr,]
testSet <- fires[-inTr,]

# trainTrans <- firesTrans[inTr,]
# testTrans <- firesTrans[-inTr,]

# Creating a control object ----

ctrl <- trainControl(method = "cv", n = 10,
                     savePredictions = "all",
                     summaryFunction = defaultSummary) # control object to use repeatedly on models

# Fitting Some models ----

# Random Forests ----

rfGrid <- expand.grid(mtry = seq(2, 12, by = 2)) # setting a grid search on hyperparameter 'mtry'

set.seed(1025)
library(doMC)
registerDoMC(2)
rfTune <- train(area ~.,
                data = trainSet,
                method = "rf",
                ntree = 500,
                metric = "RMSE",
                trControl = ctrl,
                tuneGrid = rfGrid) # random forest model tuning

rfTune

# SVM with Radial Kernel ----

library(kernlab)
set.seed(201)
sigVal <- sigest(area ~., data = fires)

svmGrid <- expand.grid(sigma = sigVal[1],
                       C = 2^(-1:3)) # hyperparameters for SVM-radial

set.seed(1025)
svmTune <- train(area ~.,
                 data = fires,
                 method = "svmRadial",
                 metric = "RMSE",
                 tuneGrid = svmGrid,
                 trControl = ctrl) # support vector machine with radial kernel use

svmTune

# MARS ----

marsGrid <- expand.grid(degree = 1:2,
                        nprune = 4:7)

set.seed(1025)
marsTune <- train(area ~.,
                  data = fires,
                  method = "earth",
                  metric = "RMSE",
                  tuneGrid = marsGrid,
                  trControl = ctrl) # multi-variate adaptive regression spline

marsTune

# Predictions for the Test Set ----

modelPred <- function(obj, newSet){
  pred <- predict(obj, newdata = newSet)
  return(pred)
} # a function to predict on test set

predSet <- data.frame(RF = modelPred(rfTune, testSet),
                      SVM = modelPred(svmTune, testSet),
                      MARS = modelPred(marsTune, testSet),
                      Reference = testSet$area) # generating prediction set along with actual test set values

names(predSet)[3] <- 'MARS' # for some reason, this needs to be manually corrected.
head(predSet, 7)  # explore the prediction and reference values                    

# DALEX for descriptive explanations with explain() ----

rfExplainer <- DALEX::explain(rfTune, label = "rf",
                              data = testSet,
                              y = testSet$area)
    
svmExplainer <- DALEX::explain(svmTune, label = "svm",
                              data = testSet,
                              y = testSet$area)

marsExplainer <- DALEX::explain(marsTune, label = "mars",
                              data = testSet,
                              y = testSet$area)


# Model Performance Considerations ----

rfPerformance <- model_performance(rfExplainer)
svmPerformance <- model_performance(svmExplainer)
marsPerformance <- model_performance(marsExplainer)

rfPerformance # for instance, gives the quantile residuals
plot(rfPerformance) # residuals distribution for single model

plot(rfPerformance, svmPerformance, marsPerformance) +
  theme_light() +
  ggtitle('Residual Distribution of Models') # residuals distribution for all models, line type

plot(rfPerformance, svmPerformance, marsPerformance,
     geom = "boxplot") +
  ggtitle('Residual Distribution of Models') +
  theme_light()# residuals distribution for all models with boxplot


# DALEX and variable importance with variable_importance() ----

library(ingredients)
rfImp <- feature_importance(rfExplainer, loss_function = loss_root_mean_square)
svmImp <- feature_importance(svmExplainer, loss_function = loss_root_mean_square)
marsImp <- feature_importance(marsExplainer, loss_function = loss_root_mean_square)

plot(rfImp, svmImp, marsImp) +
  theme_light() # despite some models have intrinsic calculation technique for
                # variable importance, DALEX is also proven to be handy...

# Ceteris Paribus and Partial Dependence Plots ----

rfCP <- ingredients::ceteris_paribus(rfExplainer, testSet[1:50,]) # first 50 observations of test-set
svmCP <- ingredients::ceteris_paribus(svmExplainer, testSet[1:50,])
marsCP <- ingredients::ceteris_paribus(marsExplainer, testSet[1:50,])

plot(svmCP, alpha = .5, color = "gray40") # SVM profile is taken as an example 
  
svmPDP <- aggregate_profiles(svmCP) # partial dependence plots  
plot(svmPDP, alpha = .7, color = "firebrick") +
  ggtitle('Partial Dependence Plots of Variables') +
  theme_bw()

plot(rfCP, variables = c('day', 'month'), only_numerical = FALSE) # ceteris paribus profiles for categorical predictors


