
##

# Machine Learning Session with a Open Dataset
# provided by UCI Machine Learning Repository

# Source Credit:
# Ronny Kohavi and Barry Becker


# -----Adult Census Income-------

##
# Importing and Setting Up Variable Types

setwd("~/Downloads")

library(tidyverse)
library(caret)
library(data.table)
library(pROC)
library(doMC)


incomeAdult <- fread("income.csv")
glimpse(incomeAdult)

incomeAdult[!complete.cases(incomeAdult),]

incomeAdult <- as.data.frame(apply(incomeAdult, 2, function(x) gsub("\\?", "Unknown", x)))
glimpse(incomeAdult)

floats <- c('age', 'fnlwgt', 'education.num',
            'capital.gain', 'hours.per.week',
            'capital.loss')


incomeAdult[floats] <- map(incomeAdult[floats], ~ as.numeric(.))
glimpse(incomeAdult)

# apply(df, 2 , function(x) length(unique(x)))

incomeAdult %>%
  mutate(income = case_when(
    income == "<=50K" ~ "low",
    income == ">50K" ~ "high",
    TRUE ~ "Unk"
  ))  -> incomeAdult

glimpse(incomeAdult)
incomeAdult$income <- as.factor(incomeAdult$income)

###
##
# Splitting the Data into Training and Test Set

set.seed(4326)
inTr <- createDataPartition(incomeAdult$income,
                            p = .7,
                            list = FALSE)



trainSet <- incomeAdult[inTr,]
testSet <- incomeAdult[-inTr,]

prop.table(table(incomeAdult$income))
levels(trainSet$income)

set.seed(4042)
indx <- sample(1:nrow(trainSet), size = nrow(trainSet)*.7,
               replace = FALSE)

###
##
# Creating Appropriate Set Type for Linear Models

which(names(trainSet) == "income")  # [15]
which(names(testSet) == "income")  # [15]


trainX = trainSet[, -15]
trainY = trainSet[, 15]
testX = testSet[, -15]
testY = testSet[, 15]

library(Matrix)
library(MatrixModels)
trainXtrans <- as.matrix(sparse.model.matrix(~ ., data = trainX))
testXtrans <- as.matrix(sparse.model.matrix(~ ., data = testX))


isZV <- nearZeroVar(trainXtrans)
trainXtrans <- trainXtrans[, -isZV]
testXtrans <- testXtrans[, -isZV]


# corVal <- cor(trainXtrans)
# highCor <- findCorrelation(corVal, cutoff = .9) # 20 th column
# 
# names(as.data.frame(trainXtrans))[20] # occupationUnknown


###
## Creating the Control Object

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     savePredictions = TRUE,
                     classProbs = TRUE,
                     index = list(TrainSet = indx))

### --------- Linear Models ---------
##
# Linear Discriminant Analysis

set.seed(1025)
ldaFit <- train(x = trainXtrans,
                y = trainY,
                method = "lda",
                preProc = c('center', 'scale'),
                trControl = ctrl,
                metric = "ROC")

ldaFit

roc(response = ldaFit$pred$obs,
    predictor = ldaFit$pred$high,
    levels = rev(levels(ldaFit$pred$obs))) -> ldaROC

###
##
# Logistic Regression

set.seed(1025)
glmFit <- train(x = trainXtrans,
                y = trainY,
                method = "glm",
                metric = "ROC",
                trControl = ctrl)

glmFit

glmROC <- roc(response = glmFit$pred$obs,
              predictor = glmFit$pred$high,
              levels = rev(levels(glmFit$pred$obs)))



### -------- NON-LINEAR MODELS -----------
##
# Linear Support Vector Machines

set.seed(1025)
svmFit       <- train(income ~.,
                      data = trainSet,
                      method = "svmLinear",
                      metric = "ROC",
                      trControl = ctrl,
                      preProc = c('center', 'scale'),
                      tuneLength = 5)


svmFit 

svmROC <- roc(response = svmFit$pred$obs,
              predictor = svmFit$pred$high,
              levels = rev(levels(svmFit$pred$obs)))


# par(pty = "square")    # BASE R METHOD OF PLOTTING ROC CURVES
# 
# plot.roc(svmROC, legacy.axes = TRUE, col = "blue")
# plot.roc(ldaROC, legacy.axes = TRUE, add = TRUE, col = "purple")
# plot.roc(glmROC, legacy.axes = TRUE, add = TRUE, col = "gray30")
# 
# legend("bottomright",
#        trace = TRUE,
#        cex = .5, lwd= 2, seg.len = .2,
#        legend = c("SVM", "LDA", "GLM"),
#        col = c("blue", "purple", "gray30"))


##
# Support Vector Machines with Radial Kernel

library(kernlab)

set.seed(442)
sigVal <- sigest(income ~.,
                 data = trainSet)

svmRGrid <- expand.grid(sigma = sigVal[1],
                       C = 2^(-5:3))


registerDoMC(2)
set.seed(1025)
svmRFit <- train(income ~.,
                data = trainSet,
                method = "svmRadial",
                metric = "ROC",
                trControl = ctrl,
                preProc = c('center', 'scale'),
                tuneGrid = svmGrid)



plot(svmFit,
     scales = list(x = list(log = 2)),
     col = "steelblue")


svmRROC <- roc(response = svmFit$pred$obs,
              predictor = svmFit$pred$high,
              levels = rev(levels(svmFit$pred$obs)))


plot(svmRROC, legacy.axes = TRUE,
     col = "purple")

svmRImp <- varImp(svmFit, scale = FALSE)
plot(svmRImp, top = 5)




          # A NOTE: Possibly due to the dataset being intrinsically linear,
          # SVM with Linear Boundaries has achieved superior AUC-ROC
          # value than SVM with Radial Kernel. So, it is recommended to
          # use Linear SVM for further analysis, and discard the Radial Kernel...
          # However, a detailed version of SVM-Radial was performed for demonstration


###
##
# K-Nearest Neighboorhood (KNN)

set.seed(1025)
knnFit <- train(income ~.,
                data = trainSet,
                method = "knn",
                preProc = c('center', 'scale'),
                metric = "ROC",
                tuneGrid = expand.grid(k = seq(13, 27, by = 2)),
                trControl = ctrl)

knnFit
plot(knnFit, col = "navy")

knnROC <- roc(response = knnFit$pred$obs,
              predictor = knnFit$pred$high,
              levels = rev(levels(knnFit$pred$obs)))


###
##
# -------------- TREE BASED MODELS -----------


# A Basic Decision Tree

library(party)

set.seed(1025)
treeFit <- ctree(income ~., data = trainSet,
                 controls = ctree_control(testtype = "MonteCarlo",
                                          maxdepth = 3,
                                          nresample = 500,
                                          minbucket = 100))

plot(treeFit)

###
##
# Bagged Trees - Bootstrap Aggregation

set.seed(1025)
baggedFit <- train(income ~.,
                   data = trainSet,
                   method = "treebag",
                   nbagg = 50,
                   metric = "ROC",
                   trControl = ctrl)

baggedFit

bagROC <- roc(response = baggedFit$pred$obs,
              predictor = baggedFit$pred$high,
              levels = rev(levels(baggedFit$pred$obs)))



###
## 
# Random Forests

rfGrid <- expand.grid(mtry = seq(4, 14, by = 2))

registerDoMC(2)
set.seed(1025)
rfFit <- train(income ~.,
               data = trainSet,
               method = "rf",
               ntree = 1000,
               metric = "ROC",
               importance = TRUE,
               tuneGrid = rfGrid,
               trControl = ctrl)

rfFit
plot(rfFit)

# rfImp <- varImp(rfFit, scale = FALSE)
# plot(rfImp, top = 5, col = "red")

# confusionMatrix(rfFit, norm = "none")

rfROC <- roc(response = rfFit$pred$obs,
             predictor = rfFit$pred$high,
             levels = rev(levels(rfFit$pred$obs)))




# Random Forests with Discrete Categorical Variables

# dim(as.data.frame(trainXtrans)) # 28 predictors
# 
# rfmatGrid <- expand.grid(mtry = seq(4, 18, by = 2))
# 
# 
# registerDoMC(2)
# set.seed(1025)
# rfmatFit <- train(x = trainXtrans,
#                   y = trainY,
#                   method = "rf",
#                   metric = "ROC",
#                   ntrees = 1000,
#                   importance = TRUE,
#                   tuneGrid = rfmatGrid,
#                   trControl = ctrl)
# 
# rfmatFit --- # roc: 0.8867
# plot(rfmatFit)

 # A NOTE: as pretty much expected, there is a decline in ROC value when compared to
 # the dataset with factors in it (i.e not discrete predictors as in model matrix,
                                  #but all categorical values show up under a predictor)
 
 # (above .9 for the ''trainSet'' and .8867 for the ''trainXtrans'')
 # This is possibly due to the fact that we eliminated the near zero variance
 # predictors for linear models, that might add some information when it comes down
 # to complicated and flexible models like random forests, GBM, etc.
 # Analysis with discrete categorical predictors is discarded for that reason.

###
##
# Gradient Boosting Machine

gbmGrid <- expand.grid(n.trees = 1000,
                       interaction.depth = c(3, 4),
                       shrinkage = c(.05, .1),
                       n.minobsinnode = 10)

registerDoMC(2)
set.seed(1025)
gbmFit <- train(income ~.,
                data = trainSet,
                method = "gbm",
                metric = "ROC",
                verbose = FALSE,
                trControl = ctrl,
                tuneGrid = gbmGrid)


gbmROC <- roc(response = gbmFit$pred$obs,
              predictor = gbmFit$pred$high,
              levels = rev(levels(gbmFit$pred$obs)))


###
##
#  ---*Multiple and One-by-One ROC Curves*---


visRoc <- ggroc(list(RF = rfROC, SVM = svmROC, LDA = ldaROC, 
                     LogReg = glmROC, Bagged = bagROC,
                     KNN = knnROC, GBM = gbmROC),
          legacy.axes = TRUE, aes = c("color")) +
          theme_bw() +
          labs(x = "False Positive Rate",
          y = "True Positive Rate",
          title = "ROC Comparison for Various Models") +
          geom_line(size = .75, alpha = 2/3)  
  

visRoc


indROC <- function(obj){
  ggroc(obj, legacy.axes = TRUE,
        lty = 1, col = "navy") +
    
    theme_minimal() +
    annotate(geom = "rect",
             xmin = .65, xmax = .95,
             ymin = .4, ymax = .6,
             fill = "red", alpha = .25) +
    
    annotate(geom = "text",
             x = .8:1, y = .5,
             label = paste0("AUC = ", round(obj$auc, 4))) +
    
    labs(x = "FPR", y = "TPR")
}

roc1 <- indROC(gbmROC) + ggtitle('GBM')
roc2 <- indROC(rfROC) + ggtitle('RF')
roc3 <- indROC(knnROC) + ggtitle('KNN')
roc4 <- indROC(ldaROC) + ggtitle('LDA')
roc5 <- indROC(glmROC) + ggtitle('LogReg')
roc6 <- indROC(bagROC) + ggtitle('Bagged')
roc7 <- indROC(svmROC) + ggtitle('SVM')

library(patchwork)
cumulativeRoc <- (roc1 | roc2 | roc3) / (roc4 | roc5 | roc6 | roc7)
cumulativeRoc


### 
##
# Predictiing the Test Set and Metric Comparisons

glmPred <- predict(glmFit, newdata = testXtrans, type = "raw")
ldaPred <- predict(ldaFit, newdata = testXtrans, type = "raw")
rfPred <- predict(rfFit, newdata = testSet[, -15], type = "raw")
bagPred <- predict(baggedFit, newdata = testSet[, -15], type = "raw")
svmPred <- predict(svmFit, newdata = testSet[, -15], type = "raw")
treePred <- predict(treeFit, newdata = testSet[, -15], type = "response")
gbmPred <- predict(gbmFit, newdata = testSet[, -15], type = "raw")
knnPred <- predict(knnFit, newdata = testSet[, -15], type = "raw")


postResults <- data.frame(
  Logistic = postResample(glmPred, testY),
  LDA = postResample(ldaPred, testY),
  SVMlinear = postResample(svmPred, testSet[, 15]),
  BaggedTrees = postResample(bagPred, testSet[, 15]),
  RandomForests = postResample(rfPred, testSet[, 15]),
  SingleTree = postResample(treePred, testSet[, 15]),
  BoostedTrees = postResample(gbmPred, testSet[, 15]),
  KNN = postResample(knnPred, testSet[, 15]))

postResults <- tidyr::gather(postResults, model, rate)
postResults$type <- c('Accuracy', 'Kappa')
str(postResults)

postResults$model <- as.factor(postResults$model)
postResults$rate <- as.numeric(postResults$rate)
postResults$type <- as.factor(postResults$type)

postResults$rate <- as.numeric(round(postResults$rate, 4))

postResults

postResults1 <- postResults %>%
  filter(type == "Accuracy")

vis1 <-  ggplot(postResults1, aes(x = reorder(model, rate), y = rate)) +
  ggthemes::theme_economist() +
  expand_limits(y = c(0, 1.0)) +
  scale_y_continuous(breaks = seq(0, 1, by = .1)) +
  geom_col(position = "dodge", fill = "firebrick",
           width = .5) +
  
  labs(title = "Accuracy Rate of Models",
       subtitle = "Head-to-Head Comparison",
       x = "Model",
       y = NULL) +
  coord_flip()

vis1

#######
# end of the project...

save.image("IncomeML.RData")

sessionInfo()
gc("no")

