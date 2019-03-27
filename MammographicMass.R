
# Building Various Predictive Models for the
# Mammographic Mass Dataset provided @ UCI Machine Learning
# Repository. 


### Importing and Setting Up the Data Properly ###

setwd("~/R/UCI_ML_Repo/MammographicMass")

require(tidyverse)
require(caret)
require(Matrix)
require(MatrixModels)
require(pROC)
require(party)
require(doMC)
require(kernlab)

mass <- read.csv("mammographic.csv", header = TRUE)
mass[!complete.cases(mass),] # no missing value.

glimpse(mass)
map_dbl(mass, ~ length(unique(.)))

which(names(mass) == "Age")
categoricals <- names(mass[-2])

mass[categoricals] <- map(mass[categoricals], ~ as.factor(.))
glimpse(mass)

mass <- mass %>%
  mutate(Severity = as.factor(ifelse(Severity == 1, "malign", "benign")))

names(mass)[1] <- 'BiRads' # change the BI-RADS name, not suitable for creating modelmatrix
head(mass)

### Making Splits and Creating Control Object ###

set.seed(4221)
indices <- createDataPartition(mass$Severity, p = .7,
                               list = FALSE)

trainSet <- mass[indices,]
testSet <- mass[-indices,]

prop.table(table(trainSet[, "Severity"]))

trainX <- trainSet[, -6]
trainY <- trainSet[, 6]
testX <- testSet[, -6]
testY <- testSet[, 6]

trainModel <- as.matrix(sparse.model.matrix(~., data = trainX))
testModel <- as.matrix(sparse.model.matrix(~., data = testX))

ctrl <- trainControl(method = "repeatedcv",
                     n = 10,
                     repeats = 10,
                     summaryFunction = twoClassSummary,
                     savePredictions = TRUE,
                     classProbs = TRUE)

### Building Models ##
#-------------------#
registerDoMC(2)

# ------- Logistic Regression ------------

set.seed(1989)
glmFit <- train(x = trainModel,
                y = trainY,
                method = "glm",
                metric = "ROC",
                trControl = ctrl)

glmFit


# ------- Linear Discriminant Analysis ----------

set.seed(1989)
ldaFit <- train(x = trainModel[, -1],
                y = trainY,
                method = "lda",
                preProc = c('center', 'scale'),
                trControl = ctrl)

ldaFit # by removing intercept

# -------- Linear Support Vector Machines ---------

svmGridL <- expand.grid(C = 2^(-12:-4))

set.seed(1989)
svmFitL <- train(Severity ~.,
                 data = trainSet,
                 method = "svmLinear",
                 metric = "ROC",
                 preProc = c('center', 'scale'),
                 trControl = ctrl,
                 tuneGrid = svmGridL)

svmFitL
plot(svmFitL, scales = list(x = list(log = 2)),
     col = "navy")

# -------- Support Vector Machines with Radial Kernel --------

set.seed(744)
sigVal <- sigest(Severity ~., data = trainSet)

svmGridR <- expand.grid(sigma = sigVal[1],
                        C = 2^(-7:0))

set.seed(1989)
svmFitR <- train(Severity ~.,
                 data = trainSet,
                 method = "svmRadial",
                 metric = "ROC",
                 preProc = c('center', 'scale'),
                 tuneGrid = svmGridR,
                 trControl = ctrl)

svmFitR
plot(svmFitR, scales = list(x = list(log = 2)),
     col = "red")

# ---------- Creating ROC Objects for Models --------------

createRoc <- function(objFit) {
  objRoc <- pROC::roc(response = objFit$pred$obs,
                      predictor = objFit$pred$malign,
                      levels = rev(levels(objFit$pred$obs)))
  
  return(objRoc)
}

LogisticROC <- createRoc(glmFit)
ldaROC <- createRoc(ldaFit)
svmROC_Linear <- createRoc(svmFitL)
svmROC_Radial <- createRoc(svmFitR)

ggroc(list(LogReg = LogisticROC, LDA = ldaROC, 
           svmLinear = svmROC_Linear, svmRadial = svmROC_Radial),
      legacy.axes = TRUE, lty = 1, lwd = .75, aes = c('color')) + theme_minimal() +
  labs(title = "Multiple ROC for Various Models",
       x = "False Positive Rate",
       y = "True Positive Rate")


rocs <- tribble(~ name,
                   "LogisticROC",
                   "ldaROC",
                   "svmROC_Linear",
                   "svmROC_Radial")


singleRoc <- function(objRoc) {
  vis <- ggroc(objRoc, legacy.axes = TRUE,
               lty = 2, lwd = 1) +
    theme_minimal() +
    
    annotate(geom = "rect",
             xmin = .5, xmax = .75,
             ymin = .3, ymax = .5,
             fill = "red", alpha = .25) +
    
    annotate(geom = "text",
             x = .5, y = .4,
             hjust = 0,
             label = paste0("AREA: ", round(objRoc$auc, 4))) +
    
    xlab("False Positive Rate") +
    ylab("True Positive Rate")
  
  return(vis)
}

singleRoc(ldaROC)
singleRoc(svmROC_Linear) # and so forth...

# ---------- Generating Predictions for the Test Set ------------

predCompute <- function(model, test, testY) {
  preds <- predict(model, test, type = "raw")
  perf <- caret::postResample(preds, testY)
  return(perf)
}

logregAcc <- predCompute(glmFit, testModel, testY)
ldaAcc <- predCompute(ldaFit, testModel, testY)
svmLinAcc <- predCompute(svmFitL, testSet[, -6], testSet[, 6])
svmRadAcc <- predCompute(svmFitR, testSet[, -6], testSet[, 6])

accFrame <- as.data.frame(rbind(logregAcc, ldaAcc, svmLinAcc, svmRadAcc))


ggplot(accFrame, aes(x = reorder(rownames(accFrame), Accuracy), y = Accuracy)) +
  ggthemes::theme_economist() +
  geom_col(fill = "steelblue", width = .5) +
  expand_limits(y = c(0, 1)) +
  scale_y_continuous(breaks = seq(.7, 1, by = .05)) +
  coord_flip() +
  labs(title = "Accuracy Performance of Some Non-Tree Based Models",
       x = "Model-Accuracy",
       y = "Rate")


CM <- function(model, testFrame, testLabels){
  cm <- confusionMatrix(predict(model, testFrame), testLabels)
  
  return(cm)
}

CM(glmFit, testModel, testY)
CM(ldaFit, testModel, testY) # and so forth...

###
sessionInfo()
gc()
