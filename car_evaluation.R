#load packages
library(data.table)
library(ggplot2)
library(tidyverse)
library(caret)
library(DMwR)
library(doSNOW)
set.seed(12345)

#load data
data <- fread("data.csv",na.strings = c(""," ","?","NA",NA))

#View data & modify column names
head(data)
names(data) <- c("buying","maint","doors","persons","lug_boot","safety","target")

data <- data %>% 
            mutate_if(is.character,as.factor)

data$buying <- factor(data$buying, levels = c("low", "med", "high","vhigh"))
data$maint <- factor(data$maint, levels = c("low", "med", "high","vhigh"))
data$lug_boot <- factor(data$lug_boot, levels = c("small", "med", "big"))
data$safety <- factor(data$safety, levels = c("low", "med", "high"))
data$target <- factor(data$target, levels = c("unacc", "acc", "good","vgood"))

ggplot(data, aes(buying, ..count..)) + 
  geom_bar(aes(fill = target), position = "dodge")

ggplot(data, aes(maint, ..count..)) + 
  geom_bar(aes(fill = target), position = "dodge")

ggplot(data, aes(doors, ..count..)) + 
  geom_bar(aes(fill = target), position = "dodge")

ggplot(data, aes(persons, ..count..)) + 
  geom_bar(aes(fill = target), position = "dodge")

ggplot(data, aes(lug_boot, ..count..)) + 
  geom_bar(aes(fill = target), position = "dodge")

ggplot(data, aes(safety, ..count..)) + 
  geom_bar(aes(fill = target), position = "dodge")


#Sample Indetestes
indexes = sample(1:nrow(data), size=0.2*nrow(data))

# Split data
test = data[indexes,]
dim(test)
train = data[-indexes,]
dim(train)
target<-test$target
test$target<-NULL

metrics_fun<-function(cm){
  n = sum(cm) #number of observations
  nc = nrow(cm) #number of clases
  diag = diag(cm) # correct predictions 
  rowsums = apply(cm, 1, sum) # number of observations per class
  colsums = apply(cm, 2, sum) # number of predictions per class
  
  #Accuracy 
  sum(diag/n)
  #Precision Recall F1 for each class
  precision = diag / colsums 
  recall = diag / rowsums 
  f1 = 2 * precision * recall / (precision + recall) 
  #data.frame(precision, recall, f1) 
  
  #Macroaverage F score
  
  macroPrecision = mean(precision)
  macroRecall = mean(recall)
  macroF1 = mean(f1)
  cat("MacroF1:",macroF1)
  #data.frame(macroPrecision, macroRecall, macroF1)
  
  #One vs All
  
  oneVsAll = lapply(1 : nc,
                    function(i){
                      v = c(cm[i,i],
                            rowsums[i] - cm[i,i],
                            colsums[i] - cm[i,i],
                            n-rowsums[i] - colsums[i] + cm[i,i]);
                      return(matrix(v, nrow = 2, byrow = T))})
  #oneVsAll
  
  sumOfAll = matrix(0, nrow = 2, ncol = 2)
  for(i in 1 : nc){sumOfAll = sumOfAll + oneVsAll[[i]]}
  #sumOfAll
  
  #Average Accuracy
  avgAccuracy = sum(diag(sumOfAll)) / sum(sumOfAll)
  
  #Microaverage F score
  micro_prf = (diag(sumOfAll) / apply(sumOfAll,1, sum))[1]
  cat("\nMicro F1 score:",micro_prf)
}

###Bootstrap

set.seed(12345)
cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

# define training control
train_control <- trainControl(method="boot632", number=100,sampling = "up")
# train the model
model1 <- caret::train(target~., data=train, trControl=train_control,
               method="rf")
# summarize results
model1.pred <- predict(model1,test)

confmat1 = as.matrix(table(Actual = target, Predicted = model1.pred))

stopCluster(cl)

# define training control
set.seed(12345)
c2 <- makeCluster(6, type = "SOCK")
registerDoSNOW(c2)
train_control <- trainControl(method="boot632", number=100,sampling = "down")

model2 <- caret::train(target~., data=train, trControl=train_control,
                      method="rf")
# summarize results
model2.pred <- predict(model2,test)

confmat2 = as.matrix(table(Actual = target, Predicted = model2.pred))
stopCluster(c2)

# define training control
set.seed(12345)
c3 <- makeCluster(6, type = "SOCK")
registerDoSNOW(c3)
train_control <- trainControl(method="boot632", number=100,sampling = "smote")

model3 <- caret::train(target~., data=train, trControl=train_control,
                      method="rf")
# summarize results
model3.pred <- predict(model3,test)

confmat3 = as.matrix(table(Actual = target, Predicted = model3.pred))
stopCluster(c3)

#Repeated CV
set.seed(12345)
c4 <- makeCluster(6, type = "SOCK")
registerDoSNOW(c4)
ctrl <- trainControl(method = "repeatedcv",
                     number=10,
                     repeats = 3,
                     sampling = "up")
# train the model

model4 <- caret::train(target~., data=train, trControl=ctrl,
                      method="rf")
# summarize results
model4.pred <- predict(model4,test)

confmat4 = as.matrix(table(Actual = target, Predicted = model4.pred))
stopCluster(c4)

#Repeated CV

set.seed(12345)
c5 <- makeCluster(6, type = "SOCK")
registerDoSNOW(c5)
ctrl <- trainControl(method = "repeatedcv",
                     number=10,
                     repeats = 3,
                     sampling = "down")



# train the model
model5 <- caret::train(target~., data=train, trControl=ctrl,
                       method="rf")
# summarize results
model5.pred <- predict(model5,test)

confmat5 = as.matrix(table(Actual = target, Predicted = model5.pred))
stopCluster(c5)

####Repeated CV with smote
set.seed(12345)
c6 <- makeCluster(6, type = "SOCK")
registerDoSNOW(c6)
ctrl <- trainControl(method = "repeatedcv",
                     number=10,
                     repeats = 3,
                     sampling = "smote")


# train the model

model6 <- caret::train(target~., data=train, trControl=ctrl,
                       method="rf")
# summarize results
model6.pred <- predict(model6,test)

confmat6 = as.matrix(table(Actual = target, Predicted = model6.pred))
stopCluster(c6)

#k fold CV
set.seed(12345)
c7 <- makeCluster(6, type = "SOCK")
registerDoSNOW(c7)
ctrl <- trainControl(method = "cv",
                     number=10,
                     sampling = "up")



# train the model

model7 <- caret::train(target~., data=train, trControl=ctrl,
                       method="rf")
# summarize results
model7.pred <- predict(model7,test)

confmat7 = as.matrix(table(Actual = target, Predicted = model7.pred))
stopCluster(c7)

####k fold CV
set.seed(12345)
c8 <- makeCluster(6, type = "SOCK")
registerDoSNOW(c8)
ctrl <- trainControl(method = "cv",
                     number=10,
                     sampling = "down")


# train the model

model8 <- caret::train(target~., data=train, trControl=ctrl,
                       method="rf")
# summarize results
model8.pred <- predict(model8,test)

confmat8 = as.matrix(table(Actual = target, Predicted = model8.pred))
stopCluster(c8)

####k fold CV with smote
set.seed(12345)
c9 <- makeCluster(6, type = "SOCK")
registerDoSNOW(c9)
ctrl <- trainControl(method = "cv",
                     number=10,
                     sampling = "smote")


# train the model

model9 <- caret::train(target~., data=train, trControl=ctrl,
                       method="rf")
# summarize results
model9.pred <- predict(model9,test)

confmat9 = as.matrix(table(Actual = target, Predicted = model9.pred))
stopCluster(c9)

metrics_fun(confmat1)
# MacroF1: 0.8480236
# Micro F1 score: 0.9478261
metrics_fun(confmat2)
# MacroF1: 0.5583136
# Micro F1 score: 0.6985507
metrics_fun(confmat3)
# MacroF1: 0.565931
# Micro F1 score: 0.7913043
metrics_fun(confmat4)
# MacroF1: 0.8560097
# Micro F1 score: 0.942029
metrics_fun(confmat5)
# MacroF1: 0.6303809
# Micro F1 score: 0.7391304
metrics_fun(confmat6)
# MacroF1: 0.6530739
# Micro F1 score: 0.8347826
metrics_fun(confmat7)
# MacroF1: 0.8532135
# Micro F1 score: 0.9449275
metrics_fun(confmat8)
# MacroF1: 0.613031
# Micro F1 score: 0.7275362
metrics_fun(confmat9)
# MacroF1: 0.6182471
# Micro F1 score: 0.8347826

### Our dataset is Imbalanced. So microaverage F1 score is  preferable.
### One has the best microaverage F1 score. That is 
### 10 fold cv with up sampling works best.

#Now, lets compare different algorithms

#k fold CV
set.seed(12345)
c201 <- makeCluster(6, type = "SOCK")
registerDoSNOW(c201)
ctrl <- trainControl(method="boot632", number=100,sampling = "up")

# train the model

model_xgbtree <- caret::train(target~., data=train, trControl=ctrl,
                       method="xgbTree")
# summarize results
model_xgbtree.pred <- predict(model_xgbtree,test)

confmat_xgbtree = as.matrix(table(Actual = target, Predicted = model_xgbtree.pred))
stopCluster(c201)

#k fold CV
set.seed(12345)
c202 <- makeCluster(6, type = "SOCK")
registerDoSNOW(c202)
ctrl <- trainControl(method="boot632", number=100,sampling = "up")

# train the model

model_xgblinear <- caret::train(target~., data=train, trControl=ctrl,
                              method="xgbLinear")
# summarize results
model_xgblinear.pred <- predict(model_xgblinear,test)

confmat_xgblinear = as.matrix(table(Actual = target, Predicted = model_xgblinear.pred))
stopCluster(c202)

#k fold CV
set.seed(12345)
c203 <- makeCluster(6, type = "SOCK")
registerDoSNOW(c203)
ctrl <- trainControl(method="boot632", number=100,sampling = "up")

# train the model

model_gbm <- caret::train(target~., data=train, trControl=ctrl,
                                method="gbm")
# summarize results
model_gbm.pred <- predict(model_gbm,test)

confmat_gbm = as.matrix(table(Actual = target, Predicted = model_gbm.pred))
stopCluster(c203)

#k fold CV
set.seed(12345)
c204 <- makeCluster(6, type = "SOCK")
registerDoSNOW(c204)
ctrl <- trainControl(method="boot632", number=100,sampling = "up")

# train the model

model_svm <- caret::train(target~., data=train, trControl=ctrl,
                          method="svmLinear")
# summarize results
model_svm.pred <- predict(model_svm,test)

confmat_svm = as.matrix(table(Actual = target, Predicted = model_svm.pred))
stopCluster(c204)

metrics_fun(confmat_xgbtree)
## MacroF1: 0.9753548
## Micro F1 score: 0.9942029
metrics_fun(confmat_xgblinear)
## MacroF1: 0.997976
## Micro F1 score: 0.9971014
metrics_fun(confmat_gbm)
## MacroF1: 0.9255857
## Micro F1 score: 0.9623188
metrics_fun(confmat_svm)
## MacroF1: 0.8671832
## Micro F1 score: 0.9304348






