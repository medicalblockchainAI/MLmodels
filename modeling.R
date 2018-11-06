library(utils)
library(caret)

diabities=read.csv("/Users/mohammad/Box\ Sync/MedicalBlockchain/diabetes.csv")
diabities$Outcome=as.character(diabities$Outcome)
diabities=diabities[,c(1,6,8,9)]
# compare models ###################
set.seed(14)
train_control <- trainControl(method="repeatedcv",number=10, repeats=5)
m1 <- train(Outcome~., data=diabities, trControl=train_control, method="rf",importance=TRUE)
m2 <- train(Outcome~., data=diabities, trControl=train_control, method="svmLinear",importance=TRUE)
m3 <- train(Outcome~., data=diabities, trControl=train_control, method="nnet")
m4 <- train(Outcome~., data=diabities, trControl=train_control, method="kknn")
m5 <- train(Outcome~., data=diabities, trControl=train_control, method="gamboost")
m6 <- train(Outcome~., data=diabities, trControl=train_control, method="glmboost")
m7 <- train(Outcome~., data=diabities, trControl=train_control, method="lvq")

allModels=resamples(list(RandomForest=m1,SVM=m2,NeuralNet=m3,NearestNeighbor=m4,
              gamboost=m5,glmboost=m6,LearningVector=m7))
bwplot(allModels,scales=list(relation="free"))


# Feature Selection
importance <- varImp(m1, scale=FALSE)
plot(importance)
