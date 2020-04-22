setwd("C:/Users/12177/Desktop/CS498/HW2.4")
#install.packages("caret", dependencies = TRUE)
#install.packages("randomForest")

library(caret)
library(randomForest)

data1<-read.table("C:/Users/12177/Desktop/CS498/HW2.4/agaricus-lepiota.data",sep="," ,header = FALSE)

data1[data1 == "?"]<- NA
data1 <- data1[!(data1$V12 %in% c(NA)),]

#label <-data1[,1] #select first column as label
#data1_feat <-data1[2:23] #select columns 1-30 as features

# Split into Train and Validation sets
# Training Set : Validation Set = 70 : 30 (random)
set.seed(100)
train <- sample(nrow(data1), 0.7*nrow(data1), replace = FALSE)
TrainSet <- data1[train,]
ValidSet <- data1[-train,]
#summary(TrainSet)
#summary(ValidSet)
#set.seed(42)
#data.imputed<- rfImpute(V1 ~ ., data = TrainSet, iter=2)
model <- randomForest(V1 ~ ., data=TrainSet)

# Predicting on Validation set
pred = predict(model,newdata = ValidSet[-1])
cm = table(ValidSet[,1],pred)

                  
