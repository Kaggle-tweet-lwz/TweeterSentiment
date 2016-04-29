library(ggplot2)
library(ggmap)
setwd("/Users/jinglinzong/Desktop/tweet")
tartu_housing_xy_wgs84_a <- read.csv("C:/ANTO/loengud/GIS Maps and Spatial Analyses for Urban Planning/data/accessibility/tartu_housing_xy_wgs84_a.csv", 
                                     sep = ";")

rf <- read.csv("y_rf.csv", header = FALSE)
gb <- read.csv("y_gb.csv", header = FALSE)
extraT <- read.csv("y_extratree.csv", header = FALSE)
ada <- read.csv("y_ada.csv", header = FALSE)
logridge <- read.csv("y_log.csv", header = FALSE)
y_svm <- read.csv("y_svm.csv", header = FALSE)
y_lasso <-read.csv("y_loglassp.csv", header = FALSE)
y <- read.csv("y_data.csv", header = FALSE)
coef <- read.csv("ensemble_coef2.csv", header = TRUE)
coef1 <- coef$weights_OLS
coef2 <- coef$weights_logit

data <- data.frame(rf, gb, extraT, ada, logridge, y_lasso, y_svm,y)
colnames(data) <- c("RandomForest", "GradientBoosting", "ExtraTree", "Adaboosting", "logisticRidge", "lasso", "linear svm", "y")
write.csv(data, "testProb.csv")

data2 <- data.frame(rf, gb, extraT, ada, logridge, y_lasso, y_svm)
data2 <- as.matrix(data2)
prob <- data2 %*% coef1[2:8] + coef1[1]
max(prob)

data3 <- data.frame(rf, gb, extraT, ada, logridge,y_lasso)
data3 <- as.matrix(data3)
prob2 <- data3 %*% coef2[2:8] + coef2[1]

coeflog <- read.csv("ensemble_coef1.csv", header = TRUE)
coef11 <- coeflog$Coef.4
coef12 <- coeflog$Coef3
probl <- data2 %*% coef11[2:7] + coef11[1]
probl2 <- data3 %*% coef12[2:6] + coef12[1]

avg <- rep(11/(12*7), times = 7)
avg[7] = 1/12
probb <- data2 %*% avg

probb2 <- data3 %*% rep(1/6, times = 6)


y1 <- as.numeric(prob > 0.5)
y2 <- as.numeric(prob2 > 0.5)
y3 <- as.numeric(probl > 0)
y4 <- as.numeric(probl2 > 0)
y <- data.frame(y1, y2, y3,y4)
y2 <- as.numeric(prob2 > 0)
y_t2 <- data.frame(y1, y2)
write.csv(y_t2, "y_secondTry.csv")

y_avg <- as.numeric(probb > 0.5)
y_avg2 <- as.numeric(probb2 > 0.5)
write.csv(y_avg, "y_avg2.csv")

pred1 <- read.csv("pred1.csv")
y1p <- pred1$y
sum(y_avg == pred1)/length(y_avg)
y1


y_neural <- read.csv("y_neural_predict.csv", header = FALSE)
sum(y_neural$V1 == pred1$y)/length(y_neural$V1)
write.csv(y_neural, "y_pred2.csv")
prob_neural <- read.table("y_neural.txt")
neural_prob <- prob_neural$V2
write.csv(neural_prob, "neural_prob.csv")

data4<- data.frame(rf, gb, extraT, ada, logridge, neural_prob)
avg <- (rep(1/6, 6))
prob4 <- as.matrix(data4) %*% avg
y_pred4 <- as.numeric(prob4 > 0.5)
sum(y_neural$V1 == y_pred4)/length(y_pred4)


