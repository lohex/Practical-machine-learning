---
title: "Practical Machine Learning"
author: "Lorenz Hexemer"
date: "14 8 2022"
output: 
  html_document: 
    keep_md: yes
---



# Executive Summary

Accelerometers can be used to quantify body movement. The data used in this project is collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal of this project is to predict the manner in which they did these exercises. To achieve this, after an exploratory data analysis, different machine learning models will be applied and compared in their performance in prediction. Cross validation will be used to estimate the out of sample error.



# Data import and cleaning


```r
  full_data <- read.csv("pml-training.csv",  na.strings=c("NA","#DIV/0!", ""))
  full_data <- full_data[,colSums(is.na(full_data)) == 0]
  full_data <- full_data[,-c(1:7)]
  
  varNames <- names(full_data)
  for (i in 1:(length(varNames)-1)) {
    full_data[,i] <- as.numeric(full_data[,i])
  }
  
  subSamples <- createDataPartition(y=full_data$classe, p=0.75, list=FALSE)
  subTraining <- full_data[subSamples, ] 
  subTesting <- full_data[-subSamples, ]
  table(subTraining$classe)
```

```
## 
##    A    B    C    D    E 
## 4185 2848 2567 2412 2706
```
First, variables that contain only NaNs are removed. Moreover, the X variable is removed since the data set is sorted according to the classes. Therefore, this variable implicitly labeles the classes. Moreover, the first 6 variables ("user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp",       "new_window" and "num_window") are removed since they do not encode data on the movements. 

The 5 classes A to E correspond to the different ways, the barbell lifts were performed. The classes are unevenly represented in the dataset: Class A is occuring signifficantly more often than the other classes.

# Data analysis

## Exploratory Data Analysis

![](PracticalMachineLearning_files/figure-html/r-1.png)<!-- -->

Many variables are uncorrelated and hence probably contribute independent information. In No sigle variable or combination of two variables, a clear separation of the classes is observable.


```r
pcaResult <- prcomp(subTraining[,-53],scale=TRUE);
f3 <- fviz_eig(pcaResult);
f4 <- fviz_pca_ind(pcaResult,col.ind = subTraining$classe, geom = "point");
f5 <- fviz_pca_var(pcaResult);
```
The first two prinicpal components contribute ~ 15% of explained variability. Still, the classes seem not easily seperatable in the PCA (figures see appendix).

## Modeling

Different models are trained on the same data: A logistic regression model, decision trees with and without boosting and random forest.
In the next section, the best model is slected.

### Logistic Model


```r
ctrl <- trainControl(method = "repeatedcv", 
                     repeats = 1, 
                     classProbs = TRUE,
                     #preProcOptions =  list(pcaComp = 46),
                     summaryFunction = multiClassSummary,
                     verbose = FALSE)

logitModel <- train(classe ~ ., subTraining, 
                     method = "multinom", 
                     family=binomial, 
                     metric = "Accuracy",  
                     verbose = FALSE, 
                     preProcess=c("center", "scale"), #,"pca"
                     trControl = ctrl)

predictTest <- predict(logitModel,subTesting)
logitConfusion <- confusionMatrix(subTesting$classe, predictTest)
logitOSPR <- mean(predictTest != subTesting$classe)
print(logitConfusion)
```

## Decision Tree

As an alternative, a decision tree model is fitted.


```r
library(rpart)
decisionTree <- rpart(classe ~ ., data=subTraining, method="class");
predictTest <- predict(decisionTree,subTesting, type = "class")
treeOSPR <- mean(predictTest != subTesting$classe)
treeConfusion <- confusionMatrix(predictTest, subTesting$classe);
print(treeConfusion)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1250  205   25   87   41
##          B   40  517   32   22   61
##          C   33  105  687  124  108
##          D   45   72   59  513   59
##          E   27   50   52   58  632
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7339          
##                  95% CI : (0.7213, 0.7462)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6618          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8961   0.5448   0.8035   0.6381   0.7014
## Specificity            0.8980   0.9608   0.9086   0.9427   0.9533
## Pos Pred Value         0.7774   0.7693   0.6500   0.6858   0.7717
## Neg Pred Value         0.9560   0.8979   0.9563   0.9300   0.9341
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2549   0.1054   0.1401   0.1046   0.1289
## Detection Prevalence   0.3279   0.1370   0.2155   0.1525   0.1670
## Balanced Accuracy      0.8970   0.7528   0.8561   0.7904   0.8274
```

The performance of decision trees can be improved by boosing.


```r
ctrl <- trainControl(method = "repeatedcv", 
                     repeats = 5, 
                     classProbs = TRUE,
                     #preProcOptions =  list(pcaComp = 46),
                     summaryFunction = multiClassSummary)

boostedTree <- train(classe ~ ., subTraining, 
                     method = "gbm", 
                     metric = "Accuracy",  
                     verbose = FALSE, 
                     preProcess=c("center", "scale"), #,"pca"
                     trControl = ctrl)

predictTest <- predict(boostedTree,subTesting)
boostTreeOSPR <- mean(predictTest != subTesting$classe)
boostTreeConfusion <- confusionMatrix(subTesting$classe, predictTest);
print(boostTreeConfusion)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1370   15    7    2    1
##          B   23  896   26    3    1
##          C    0   26  821    8    0
##          D    0    6   27  768    3
##          E    3    9    4   25  860
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9615          
##                  95% CI : (0.9557, 0.9667)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9513          
##                                           
##  Mcnemar's Test P-Value : 1.997e-07       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9814   0.9412   0.9277   0.9529   0.9942
## Specificity            0.9929   0.9866   0.9915   0.9912   0.9898
## Pos Pred Value         0.9821   0.9442   0.9602   0.9552   0.9545
## Neg Pred Value         0.9926   0.9858   0.9842   0.9907   0.9988
## Prevalence             0.2847   0.1941   0.1805   0.1644   0.1764
## Detection Rate         0.2794   0.1827   0.1674   0.1566   0.1754
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9871   0.9639   0.9596   0.9720   0.9920
```
### Random Forest


```r
randomForestModel <- randomForest(classe ~ ., data=subTraining, method="class")
predictTest <- predict(randomForestModel,subTesting)
randForestOSPR <- mean(predictTest != subTesting$classe)
randForestConfusion <- confusionMatrix(subTesting$classe, predictTest)
print(randForestConfusion)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    1    0    0    0
##          B    8  940    1    0    0
##          C    0    3  850    2    0
##          D    0    0   12  791    1
##          E    0    0    0    3  898
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9937         
##                  95% CI : (0.991, 0.9957)
##     No Information Rate : 0.2859         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.992          
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9943   0.9958   0.9849   0.9937   0.9989
## Specificity            0.9997   0.9977   0.9988   0.9968   0.9993
## Pos Pred Value         0.9993   0.9905   0.9942   0.9838   0.9967
## Neg Pred Value         0.9977   0.9990   0.9968   0.9988   0.9998
## Prevalence             0.2859   0.1925   0.1760   0.1623   0.1833
## Detection Rate         0.2843   0.1917   0.1733   0.1613   0.1831
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9970   0.9967   0.9918   0.9953   0.9991
```


# Model Selection and Discussion

To compare the performance, the out of sample error was used. The out of sample error was  estimated by evaluating the prediction performance on the test-dataset which was not used for fittig. 


```r
OSPR <- data.frame(Model=c("Logistic Model","Decision Tree","Boosted Decision Tree","Random Forest"),OSPR=c(logitOSPR,treeOSPR,boostTreeOSPR,randForestOSPR))
ggplot(OSPR,aes(y=OSPR,x=Model)) + geom_bar(stat="identity")
```

![](PracticalMachineLearning_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

The Decision Tree model did not perform better than the logistic model. The gradient boosting, though, improved the performance of the Decision Tree to achieve a out of sample error < 5%. Finally, the Random Forest model is selected for the prediction since its out of sample error is estimated to be < 1%.

# Prediction

In this section, the best model is used to predict the outcome of the 20 cases from the independent testing dataset.


```r
  full_data <- read.csv("pml-testing.csv",  na.strings=c("NA","#DIV/0!", ""))
  full_data <- full_data[,colSums(is.na(full_data)) == 0]
  full_data <- full_data[,-c(1:7)]
  
  varNames <- names(full_data)
  for (i in 1:(length(varNames)-1)) {
    full_data[,i] <- as.numeric(full_data[,i])
  }
  predict(randomForestModel,full_data)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


# Appendix

```r
f2
```

![](PracticalMachineLearning_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

```r
f3
```

![](PracticalMachineLearning_files/figure-html/unnamed-chunk-7-2.png)<!-- -->

```r
f4
```

![](PracticalMachineLearning_files/figure-html/unnamed-chunk-7-3.png)<!-- -->

```r
f5
```

![](PracticalMachineLearning_files/figure-html/unnamed-chunk-7-4.png)<!-- -->

