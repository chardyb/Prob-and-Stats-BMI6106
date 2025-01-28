####January 15#####
####R Session Probability####

###1. We are going to use our use of R and probabilities to be able to determine which features help 
##to separate diabetes patietns

###First load the dataset

###Where is your dataset located? you can either import it from where it is located or set your working directory.
getwd()
setwd("/Users/carterhardy/spring\ 2025\ UU/Prob-and-Stats-BMI6106/Week2")

##load csv
diabetes = read.csv("diabetes.csv")

#Pregnancies: To express the Number of pregnancies
#Glucose: To express the Glucose level in blood
#BloodPressure: To express the Blood pressure measurement
#SkinThickness: To express the thickness of the skin
#Insulin: To express the Insulin level in blood
#BMI: To express the Body mass index
#DiabetesPedigreeFunction: To express the Diabetes percentage
#Age: To express the age
#Outcome: To express the final result 1 is Yes and 0 is No

###
###How many observations are there?
###How many features??
##Which are your dependent and independent variables?
##How many patients have diabetes?
##How many diabetes patients have a high BMI > 40
nrow(diabetes[diabetes$Outcome==1,])
#install.packages("Amelia",dependencies = T)

library(Amelia)
library(tidyverse)
library(psych)



###Let's look at the different features and what data type they belong to
str(diabetes)

table(diabetes$Pregnancies)
plot(density(diabetes$Glucose))
###Many packages give nice data summaries that are very useful while evaluating your data
psych::describe(diabetes)

?describe
##Frequency Distribution of the outcome

table(diabetes$Outcome)
table(diabetes$Glucose)

##
##Let's look at some of our features  distributions

hist(diabetes$Pregnancies)

##Facet plot of densities for features 
diabetes %>% 
  select(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,Age) %>% 
  gather(metric, value) %>% 
  ggplot(aes(value, fill = metric)) + 
  geom_density(show.legend = FALSE) + 
  facet_wrap(~ metric, scales = "free")

##Let's compare the distribution of Insulin on the different outcomes
diabetes %>%
  ggplot(aes(x=Outcome,y=Insulin, fill = Outcome)) +
  geom_boxplot() +theme_bw()+
  ggtitle("Box Plot")

##Convert the outcome variable to a factor variable.
diabetes$Outcome = as.factor(diabetes$Outcome)

range(diabetes$Age)

#visualize the missing data using a function from the Amelia package
missmap(diabetes)

#Convert '0' values into NA
diabetes[,2:7][diabetes[, 2:7] == 0] <- NA
head(diabetes,20)##Check the first 20 lines of the dataframe

##Get means of a column
mean(diabetes$Glucose)
mean(diabetes$Glucose,na.rm=TRUE)
sd = sd(diabetes$Glucose,na.rm=TRUE)

#define standard error of mean function
std.error <- function(x) {
  sd = sd(x,na.rm=TRUE)
  size = length(x[!is.na(x)])
  serr = sd/sqrt(size)
  return(serr)
}


#calculate standard error of the mean 
std.error(diabetes$Glucose)

apply(diabetes, 2, std.error)
#visualize the missing data
missmap(diabetes)

##How about the density plots 

diabetes %>% 
  select(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,Age) %>% 
  gather(metric, value) %>% 
  ggplot(aes(value, fill = metric)) + 
  geom_density(show.legend = FALSE) + 
  facet_wrap(~ metric, scales = "free")


#BMI>=40 and diabetes
nrow(diabetes[diabetes$Outcome==1 & diabetes$BMI>=40,]) ##146
###No diabetes and age < 30
##
nrow(diabetes[diabetes$Outcome==0 & diabetes$Age<30,]) ##834

###############The code below will guide you through the steps to run and visualize a naive bayes classifier

library(mice)
library(caret)
###We need to impute missing values to run tha naive bayes
##can use multiple approaches (k-mers, nearest neighbors)
mice_mod <- mice(diabetes[, c("Glucose","BloodPressure","SkinThickness","Insulin","BMI")], method='rf')
?mice
mice_complete <- complete(mice_mod)
?complete
##Let's transfer the imputed columns to the dataset
diabetes$Glucose = mice_complete$Glucose
diabetes$BloodPressure = mice_complete$BloodPressure
diabetes$SkinThickness = mice_complete$SkinThickness
diabetes$BMI = mice_complete$BMI
diabetes$Insulin = mice_complete$Insulin


##Any missing data
missmap(diabetes)
##Let's Build the naive bayes Classifier

#Building a model
#split data into training and test data sets you set the partition in this case we will do 75-25

set.seed(998)
indxTrain <- createDataPartition(y = diabetes$Outcome,p = 0.75,list = FALSE)
training <- diabetes[indxTrain,]
testing <- diabetes[-indxTrain,]

#Check dimensions of the split

prop.table(table(diabetes$Outcome)) * 100
prop.table(table(training$Outcome)) * 100
prop.table(table(testing$Outcome)) * 100

#create objects x which holds the predictor variables and y 
#which holds the response variables
x = training[,-9]
y = training$Outcome

###Run the naive bayes algorithm on the training dataset using a resampling method, remember the goal is to maximize
##The class prediction.
model = train(x,y,'naive_bayes',trControl=trainControl(method='cv',number=10))

#Model Evaluation
#Predict using the testing set
Predict <- predict(model,newdata = testing )



#Get the confusion matrix to see accuracy value and other parameter values

confusionMatrix(Predict, testing$Outcome )


#Plot Variable performance
X <- varImp(model)
plot(X)

###It appears that glucose concentration and age are the top two variables. Lets see a scatterplot of these two

testing$prediction = as.character(Predict)
testing$index = 1:nrow(testing)
testing$logic = ifelse((testing$Outcome==testing$prediction) & (testing$Outcome==0),0,
                       ifelse(testing$Outcome==testing$prediction & testing$Outcome==1,1,
                              ifelse(!(testing$Outcome==testing$prediction) & testing$Outcome==0,2,3)))

table(testing$logic)

ggplot(testing, aes(x=Age, y=Glucose,color = factor(logic))) + geom_point(size = 3)

##Check the mean of the factors for each classification 
##Remember 0 = TN no diabetes; 1 = TP diabetes; 2 = FP diabetes 3 = FN No diabetes

ggplot(testing, aes(x=factor(logic),y = Glucose,fill = factor(logic)))+
  geom_bar(stat = "summary",  fun = mean)

###Did the classifier struggle with these patients?

predict(model,newdata = testing,type = "prob")

##Plot ROC curve
library(pROC)
roc_ = roc(testing$Outcome,predict(model, newdata = testing, type ="prob")[,2])

plot(roc_,print.auc=T)




