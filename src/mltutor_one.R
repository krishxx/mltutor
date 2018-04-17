library(dummies)
library(psych)

#preparing data
#reading salesdata into a dataframe
salesdata <- read.csv("D:\\projects\\machineLearning\\mltutor\\sales_data_sample.csv", header = TRUE, sep=",")

#removing columns 1, 6 and then 13 to 25
salesdata <- salesdata[ -c(1, 6, 13:25)]

'''
hist(salesdata$SALES)
hist(salesdata$PRICEEACH)
plot(salesdata$PRICEEACH, salesdata$SALES)
cor(salesdata$PRICEEACH, salesdata$SALES)
hist(salesdata$STATUS)
'''

#one hot encoding method for $PRODUCTLINE, $STATUS
PRODUCTLINE_ = factor(salesdata$PRODUCTLINE) 
dumm = as.data.frame(model.matrix(~PRODUCTLINE_)[,-1])
salesdata = cbind(salesdata, dumm)

#str(salesdata)
STATUS_ = factor(salesdata$STATUS) Y
dumm = as.data.frame(model.matrix(~STATUS_)[,-1])
salesdata = cbind(salesdata, dumm) 
str(salesdata)

#removing STATUS, PRODUCT line original categorical variables
salesdata <- salesdata[-c(5,9)]

#multicollinearity
pairs.panels(salesdata[,-21],
             gap=0,
             bg=c("red", "yellow", "blue")[salesdata$SALES],
             pch=21)

#Principal Component Analysis
pc <- prcomp(salesdata[,-6],
             center=TRUE,
             scale.=TRUE)

attributes(pc)
print(pc)

#TODO

#try with linear regression
#Split dataset into "training" (80%) and "validation" (20%)
ind <- sample(2, nrow(salesdata), replace=TRUE, prob = c(0.8,0.2))

#training data
tdata1 <- salesdata[ind==1,]
#validating data
vdata1 <- salesdata[ind==2,]

#multiple linear regression model
results <- lm(SALES~., tdata1)
summary(results)

pred <- predict(results, vdata1)

head(pred)
head(vdata1$SALES)