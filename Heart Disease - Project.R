library(ISLR2)
library(tree)
library(ggplot2)
library(cowplot)
library(randomForest)

# DATA PREPROCESSING

url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
data <- read.csv(url, header =  FALSE)
head(data)
colnames(data) <- c(
    "age",
    "sex",# 0 = female, 1 = male
    "cp", # chest pain 
    # 1 = typical angina, 
    # 2 = atypical angina, 
    # 3 = non-anginal pain, 
    # 4 = asymptomatic
    "trestbps", # resting blood pressure (in mm Hg)
    "chol", # serum cholestoral in mg/dl
    "fbs",  # fasting blood sugar if less than 120 mg/dl, 1 = TRUE, 0 = FALSE
    "restecg", # resting electrocardiographic results
    # 1 = normal
    # 2 = having ST-T wave abnormality
    # 3 = showing probable or definite left ventricular hypertrophy
    "thalach", # maximum heart rate achieved
    "exang",   # exercise induced angina, 1 = yes, 0 = no
    "oldpeak", # ST depression induced by exercise relative to rest
    "slope", # the slope of the peak exercise ST segment 
    # 1 = upsloping 
    # 2 = flat 
    # 3 = downsloping 
    "ca", # number of major vessels (0-3) colored by fluoroscopy
    "thal", # this is short of thalium heart scan
    # 3 = normal (no cold spots)
    # 6 = fixed defect (cold spots during rest and exercise)
    # 7 = reversible defect (when cold spots only appear during exercise)
    "hd" # (the predicted attribute) - diagnosis of heart disease 
    # 0 if less than or equal to 50% diameter narrowing
    # 1 if greater than 50% diameter narrowing
)

head(data)

str(data)

sapply(data, function(x) sum(is.na(x)))

unique(data$ca)
unique(data$thal)

data$ca[data$ca == '?'] <- NA
data$ca[data$thal == '?'] <- NA
data<- data %>%
    mutate(across(c(sex, cp, fbs, restecg, exang, slope, ca, thal), factor))
str(data)

data$sex <- if_else(data$sex == 0, 'F', 'M')
data$sex <- as.factor(data$sex)
head(data)
data$hd <- if_else(data$hd == 0, 'Healthy', 'Unhealthy')
data$hd <- as_factor(data$hd)

str(data)

#impute the NAs using rfimpute

set.seed(42)
data.imputed <- rfImpute(hd ~ ., data = data, iter = 6)

# RANDOM FOREST

rf_model <- randomForest(hd ~ ., data = data.imputed, proximity = TRUE)

rf_model

# LOGISTIC REGRESSION

lg_model <- glm(hd ~ ., data = data.imputed, family = binomial)
summary(lg_model)

dim(data.imputed)
train <- sample_frac(data.imputed, size = 0.7)
lg.fit <- glm(hd ~ ., family = binomial, data = train)

# Getting the training accuracy 
glm.probs <- predict(lg.fit, type = "response")
glm.probs[1: 5]
dim(train)
contrasts(train$hd)
glm.pred <- rep("Healthy", 212)
glm.pred[glm.probs > 0.5] <- "Unhealthy"
table(glm.pred, train$hd)
mean(glm.pred == train$hd) # This gives us the training accuracy rate
mean(glm.pred != train$hd) # This gives us the training error rate

# Getting the test accuracy

test <- anti_join(data.imputed, train) #assinging our test data
dim(test)
glm.probs.test <- predict(lg.fit, newdata = test, type = 'response')
contrasts(test$hd)
glm.pred_test <- rep("Healthy", 91)
glm.pred_test[glm.probs.test > 0.5] <- "Unhealthy"
table(glm.pred_test, test$hd) # This gives us the test confusion matrix
mean(glm.pred_test == test$hd) # this gives us the test accuracy
mean(glm.pred_test != test$hd) # this gives us the test error rate

# Using cross validation to check test error rate improvement
library(boot)
cv.error <- cv.glm(data.imputed, lg_model, K = 10)$delta[1]
cv.error # this gives us the test error rate
# test error rate improves from 21% to 13%
