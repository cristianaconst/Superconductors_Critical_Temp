# ####################################
# --Predicting critical temperature of superconductors--

# ####################################


# Plotting libraries
library(ggplot2)
#library(ggbiplot)
library(corrplot)

library(caret) # Cross-Validation
library(e1071) # SVM model
library(Metrics) # RSME metric


### Introduction

# Load dataset
original_data <- read.csv("Data/train.csv")
dim(original_data)

# Output:
# 21263    82

### Exploratory Data Analysis
ggplot(original_data, aes(x = critical_temp)) +
      geom_histogram(aes(y = ..density..), col = 'gray66', 
                     fill = 'gray', binwidth = 2.5) +
      labs(x = 'Critical Temperature', y = 'Density') +
      geom_density(col = 'red') +
      theme_minimal()


summary(original_data$critical_temp)


corrplot(cor(original_data), method = "square", order = "original", tl.pos = 'n')



### Dimension Reduction
# Principal Component Analysis
data_pca <- prcomp(original_data[, -ncol(original_data)], scale. = T)
summary(data_pca)


corrplot(cor(data_pca$x), method= "square", order = "original", tl.pos = 'n')

plot(summary(data_pca)$importance[3,], ylab = 'Cumulative Percentage')

# Create a dataframe containing the first 13 principal components
clean_data <- as.data.frame(data_pca$x[,1:13])

par(mfrow=(c(2,2)))
plot(x = clean_data$PC1, y = original_data$critical_temp, xlab = 'PC1', ylab = 'Critical Temperature', main = paste0('Correlation = ', round(cor(clean_data$PC1, original_data$critical_temp), 2)))
plot(x = clean_data$PC2, y = original_data$critical_temp, xlab = 'PC2', main = paste0('Correlation = ', round(cor(clean_data$PC2, original_data$critical_temp), 2)))
plot(x = clean_data$PC3, y = original_data$critical_temp, xlab = 'PC3', ylab = 'Critical Temperature', main = paste0('Correlation = ', round(cor(clean_data$PC3, original_data$critical_temp), 2)))
plot(x = clean_data$PC4, y = original_data$critical_temp, xlab = 'PC4', main = paste0('Correlation = ', round(cor(clean_data$PC4, original_data$critical_temp), 2)))


clean_data <- cbind(clean_data, critical_temp = original_data$critical_temp)

clean_data <- clean_data[clean_data$critical_temp != max(clean_data$critical_temp),]


### Supervised Learning
set.seed(50)

k <- 5 # Number of K-Folds

KFolds <- createFolds(clean_data$critical_temp, k = k) # Creating the folds

poly_fold_err <- vector(length = k) # Empty vector for the Polynomial SVM RSME
radial_fold_err <- vector(length = k) # Empty vector for the Radial SVM RSME


for (i in 1:k)
  {
   # Creating the testing set for the fold i
   train <- clean_data[KFolds[[i]],]
   
   # Creating the training set from the remaining data for the fold i
   test <- clean_data[-KFolds[[i]],]
   
   # Fit the models with the training set and tune for the best parameters
   poly_fit <- tune.svm(critical_temp ~ ., data = train, kernel = 'polynomial', 
                     degree = c(2, 3, 5), 
                     cost = c(0.01, 0.1, 0.2, 0.5), 
                     scales = c(1))
   radial_fit <- tune.svm(critical_temp ~ ., data = train, kernel = 'radial', 
                     sigma = c(0.05, 0.1, 0.5, 1),
                     cost = c(0.01, 0.1, 0.2, 0.5))
   
   # Use the fitted models to predict the testing set
   poly_predict <- predict(poly_fit$best.model, newdata = test)
   radial_predict <- predict(radial_fit$best.model, newdata = test)
   
   # Obtaining metric
   poly_fold_err[i] <- rmse(clean_data$critical_temp, poly_predict)
   radial_fold_err[i] <- rmse(clean_data$critical_temp, radial_predict)
   
   
  }


#Plotting
plot(poly_fit)
plot(radial_fit)
   
results <- matrix(c(poly_fold_err, radial_fold_err), ncol = 2)
results <- rbind(results, c(mean(poly_fold_err), mean(radial_fold_err)))
results