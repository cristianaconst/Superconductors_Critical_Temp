# To be deleted before upload on GitHub

### Load libraries
# Plotting libraries
library(ggplot2)
library(corrplot)

library(Rtsne) # t-Distributed Stochastic Neighbor Embedding (t-SNE)
library(caret) # Cross-Validation
library(e1071) # SVM model
library(Metrics) # RSME metric



### Introduction

# Load dataset
original_data <- read.csv("Data/train.csv")
dim(original_data)


## Standardization
# Separate features and target
target <- original_data["critical_temp"]
original_data_features <- subset(original_data, select = -critical_temp)

# Standardize dataset
data_standardized <- scale(original_data_features)

# Check data for duplicates
duplicated_rows <- duplicated(data_standardized)
str(duplicated_rows) # It turns out that there are no duplicates

## Feature Engineering

# t-SNE
# Iterate through hyper-parameters and seeds
# Hyperparameter values
#perplexities <- c(5, 10, 25, 50)
#thetas <- c(0.1, 0.3, 0.5, 0.8)
#etas <- c(50, 100, 200, 500)
#max_iters <- c(500, 1000)


perplexities <- c(10, 25, 50)
thetas <- c(0.1, 0.3, 0.5, 0.8)
etas <- c(50, 100, 200, 500)


# Generate list of seeds
set.seed(123)
numb_seeds <- 5 # Number of seeds
seeds <- sample.int(1e3, numb_seeds) # Generate random values for the seeds
# c(415, 463, 179, 526, 195)


tSNE_iter_results <- list()

# Create a directory to save the plots
dir.create("tsne_plots", showWarnings = FALSE)

# Loop through each hyperparameter and seed
for (perplexity in perplexities) {
    for (theta in thetas) {
        for (eta in etas) {
           # for (max_iter in max_iters) {
                for (seed in seeds) {
                    # Run t-SNE
                    tsne_results <- Rtsne(data_standardized,
                                            check_duplicates = FALSE,
                                            perplexity = perplexity,
                                            theta = theta,
                                            eta = eta,
                                            max_iter = 500,
                                            seed = seed)

                    #tSNE_iter_results[[paste('Perp: ', perplexity, ', Theta: ', theta, ', Eta: ', eta, ', Max_iter', max_iter, ', Seed', seed)]] <- tsne_results$Y

                    # PLot the results
                    #plot(tsne_results$Y, main = paste('Perp: ', perplexity, ', Theta: ', theta, ', Eta: ', eta, ', Max_iter', 500, ', Seed', seed))

                    ggplot(data.frame(tsne_results$Y), aes(x = X1, y = X2)) +
                        geom_point() +
                        ggtitle(paste("Perp:", perplexity, "| Theta:", theta, "| Eta:", eta))

                    # Save the plot
                    filename <- sprintf('tsne_plots/tsne_plot_perp_%d_theta_%1f_eta_%d_maxIter_500_seed_%d.png', perplexity, theta, eta, seed)
                    #savePlot(filename)

                    ggsave(filename)
                }
            #}
        }
    }
}



# Plot results
plot(tSNE_results$Y, asp = 1)
