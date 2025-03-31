library(dplyr)
library(readr)
library(caret)
library(Metrics)
library(tidyr)
library(spdep)
library(spatialreg)
library(spgwr)
library(sp)
library(sf)
library(mapview)

base_path <- "/Users/Xavier/Maitrise - Cours/Hiver 2025/Méthodes avancées en exploitation de données/code/projet_methodes"

load_and_prepare_data <- function() {
  data <- read_delim(file.path(base_path, "data_final.csv"), 
                     delim = ";", show_col_types = FALSE)
  
  # Handle missing values in ln_distdt (if present)
  data <- data %>% replace_na(list(ln_distdt=0))
  
  # Drop rows with duplicate coordinates
  message(paste("Number of rows before dropping duplicates:", nrow(data)))
  data <- data %>% distinct(x, y, .keep_all = TRUE)
  message(paste("Number of rows after dropping duplicates:", nrow(data)))
  
  # Remove pi = 0
  data <- data %>% filter(pi != 0)
  
  # Relevant columns
  coordinates_features <- c('x', 'y')
  traffic_flow_features <- c('ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 
                             'ln_fti', "distdt", "ln_distdt") 
  interaction_features  <- c('ln_cti', 'ln_cli', 'ln_cri')
  geometric_features    <- c('total_lane', 'avg_crossw', 'tot_road_w', 'tot_crossw',
                             'commercial', 'number_of_', 'of_exclusi', 'curb_exten')
  safety_features       <- c('median', 'all_pedest', 'half_phase', 'new_half_r',
                             'any_ped_pr', 'ped_countd', 'lt_restric', 'lt_prot_re',
                             'lt_protect', 'any_exclus', 'green_stra', 'parking')
  directional_features  <- c('north_veh', 'north_ped', 'east_veh', 'east_ped')
  response_variable     <- c('acc')
  
  feature_cols <- c(traffic_flow_features, interaction_features, 
                    geometric_features, safety_features, 
                    coordinates_features, response_variable)
  
  data <- data[, feature_cols]
  return(data)
}

#----------------------------------------------------------------
# 1. Load and prepare data
#----------------------------------------------------------------
df <- load_and_prepare_data()

# Convert to spatial object
coordinates(df) <- ~ x + y

# Specify the formula
formulaa <- acc ~ ln_fli + ln_pi + ln_fti + ln_distdt +
  ln_cli + ln_cti + total_lane + tot_road_w +
  commercial + as.factor(curb_exten) + as.factor(median) + as.factor(all_pedest) +
  as.factor(half_phase) + as.factor(ped_countd) + as.factor(lt_restric) + as.factor(lt_protect) +
  as.factor(any_exclus) + as.factor(green_stra) + as.factor(parking)

#----------------------------------------------------------------
# 2. Helper to compute spatial weights for a given training subset
#----------------------------------------------------------------
make_spatial_weights <- function(sp_data, k) {
  # sp_data must already be a Spatial object with coordinates
  id <- row.names(as(sp_data, "data.frame"))
  neighbours <- knn2nb(knearneigh(coordinates(sp_data), k = k), row.names = id)
  listw <- nb2listw(neighbours, style = "W")
  return(listw)
}

#----------------------------------------------------------------
# 3. Cross-validation function
#    - Allows OLS, SEM, and SLM
#----------------------------------------------------------------
cross_validate_model <- function(sp_data, formulaa, k_folds=5, model_type="OLS", k_nb=4) {
  # We want to return fold-wise RMSE, MAE, and possibly R-squared
  set.seed(123)  # for reproducibility if needed
  # Create folds (stratified on 'acc' if you prefer, or plain random)
  folds <- createFolds(sp_data@data$acc, k = k_folds, list = TRUE, returnTrain = FALSE)
  
  results <- data.frame(
    Fold = integer(),
    RMSE = numeric(),
    MAE  = numeric(),
    R2   = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Global weight matrix used for prediction
  w_global <- make_spatial_weights(sp_data, k = k_nb)
  
  for (i in seq_along(folds)) {
    test_idx  <- folds[[i]]
    train_idx <- setdiff(seq_len(nrow(sp_data@data)), test_idx)
    
    # Subset into training and test sets
    train_sp <- sp_data[train_idx, ]
    test_sp  <- sp_data[test_idx, ]
    
    # Recompute spatial weights for the training subset only
    w_train <- make_spatial_weights(train_sp, k = k_nb)
    
    # Fit the model
    if (model_type == "OLS") {
      m <- lm(formulaa, data = train_sp@data)
      preds <- predict(m, newdata = test_sp@data)
      
    } else if (model_type == "SEM") {
      # Spatial Error Model
      m <- errorsarlm(formulaa, data = train_sp, listw = w_train)
      preds <- predict(m, newdata = test_sp, listw = w_global)
      
    } else if (model_type == "SLM") {
      # Spatial Lag Model
      m <- lagsarlm(formulaa, data = train_sp, listw = w_train)
      preds <- predict(m, newdata = test_sp, listw = w_global)
      
    } else {
      stop("model_type must be one of 'OLS', 'SEM', or 'SLM'.")
    }
    
    # Actual values
    actuals <- test_sp@data$acc
    
    # Compute metrics
    rmse_val <- rmse(actuals, preds)
    mae_val  <- mae(actuals, preds)
    
    # For R^2, we can do a simple correlation-based approach
    # or do 1 - SSE/TSS
    sse <- sum((actuals - preds)^2)
    sst <- sum((actuals - mean(actuals))^2)
    r2  <- 1 - sse/sst
    
    results <- rbind(results, 
                     data.frame(Fold = i, RMSE = rmse_val, MAE = mae_val, R2 = r2))
  }
  
  # Return the average across folds, plus fold-by-fold
  summary_results <- results %>%
    summarise(
      Avg_RMSE = mean(RMSE),
      Avg_MAE  = mean(MAE),
      Avg_R2   = mean(R2)
    )
  
  list(
    fold_metrics = results,
    summary      = summary_results
  )
}

#----------------------------------------------------------------
# 4. Run cross-validation for each model type
#----------------------------------------------------------------
# OLS
cv_ols <- cross_validate_model(sp_data = df, formula = formulaa, 
                               k_folds = 5, model_type = "OLS")
cv_ols$summary
cv_ols$fold_metrics

# SEM
cv_sem <- cross_validate_model(sp_data = df, formula = formulaa,
                               k_folds = 5, model_type = "SEM")
cv_sem$summary
cv_sem$fold_metrics

# SLM
cv_slm <- cross_validate_model(sp_data = df, formula = formulaa,
                               k_folds = 5, model_type = "SLM")
cv_slm$summary
cv_slm$fold_metrics

#----------------------------------------------------------------
# 5. Compare the aggregated performance
#----------------------------------------------------------------
compare <- data.frame(
  Model = c("OLS", "SEM", "SLM"),
  RMSE  = c(cv_ols$summary$Avg_RMSE,
            cv_sem$summary$Avg_RMSE,
            cv_slm$summary$Avg_RMSE),
  MAE   = c(cv_ols$summary$Avg_MAE,
            cv_sem$summary$Avg_MAE,
            cv_slm$summary$Avg_MAE),
  R2    = c(cv_ols$summary$Avg_R2,
            cv_sem$summary$Avg_R2,
            cv_slm$summary$Avg_R2)
)

compare


# --------- Check visuellement s'il y a correlation dans residus de lm standard
ols <- lm(formulaa, data=data) # régression linéaire de base
summary(ols)
data$resid <- residuals(ols)
spplot(data, "resid")

## -------- Choix de la matrice de poids -----
id <- row.names(as(data, "data.frame"))

neighbours <- knn2nb(knearneigh(coordinates(data), k = 3), row.names= id)
weights <- nb2listw(neighbours, style = "W", zero.policy=FALSE) # Pour poids binaires
weights <- nb2listwdist(neighbours, data) # Pour poids proportionnels à la distance

plot(data)
plot(neighbours, coordinates(data), main="KNN = 4 neighborhood")
lm.morantest(ols, weights)


sem <- errorsarlm(formulaa, data=data, listw=weights)
summary(sem)



slm <- lagsarlm(formula, data=data, weights)
summary(slm)

# Lagrange Multiplier Test. Tests for dependence in errors and/or lagged (nearby) variables
LM <- lm.RStests(ols, weights, test="all")

