library(dplyr)
library(readr)
library(caret)
library(Metrics)
library(tidyr)

# Add spatial analysis packages
library(spdep)    # For spatial dependency and autocorrelation analysis
library(spatialreg) # For spatial regression models
library(spgwr)    # For geographically weighted regression
library(sp)       # Classes and methods for spatial data
library(sf)       # Simple Features for R
library(mapview)  # Interactive viewing of spatial data in R

setwd("/Users/Xavier/Maitrise - Cours/Hiver 2025/Méthodes avancées en exploitation de données/code")
base_path <- "./Road_Safety/prog/data"


load_and_prepare_data <- function() {
  data <- read_delim(file.path(base_path, "data_final.csv"), delim = ";", show_col_types = FALSE)
  
  # Handle missing values in ln_distdt (if present)
  data <- data %>% replace_na(list(ln_distdt=0))

  #remove pi = 0
  data <- data %>% filter(pi != 0)
  
  # Handle any other missing values in features
  coordinates_features <- c('x', 'y')
  traffic_flow_features <- c('ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 'fi', 'fri', 'fli', 'pi', 'traffic_10000', 'fti', 'ln_fti', "distdt", "ln_distdt")
  interaction_features <- c('ln_cti', 'ln_cli', 'ln_cri', 'cti', 'cli', 'cri')
  geometric_features <- c('total_lane', 'avg_crossw', 'tot_road_w', 'tot_crossw',
                         'commercial', 'number_of_', 'of_exclusi', 'curb_exten')
  safety_features <- c('median', 'all_pedest', 'half_phase', 'new_half_r',
                      'any_ped_pr', 'ped_countd', 'lt_restric', 'lt_prot_re',
                      'lt_protect', 'any_exclus', 'all_red_an', 'green_stra',
                      'parking')
  directional_features <- c('north_veh', 'north_ped', 'east_veh', 'east_ped')
  response_variable <- c('acc')
  
  feature_cols <- c(traffic_flow_features, interaction_features, 
                   geometric_features, safety_features, coordinates_features, response_variable)
  
  X_features <- data[, feature_cols]
  
  # Replace any remaining NA values with 0
  X_features <- replace(X_features, is.na(X_features), 0)
  
  return(list(
    X = X_features,
    y = data$acc,
    int_no = data$int_no,
    pi = data$pi
  ))
}


data <- load_and_prepare_data()

geo_data <- st_as_sf(data$X, coords = c("x", "y"))
# Filter out duplicate coordinates
geo_data <- geo_data[!duplicated(st_coordinates(geo_data)), ]



# Basic visualization of accidents
map_viz <- mapview(geo_data, zcol = "acc", layer.name = "Accidents")
map_viz

# 1. Create spatial weights matrix
# Convert sf object to sp for compatibility with spdep
sp_data <- as(geo_data, "Spatial")


# Create a neighbors list using k-nearest neighbors (k=5)
cat("Creating spatial weights matrix using k-nearest neighbors...\n")
knn5 <- knearneigh(coordinates(sp_data), k=5)
nb_knn5 <- knn2nb(knn5)

# Create a spatial weights matrix
w_knn5 <- nb2listw(nb_knn5, style="W")

# 2. Test for spatial autocorrelation using Moran's I
cat("Testing for spatial autocorrelation using Moran's I...\n")
moran_result <- moran.test(geo_data$acc, w_knn5)
print(moran_result)

# Local indicators of spatial association (LISA)
local_moran <- localmoran(geo_data$acc, w_knn5)
geo_data$local_moran_i <- local_moran[, 1]
geo_data$local_moran_p <- local_moran[, 5]

# Visualize local Moran's I
mapview(geo_data, zcol = "local_moran_i", layer.name = "Local Moran's I")

# 3. Create spatial lag and spatial error predictors
# These can be used as features in your models
geo_data$acc_spatial_lag <- lag.listw(w_knn5, geo_data$acc)

# 4. Implement spatial regression models
# Prepare formula with all relevant variables (excluding x and y which are now coordinates)
cat("Fitting spatial regression models...\n")

# List of features, excluding coordinates and the response variable
features <- names(geo_data)[!names(geo_data) %in% c("acc", "geometry", "local_moran_i", "local_moran_p", "acc_spatial_lag")]

# Create formula for models
formula_str <- paste("acc ~", paste(features, collapse = " + "))
model_formula <- as.formula(formula_str)

# Fit OLS model (for comparison)
ols_model <- lm(model_formula, data = geo_data)
summary(ols_model)

# Fit spatial lag model (SLM)
slm_model <- lagsarlm(model_formula, data = geo_data, listw = w_knn5)
summary(slm_model)

# Fit spatial error model (SEM)
sem_model <- errorsarlm(model_formula, data = geo_data, listw = w_knn5)
summary(sem_model)

# 5. Geographically Weighted Regression (GWR)
# This allows coefficients to vary across space
cat("Fitting Geographically Weighted Regression model...\n")
bw <- gwr.sel(model_formula, data = sp_data, coords = coordinates(sp_data))
gwr_model <- gwr(model_formula, data = sp_data, coords = coordinates(sp_data), bandwidth = bw, hatmatrix = TRUE)
print(gwr_model)

# 6. Compare models using AIC
cat("\nModel comparison using AIC:\n")
cat("OLS AIC:", AIC(ols_model), "\n")
cat("SLM AIC:", AIC(slm_model), "\n")
cat("SEM AIC:", AIC(sem_model), "\n")

# 7. Create custom function for prediction with spatial models
predict_with_spatial <- function(new_data, model_type = "ols", model = NULL, w = NULL) {
  if (model_type == "ols") {
    return(predict(model, new_data))
  } else if (model_type == "slm") {
    # For spatial lag models, we need to account for the spatial lag term
    return(predict(model, new_data, listw = w, pred.type = "trend"))
  } else if (model_type == "sem") {
    # For spatial error models
    return(predict(model, new_data, listw = w, pred.type = "trend"))
  } else if (model_type == "gwr") {
    # For GWR models
    coords <- st_coordinates(st_centroid(new_data$geometry))
    return(gwr.predict(model, new_data, coords))
  }
}

# 8. Cross-validation using spatial models
perform_spatial_cv <- function(data, k_folds = 5) {
  # Create folds preserving spatial structure
  coords <- st_coordinates(st_centroid(data$geometry))
  knn_cv <- knearneigh(coords, k = floor(nrow(data)/k_folds))
  nb_cv <- knn2nb(knn_cv)
  
  # Assign folds
  set.seed(123)
  folds <- rep(1:k_folds, length.out = nrow(data))
  folds <- sample(folds)
  
  results <- data.frame(
    fold = integer(),
    model = character(),
    rmse = numeric(),
    mae = numeric()
  )
  
  for (fold in 1:k_folds) {
    train_idx <- which(folds != fold)
    test_idx <- which(folds == fold)
    
    train_data <- data[train_idx, ]
    test_data <- data[test_idx, ]
    
    # Create weights for training data
    train_coords <- st_coordinates(st_centroid(train_data$geometry))
    train_knn <- knearneigh(train_coords, k = 5)
    train_nb <- knn2nb(train_knn)
    train_w <- nb2listw(train_nb, style = "W")
    
    # Formula
    features <- names(train_data)[!names(train_data) %in% c("acc", "geometry", "local_moran_i", "local_moran_p", "acc_spatial_lag")]
    formula_str <- paste("acc ~", paste(features, collapse = " + "))
    model_formula <- as.formula(formula_str)
    
    # Fit models
    ols <- lm(model_formula, data = train_data)
    slm <- tryCatch({
      lagsarlm(model_formula, data = train_data, listw = train_w)
    }, error = function(e) {
      cat("Error in SLM for fold", fold, ":", e$message, "\n")
      return(NULL)
    })
    sem <- tryCatch({
      errorsarlm(model_formula, data = train_data, listw = train_w)
    }, error = function(e) {
      cat("Error in SEM for fold", fold, ":", e$message, "\n")
      return(NULL)
    })
    
    # Predict and evaluate
    ols_pred <- predict(ols, test_data)
    results <- rbind(results, data.frame(
      fold = fold,
      model = "OLS",
      rmse = sqrt(mean((test_data$acc - ols_pred)^2)),
      mae = mean(abs(test_data$acc - ols_pred))
    ))
    
    if (!is.null(slm)) {
      slm_pred <- predict(slm, test_data, listw = train_w, pred.type = "trend")
      results <- rbind(results, data.frame(
        fold = fold,
        model = "SLM",
        rmse = sqrt(mean((test_data$acc - slm_pred)^2)),
        mae = mean(abs(test_data$acc - slm_pred))
      ))
    }
    
    if (!is.null(sem)) {
      sem_pred <- predict(sem, test_data, listw = train_w, pred.type = "trend")
      results <- rbind(results, data.frame(
        fold = fold,
        model = "SEM",
        rmse = sqrt(mean((test_data$acc - sem_pred)^2)),
        mae = mean(abs(test_data$acc - sem_pred))
      ))
    }
  }
  
  # Summarize results
  summary_results <- results %>%
    group_by(model) %>%
    summarize(
      avg_rmse = mean(rmse),
      avg_mae = mean(mae),
      sd_rmse = sd(rmse),
      sd_mae = sd(mae)
    )
  
  return(list(
    fold_results = results,
    summary = summary_results
  ))
}

# Run spatial cross-validation
cv_results <- perform_spatial_cv(geo_data)
print(cv_results$summary)
