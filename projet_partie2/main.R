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
library(MASS)

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

  # First correct borough names with encoding issues
  borough_corrections <- list(
    '?le-Bizard-Sainte-GeneviÞve' = 'Île-Bizard-Sainte-Geneviève',
    'C¶te-Saint-Luc' = 'Côte-Saint-Luc',
    'C¶te-des-Neiges-Notre-Dame-de-Graces' = 'Côte-des-Neiges-Notre-Dame-de-Grâce',
    'MontrÚal-Est' = 'Montréal-Est',
    'MontrÚal-Nord' = 'Montréal-Nord',
    'Pointe-aux-Trembles-RiviÞres-des-Prairies' = 'Rivière-des-Prairies-Pointe-aux-Trembles',
    'St-LÚonard' = 'Saint-Léonard'
  )
  
  # Then group boroughs into zones
  borough_zones <- list(
    # Zone ouest
    'Kirkland' = 'Zone ouest',
    'Beaconsfield' = 'Zone ouest',
    'Île-Bizard-Sainte-Geneviève' = 'Zone ouest',
    'Pierrefonds-Roxboro' = 'Zone ouest',
    'Dollard-des-Ormeaux' = 'Zone ouest',
    'Dorval' = 'Zone ouest',
    
    # Zone est
    'Rivière-des-Prairies-Pointe-aux-Trembles' = 'Zone est',
    'Montréal-Est' = 'Zone est',
    'Anjou' = 'Zone est',
    
    # Zone centre
    'Outremont' = 'Zone centre',
    'Mont-Royal' = 'Zone centre',
    
    # Zone sud
    'Sud-Ouest' = 'Zone sud',
    'Côte-Saint-Luc' = 'Zone sud',
    'Verdun' = 'Zone sud',
    'Lasalle' = 'Zone sud',
    'Lachine' = 'Zone sud',
    
    # Zone centre-sud
    'Côte-des-Neiges-Notre-Dame-de-Grâce' = 'Zone centre-sud',
    'Hampstead' = 'Zone centre-sud',
    'Westmount' = 'Zone centre-sud'
  )
  
  # Apply corrections if borough column exists
  if ("borough" %in% colnames(data)) {
    # First fix encoding issues
    for (old_name in names(borough_corrections)) {
      data$borough[data$borough == old_name] <- borough_corrections[[old_name]]
    }
    
    # Then group into zones
    for (borough in names(borough_zones)) {
      data$borough[data$borough == borough] <- borough_zones[[borough]]
    }
  }

  # Scale the features
  data[c('ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 
         'ln_fti', "distdt", "ln_distdt")] <- scale(data[c('ln_fi', 'ln_fri', 'ln_fli', 'ln_pi', 
                             'ln_fti', "distdt", "ln_distdt") ])
  data[c('ln_cti', 'ln_cli', 'ln_cri')] <- scale(data[c('ln_cti', 'ln_cli', 'ln_cri')])

  return(data)
}

#----------------------------------------------------------------
# 1. Load and prepare data
#----------------------------------------------------------------
df <- load_and_prepare_data()

# Convert to spatial object
coordinates(df) <- ~ x + y

# Specify the formula
formulaa <- acc ~ ln_fli + ln_pi + ln_distdt + ln_cli + ln_cri + total_lane + tot_road_w + tot_crossw + commercial + curb_exten + median + all_pedest + half_phase + ped_countd + lt_restric + lt_prot_re + any_exclus + all_red_an + green_stra + parking

# --------- Check visuellement s'il y a correlation dans residus de lm standard
olss <- lm(formula=formulaa, data=df) # régression linéaire de base
summary(olss)
df$resid <- residuals(olss)
spplot(df, "resid")

## -------- Choix de la matrice de poids -----
id <- row.names(as(df, "data.frame"))

neighbours <- knn2nb(knearneigh(coordinates(df), k = 3), row.names= id)
weights <- nb2listw(neighbours, style = "W", zero.policy=TRUE) # Pour poids binaires
# weights <- nb2listwdist(neighbours, df) # Pour poids proportionnels à la distance

plot(df)
plot(neighbours, coordinates(df), main="KNN = 4 neighborhood")
lm.morantest(olss, weights)

# Lagrange Multiplier Test. Tests for dependence in errors and/or lagged (nearby) variables
LM <- lm.RStests(olss, weights, test="all")


error_model <- errorsarlm(formulaa, data = df, listw = weights)
summary(error_model)


#----------------------------------------------------------------
# 2. Helper to compute spatial weights for a given training subset
#----------------------------------------------------------------
make_spatial_weights <- function(sp_data, k) {
  # sp_data must already be a Spatial object with coordinates
  id <- row.names(as(sp_data, "data.frame"))
  neighbours <- knn2nb(knearneigh(coordinates(sp_data), k = k), row.names = id)
  listw <- nb2listw(neighbours, style = "B")
  return(listw)
}

plot_spatial_folds <- function(sp_data, cluster_assignments, title = "Spatial Cross-Validation Folds") {
  # Create a copy of the spatial data to avoid modifying the original
  plot_data <- sp_data
  
  # Add cluster assignments as a factor (for better color mapping)
  plot_data$fold <- as.factor(cluster_assignments)
  
  # Convert to sf for better plotting with mapview
  plot_data_sf <- st_as_sf(plot_data)
  
  # Create the map using mapview
  m <- mapview(plot_data_sf, 
               zcol = "fold", 
               layer.name = "Fold Assignment",
               col.regions = rainbow(length(unique(cluster_assignments))),
               cex = 3,
               alpha = 1)
  
  return(m)
}


#----------------------------------------------------------------
# 3. Cross-validation function
#    - Allows OLS, SEM, and SLM
#----------------------------------------------------------------
# Function to create a spatial error model with Poisson distribution
train_poisson_sem <- function(formulaa, train_data, val_data, global_weights, fold_weights, use_poisson=TRUE) {
  
  # Step 1: Fit a standard Poisson model first
  if (use_poisson){ # Use poisson or negative binomial
    poisson_model <- glm(formulaa, data = train_data, family = poisson(link = "log"))
  } else {
    poisson_model <- glm.nb(formulaa, data = train_data)
  }

  
  # Step 2: Extract residuals (deviance residuals work well)
  residuals <- residuals(poisson_model, type = "deviance")
  
  # Step 3: Fit spatial error model to the residuals
  train_data$residuals <- residuals
  sem_residuals <- errorsarlm(as.formula("residuals ~ 1"), data = train_data, listw = fold_weights)
  
  # Step 4: For prediction, combine both models
  # First get non-spatial predictions
  val_data <- data.frame(val_data)
  poisson_predictions <- predict(poisson_model, newdata = val_data, type = "response")
  
  # Then get spatial predictions
  residual_predictions <- predict(sem_residuals, newdata = val_data, listw = global_weights, 
                                  zero.policy = TRUE, type = "response")

  # Apply adjustment (on log scale, then transform back)
  adjusted_predictions <- poisson_predictions * exp(residual_predictions)
  return(adjusted_predictions)
}


cross_validate_model <- function(sp_data, formulaa, k_folds=5, model_type="OLS", k_nb=3) {
  # We want to return fold-wise RMSE, MAE, and possibly R-squared
  set.seed(123)  # for reproducibility if needed
  # Create folds (stratified on 'acc' if you prefer, or plain random)
  spatial_clusters <- kmeans(coordinates(sp_data), centers = k_folds)
  folds <- lapply(1:k_folds, function(k) which(spatial_clusters$cluster == k))
  # folds <- createFolds(sp_data@data$acc, k = k_folds, list = TRUE, returnTrain = FALSE)
  
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
      m <- glm(formulaa, data = train_sp@data)
      preds <- predict(m, newdata = test_sp@data, type = "response")
      
    } else if (model_type == "SEM") {
      # Spatial Error Model
      m <- errorsarlm(formulaa, data = train_sp, listw = w_train)
      preds <- predict(m, newdata = test_sp, listw = w_global, zero.policy = TRUE, type = "response")
      
    } else if (model_type == "SLM") {
      # Spatial Lag Model 
      m <- lagsarlm(formulaa, data = train_sp, listw = w_train)
      preds <- predict(m, newdata = test_sp, listw = w_global, zero.policy = TRUE, type = "response")
      
    } else if (model_type == "SEM_poisson") {
      preds <- train_poisson_sem(formulaa, train_data = train_sp, val_data = test_sp, global_weights = w_global, fold_weights = w_train)
    } else if (model_type == "SEM_nb") {
      preds <- train_poisson_sem(formulaa, train_data = train_sp, val_data = test_sp, global_weights = w_global, fold_weights = w_train, use_poisson=FALSE)
    }
    
    else {
      stop("model_type must be one of 'OLS', 'SEM', 'SLM', 'SEM_poisson' or 'SEM_nb'")
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

# SEM Poisson
cv_poisson_sem <- cross_validate_model(sp_data = df, formula = formulaa, k_folds = 5, model_type = "SEM_poisson")


# SEM nb
cv_nb_sem <- cross_validate_model(sp_data = df, formula = formulaa, k_folds = 5, model_type = "SEM_nb")


#----------------------------------------------------------------
# 5. Compare the aggregated performance
#----------------------------------------------------------------
compare <- data.frame(
  Model = c("OLS", "SEM", "SLM", "SEM_poisson", "SEM_nb"),
  RMSE  = c(cv_ols$summary$Avg_RMSE,
            cv_sem$summary$Avg_RMSE,
            cv_slm$summary$Avg_RMSE,
            cv_poisson_sem$summary$Avg_RMSE,
            cv_nb_sem$summary$Avg_RMSE),
  MAE   = c(cv_ols$summary$Avg_MAE,
            cv_sem$summary$Avg_MAE,
            cv_slm$summary$Avg_MAE,
            cv_poisson_sem$summary$Avg_MAE,
            cv_nb_sem$summary$Avg_MAE),
  R2    = c(cv_ols$summary$Avg_R2,
            cv_sem$summary$Avg_R2,
            cv_slm$summary$Avg_R2,
            cv_poisson_sem$summary$Avg_R2,
            cv_nb_sem$summary$Avg_R2)
)

print(compare)
