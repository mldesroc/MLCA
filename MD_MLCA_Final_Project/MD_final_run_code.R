library(here)
library(tidyverse)
org_forest_comp <- read_csv(here("Data_Dryad_ARUN.csv"), col_names = TRUE)
forest_comp <- org_forest_comp[,c(18,30:42)]
forest_comp <- forest_comp[!duplicated(forest_comp),]

set.seed(123)
row_idx <- sample(seq_len(nrow(forest_comp)), nrow(forest_comp))
training <- forest_comp[row_idx < nrow(forest_comp) * 0.8,]
testing <- forest_comp[row_idx >= nrow(forest_comp) * 0.8,]
tuningsample <- forest_comp[row_idx < nrow(forest_comp) * 0.01,]

#dt
library(rpart)
library(rpart.plot)

decision_tree <- rpart(rela_mapBB_BA ~., data = training)

rpart(rela_mapBB_BA ~., data = training)|>
  rpart.plot(type=4)

dt_train_RMSE <- sqrt(mean((predict(decision_tree, training)-training$rela_mapBB_BA)^2))
dt_testRMSE <- sqrt(mean((predict(decision_tree, testing)-testing$rela_mapBB_BA)^2)) 

dt_testMAE <- sum(abs(testing$rela_mapBB_BA - predict(decision_tree, testing)))/nrow(testing)

#rf
library(ranger)

rf_calc_rmse <- function(rf_model, data) {
  rf_predictions <- predictions(predict(rf_model, data))
  sqrt(mean((rf_predictions - data$rela_mapBB_BA)^2))
}

rf_k_fold_cv <- function(data, k, ...) {
  per_fold <- floor(nrow(data) / k)
  fold_order <- sample(seq_len(nrow(data)), 
                       size = per_fold * k)
  fold_rows <- split(
    fold_order,
    rep(1:k, each = per_fold)
  )
  vapply(
    fold_rows,
    \(fold_idx) {
      fold_test <- data[fold_idx, ]
      fold_train <- data[-fold_idx, ]
      fold_rf <- ranger(rela_mapBB_BA ~ ., fold_train)
      rf_calc_rmse(fold_rf, fold_test)
    },
    numeric(1)
  ) |> 
    mean()
}

final_rf <- ranger(
  rela_mapBB_BA ~ .,
  testing, 
  num.trees = 800,
  mtry = 13,
  min.node.size = 8,
  replace = FALSE, 
  sample.fraction = 0.8
)
rfRMSE <- rf_calc_rmse(final_rf, testing)

rf_final_predictions <- predictions(predict(final_rf, testing))
rfMAE <- sum(abs(testing$rela_mapBB_BA - rf_final_predictions))/nrow(testing)

#gbm
library(lightgbm)

lgb_forest_comp <- forest_comp %>% 
  mutate(dummy_value = 1) %>%
  pivot_wider(
    names_from = eco_region,
    names_prefix = "eco_region",
    values_from = dummy_value, 
    values_fill = 0 
  )

set.seed(123)
row_idx <- sample(seq_len(nrow(lgb_forest_comp)), nrow(lgb_forest_comp))
lgb_training <- lgb_forest_comp[row_idx < nrow(lgb_forest_comp) * 0.8,]
lgb_testing <- lgb_forest_comp[row_idx >= nrow(lgb_forest_comp) * 0.8,]

xtrain <- as.matrix(lgb_training[setdiff(names(lgb_training), "rela_mapBB_BA")])
ytrain <- lgb_training[["rela_mapBB_BA"]]
xtest <- as.matrix(lgb_testing[setdiff(names(lgb_testing), "rela_mapBB_BA")])

lgb_calc_rmse <- function(model, data) {
  xtest <- as.matrix(data[setdiff(names(data), "rela_mapBB_BA")])
  lgb_predictions <- predict(model, xtest)
  sqrt(mean((lgb_predictions - data$rela_mapBB_BA)^2))
}

lgb_k_fold_cv <- function(data, k, nrounds = 10L, ...) {
  per_fold <- floor(nrow(data) / k)
  fold_order <- sample(seq_len(nrow(data)), 
                       size = per_fold * k)
  fold_rows <- split(
    fold_order,
    rep(1:k, each = per_fold)
  )
  vapply(
    fold_rows,
    \(fold_idx) {
      fold_test <- data[fold_idx, ]
      fold_train <- data[-fold_idx, ]
      xtrain <- as.matrix(fold_train[setdiff(names(fold_train), 
                                             "rela_mapBB_BA")])
      ytrain <- fold_train[["rela_mapBB_BA"]]
      fold_lgb <- lightgbm(
        data = xtrain,
        label = ytrain,
        verbose = -1L,
        obj = "regression",
        nrounds = nrounds,
        params = ...
      )
      lgb_calc_rmse(fold_lgb, fold_test)
    },
    numeric(1)
  ) |> 
    mean()
}

final_lgb <- lightgbm(
  data = xtrain,
  label = ytrain,
  verbose = -1L,
  obj = "regression",
  nrounds = 2000,
  params = list(
    learning_rate = 0.1,
    max_depth = 63,
    min_data_in_bin = 8,
    bagging_freq = 0,
    bagging_fraction = 0.5,
    feature_fraction = 0.7
  )
)


lgbRMSE <- lgb_calc_rmse(final_lgb, lgb_testing)
lgb_predictions <- predict(final_lgb, xtest)
lgbMAE <- sum(abs(lgb_testing$rela_mapBB_BA - lgb_predictions))/nrow(lgb_testing)