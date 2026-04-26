
rm(list = ls()); gc()

library(readxl)
library(writexl)
library(dplyr)
library(caret)
library(lightgbm)
library(pROC)

library(doParallel)
library(foreach)
library(doRNG)
library(parallel)
library(tibble)


## SETTINGS

SEED_MAIN <- 123
N_PER_CLASS <- 28

# Global (pre-CV) Wilcoxon p-value threshold
P_THRESH <- 0.01

# Outer CV: 5-fold repeated 20 times 
OUTER_K <- 5
OUTER_REPEATS <- 20

# Inner holdout proportion for early stopping (within outer-train only)
INNER_PROP <- 0.80
nrounds    <- 2000
early_stop <- 50

base_path <- "C:/Users/Carlo/Documents/Università varia/Tirocinio_Tesi/motori/"

out_dir <- "C:/Users/Carlo/Documents/Università varia/Tirocinio_Tesi/Features/LightGBM"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)


## LOAD RAW DATA (SWAY + GAIT + TUG + FRAT)

sway <- read_excel(file.path(base_path, "SWAY_combined_results_T0.xlsx"))
gait <- read_excel(file.path(base_path, "GAIT_combined_results_T0.xlsx"))
tug  <- read_excel(file.path(base_path, "TUG_combined_results_T0.xlsx"))

frat <- read_excel(file.path(base_path, "FRAT_combined_results_T0.xlsx")) %>%
  mutate(
    IlSoggetto_CadutoNegliUltimi12Mesi_ = tolower(IlSoggetto_CadutoNegliUltimi12Mesi_),
    Faller = case_when(
      IlSoggetto_CadutoNegliUltimi12Mesi_ == "true"  ~ "Faller",
      IlSoggetto_CadutoNegliUltimi12Mesi_ == "false" ~ "NonFaller",
      TRUE ~ NA_character_
    )
  ) %>%
  dplyr::select(Subject, Faller)


## SUBJECT-AVERAGE EACH MODALITY + PREFIX
# avg per subject + prefix feature names to preserve modality identity
avg_subject_prefix <- function(df, prefix) {
  out <- df %>%
    group_by(Subject) %>%
    summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
  nm <- names(out)
  nm[nm != "Subject"] <- paste0(prefix, "__", nm[nm != "Subject"])
  names(out) <- nm
  out
}

sway_avg <- avg_subject_prefix(sway, "SWAY")
gait_avg <- avg_subject_prefix(gait, "GAIT")
tug_avg  <- avg_subject_prefix(tug,  "TUG")


## MERGE (SUBJECT-LEVEL) + CLEAN (UNBALANCED)

data_merged <- frat %>%
  left_join(sway_avg, by = "Subject") %>%
  left_join(gait_avg, by = "Subject") %>%
  left_join(tug_avg,  by = "Subject") %>%
  filter(!is.na(Faller))

# replace NaN -> NA in numeric columns
data_merged <- data_merged %>%
  mutate(across(where(is.numeric), ~ ifelse(is.nan(.x), NA, .x)))

# encode outcome as factor
data_merged$Faller <- factor(data_merged$Faller, levels = c("Faller", "NonFaller"))

cat("\nAvailable subjects after merge (UNBALANCED):\n")
print(table(data_merged$Faller))


## GLOBAL FEATURE SELECTION (WILCOXON) ON UNBALANCED, BEFORE BALANCING


# candidate numeric predictors (exclude Subject; exclude outcome)
vars_all <- names(data_merged)[sapply(data_merged, is.numeric)]
# (Subject should NOT be numeric; but just in case:)
vars_all <- setdiff(vars_all, c("Subject"))

# Wilcoxon comparing Faller vs NonFaller for each feature
# if feature is all-NA or constant, return NA
wilcox_p <- sapply(vars_all, function(v) {
  x <- data_merged[[v]]
  y <- data_merged$Faller
  # If feature is all NA or constant, wilcox can fail -> NA
  tryCatch({
    if (all(is.na(x))) return(NA_real_)
    if (length(unique(x[!is.na(x)])) < 2) return(NA_real_)
    wilcox.test(x ~ y)$p.value
  }, error = function(e) NA_real_)
})

wilcox_table <- tibble(
  Feature = vars_all,
  p_value = as.numeric(wilcox_p)
) %>%
  arrange(p_value)

# keep features with p < threshold
preselected_features <- wilcox_table %>%
  filter(!is.na(p_value) & p_value < P_THRESH) %>%
  pull(Feature)

cat("\n================ WILCOXON (UNBALANCED) PRESELECTION ================\n")
cat("Threshold p <", P_THRESH, "\n")
cat("Selected features:", length(preselected_features), "out of", length(vars_all), "\n")

# Fail-safe: keep at least 1 feature if none pass
if (length(preselected_features) == 0) {
  preselected_features <- wilcox_table$Feature[1]
  cat("No features passed threshold. Keeping best single feature:", preselected_features, "\n")
}


## BALANCE (N_PER_CLASS vs N_PER_CLASS) at SUBJECT LEVEL

set.seed(SEED_MAIN)
fallers    <- data_merged %>% filter(Faller == "Faller")    %>% slice_sample(n = N_PER_CLASS)
nonfallers <- data_merged %>% filter(Faller == "NonFaller") %>% slice_sample(n = N_PER_CLASS)

data_balanced <- bind_rows(fallers, nonfallers) %>% sample_frac(1.0)

cat("\nBalanced counts:\n")
print(table(data_balanced$Faller))


## BUILD X/y (REMOVE Subject; USE ONLY PRESELECTED FEATURES)

# Ensure selected features exist after balancing 
preselected_features <- preselected_features[preselected_features %in% names(data_balanced)]

# Build numeric matrix X and binary label y (Faller = 1, NonFaller = 0)
df2 <- data_balanced %>%
  dplyr::select(all_of(preselected_features)) %>%
  as.data.frame(stringsAsFactors = FALSE) %>%
  mutate(across(everything(), as.numeric))

colnames(df2) <- make.unique(colnames(df2))

X_all <- data.matrix(df2)
y <- ifelse(data_balanced$Faller == "Faller", 1L, 0L)

cat("\nUsing N features =", ncol(X_all), "\n")
cat("Top features used (wilcoxon-selected):\n")
print(colnames(X_all))


## LIGHTGBM PARAMS (num_threads=1 for parallel safety)

lgb_params <- list(
  objective         = "binary",
  metric            = "auc",
  learning_rate     = 0.01,
  num_leaves        = 4,
  max_depth         = -1,
  min_data_in_leaf  = 3,
  feature_fraction  = 0.6,
  bagging_fraction  = 0.7,
  bagging_freq      = 1,
  lambda_l1         = 1.0,
  lambda_l2         = 1.0,
  min_gain_to_split = 0,
  verbosity         = -1,
  num_threads       = 1
)

# safe AUC computation (returns NA if only one class in y_true)
safe_auc <- function(y_true, prob) {
  if (length(unique(y_true)) < 2) return(NA_real_)
  as.numeric(pROC::auc(pROC::roc(y_true, prob, quiet = TRUE)))
}


## PARALLEL BACKEND (parallelize OUTER folds)

set.seed(SEED_MAIN)
N_CORES <- max(1, parallel::detectCores() - 1)
cl <- makePSOCKcluster(N_CORES)
registerDoParallel(cl)


## OUTER 5×20 REPEATED CV INDICES

set.seed(SEED_MAIN)
outer_folds <- caret::createMultiFolds(y = factor(y, levels=c(0,1)), k = OUTER_K, times = OUTER_REPEATS)
fold_names <- names(outer_folds)


## RUN OUTER FOLDS IN PARALLEL

# For each outer split:
# - split train/test
# - create inner holdout inside outer-train for early stopping
# - train model with early stopping to get best_iter
# - refit on full outer-train for best_iter
# - evaluate on outer-test (AUC, confusion matrix metrics)
res_outer <- foreach(
  fold_name = fold_names,
  .packages = c("lightgbm","caret","pROC"),
  .combine = rbind
) %dorng% {
  
  outer_train_idx <- outer_folds[[fold_name]]
  outer_test_idx  <- setdiff(seq_len(nrow(X_all)), outer_train_idx)
  
  X_outer_train <- X_all[outer_train_idx, , drop=FALSE]
  y_outer_train <- y[outer_train_idx]
  
  X_outer_test  <- X_all[outer_test_idx, , drop=FALSE]
  y_outer_test  <- y[outer_test_idx]
  
  # ----------------------------
  # INNER holdout for early stop
  # ----------------------------
  fold_seed <- 1000 + as.integer(gsub("\\D", "", fold_name))
  set.seed(fold_seed)
  
  inner_train_sub <- caret::createDataPartition(y_outer_train, p = INNER_PROP, list = FALSE)
  inner_train_idx <- inner_train_sub
  inner_valid_idx <- setdiff(seq_len(nrow(X_outer_train)), inner_train_idx)
  
  X_inner_train <- X_outer_train[inner_train_idx, , drop=FALSE]
  y_inner_train <- y_outer_train[inner_train_idx]
  
  X_inner_valid <- X_outer_train[inner_valid_idx, , drop=FALSE]
  y_inner_valid <- y_outer_train[inner_valid_idx]
  
  dtrain_inner <- lgb.Dataset(data = X_inner_train, label = y_inner_train)
  dvalid_inner <- lgb.Dataset(data = X_inner_valid, label = y_inner_valid)
  
  model_inner <- lgb.train(
    params  = lgb_params,
    data    = dtrain_inner,
    nrounds = nrounds,
    valids  = list(valid = dvalid_inner),
    early_stopping_rounds = early_stop,
    verbose = -1
  )
  
  best_iter <- model_inner$best_iter
  if (is.null(best_iter) || is.na(best_iter) || best_iter <= 0) best_iter <- nrounds
  
  # ----------------------------
  # FINAL fit on full outer-train
  # ----------------------------
  dtrain_outer_final <- lgb.Dataset(data = X_outer_train, label = y_outer_train)
  
  model_outer_final <- lgb.train(
    params  = lgb_params,
    data    = dtrain_outer_final,
    nrounds = best_iter,
    verbose = -1
  )
  
  # ----------------------------
  # EVAL on outer test
  # ----------------------------
  prob <- predict(model_outer_final, X_outer_test)
  pred_class <- ifelse(prob > 0.5, 1L, 0L)
  
  cm <- caret::confusionMatrix(
    factor(pred_class, levels = c(0,1)),
    factor(y_outer_test, levels = c(0,1)),
    positive = "1"
  )
  
  fold_auc <- safe_auc(y_outer_test, prob)
  
  data.frame(
    Fold        = fold_name,
    BestIter    = best_iter,
    NFeatures   = ncol(X_all),
    Accuracy    = as.numeric(cm$overall["Accuracy"]),
    Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Specificity = as.numeric(cm$byClass["Specificity"]),
    AUC         = fold_auc,
    stringsAsFactors = FALSE
  )
}


## STOP PARALLEL

stopCluster(cl)
registerDoSEQ()

cv_results_nested <- as.data.frame(res_outer)


## SUMMARY

cv_summary_nested <- cv_results_nested %>%
  summarise(
    Accuracy_mean    = mean(Accuracy, na.rm = TRUE),
    Accuracy_sd      = sd(Accuracy, na.rm = TRUE),
    Sensitivity_mean = mean(Sensitivity, na.rm = TRUE),
    Sensitivity_sd   = sd(Sensitivity, na.rm = TRUE),
    Specificity_mean = mean(Specificity, na.rm = TRUE),
    Specificity_sd   = sd(Specificity, na.rm = TRUE),
    AUC_mean         = mean(AUC, na.rm = TRUE),
    AUC_sd           = sd(AUC, na.rm = TRUE),
    BestIter_mean    = mean(BestIter, na.rm = TRUE),
    BestIter_sd      = sd(BestIter, na.rm = TRUE)
  )

cat("\n================ NESTED CV SUMMARY ================\n")
cat(sprintf("Outer CV: %d-fold × %d repeats (%d resamples)\n", OUTER_K, OUTER_REPEATS, OUTER_K*OUTER_REPEATS))
cat(sprintf("Inner split: %.0f%% train / %.0f%% valid\n", INNER_PROP*100, (1-INNER_PROP)*100))
cat(sprintf("Features (fixed pre-CV via Wilcoxon p<%.2f): %d\n\n", P_THRESH, ncol(X_all)))
print(cv_summary_nested)


## OPTIONAL: inspect AUC distribution

hist(cv_results_nested$AUC, main="AUC distribution (Nested CV) — Wilcoxon features", xlab="AUC")

## SAVE TO EXCEL

out_file <- file.path(out_dir, paste0("LightGBM_ALL_Wilcoxon_p", P_THRESH, "_NestedCV.xlsx"))

write_xlsx(
  list(
    "wilcoxon_table_unbalanced" = wilcox_table,
    "features_used"            = data.frame(Feature = colnames(X_all)),
    "cv_per_fold"              = cv_results_nested,
    "cv_summary"               = cv_summary_nested
  ),
  path = out_file
)

cat("\nSaved:", out_file, "\n")
