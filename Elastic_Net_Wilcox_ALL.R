
rm(list = ls()); gc()


### 0) PACKAGES

library(readxl)
library(dplyr)
library(tibble)
library(glmnet)
library(pROC)
library(ggplot2)
library(purrr)

library(doParallel)
library(foreach)
library(parallel)

library(writexl)

`%||%` <- function(x, y) if (is.null(x)) y else x


## SETTINGS

SEED_MAIN <- 42

# Wilcoxon pre-CV threshold
p_thresh_global <- 0.15 

# Class balance for CV
N_PER_CLASS <- 28

# CV
num_repeats <- 20
num_folds   <- 5         # for evaluation
inner_k     <- 5         # for choosing lambda

# Elastic Net alphas
# alphas <- c(0, 0.25, 0.5, 0.75, 1)
alphas <- c(0,1)

# riduzione griglia per velocizzare
N_LAMBDA_KEEP <- 40

# Paths
base_path <- "C:/Users/Carlo/Documents/Università varia/Tirocinio_Tesi/motori/"
out_path  <- "C:/Users/Carlo/Documents/Università varia/Tirocinio_Tesi/Features/ElasticNet_ALL_Wilcoxon_UNBAL_preCV_thenBalancedCV.xlsx"


# PARALLEL BACKEND (REPRODUCIBLE)

RNGkind("L'Ecuyer-CMRG")
set.seed(SEED_MAIN)

N_CORES <- min(8, max(1, parallel::detectCores() - 1))
cl <- parallel::makeCluster(N_CORES)
doParallel::registerDoParallel(cl)
cat("Parallel workers registered:", N_CORES, "\n")


## LOAD DATA (SWAY + GAIT + TUG + FRAT) 
# from FRAT creates true and false label (for identifying who is faller and who is not)


sway <- readxl::read_excel(file.path(base_path, "SWAY_combined_results_T0.xlsx"))
gait <- readxl::read_excel(file.path(base_path, "GAIT_combined_results_T0.xlsx"))
tug  <- readxl::read_excel(file.path(base_path, "TUG_combined_results_T0.xlsx"))

frat <- readxl::read_excel(file.path(base_path, "FRAT_combined_results_T0.xlsx")) %>%
  mutate(
    IlSoggetto_CadutoNegliUltimi12Mesi_ = tolower(IlSoggetto_CadutoNegliUltimi12Mesi_),
    Faller_log = case_when(
      IlSoggetto_CadutoNegliUltimi12Mesi_ == "true"  ~ TRUE,
      IlSoggetto_CadutoNegliUltimi12Mesi_ == "false" ~ FALSE,
      TRUE ~ NA
    )
  ) %>%
  group_by(Subject) %>%
  summarise(Faller_log = any(Faller_log, na.rm = TRUE), .groups = "drop") %>%
  filter(!is.na(Faller_log)) %>%
  mutate(Faller = ifelse(Faller_log, 1, 0)) %>%
  select(Subject, Faller)


## SUBJECT-AVERAGE EACH MODALITY + PREFIX
# groups all the subjects and average all numeric columns 
# adds a prefix to the features to distinguish them

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


# MERGE (UNBALANCED SUBJECT-LEVEL DATASET) by subject
# converts NaN in NA

data_all <- frat %>%
  left_join(sway_avg, by = "Subject") %>%
  left_join(gait_avg, by = "Subject") %>%
  left_join(tug_avg,  by = "Subject") %>%
  filter(!is.na(Faller))

# Replace NaN -> NA
data_all <- data_all %>%
  mutate(across(where(is.numeric), ~ ifelse(is.nan(.x), NA, .x)))

cat("\nUNBALANCED counts (all subjects available):\n")
print(table(data_all$Faller))
cat("N subjects (rows):", nrow(data_all), "\n")


## Feature selection with Wilcoxon
# For every numeric feature computes the Wilcoxon rank-sum test between Non Faller and Faller
# If no one is below the threshold, takes the minimum

vars_all <- setdiff(colnames(data_all %>% select(where(is.numeric))), "Faller")

X_all_df <- data_all %>% select(all_of(vars_all))
y_all    <- factor(data_all$Faller, levels = c(0, 1))

wilcox_pvals <- sapply(vars_all, function(v) {
  tryCatch(wilcox.test(X_all_df[[v]] ~ y_all)$p.value, error = function(e) NA_real_)
})

wilcox_table <- tibble(
  Feature = vars_all,
  p_value = as.numeric(wilcox_pvals)
) %>% arrange(p_value)

preselected_features <- wilcox_table %>%
  filter(!is.na(p_value) & p_value < p_thresh_global) %>%
  pull(Feature)

cat("\n================ WILCOXON (UNBALANCED) PRE-CV SELECTION ================\n")
cat("Threshold p <", p_thresh_global, "\n")
cat("Selected features:", length(preselected_features), "out of", length(vars_all), "\n")

# Fail-safe: keep at least one feature
if (length(preselected_features) == 0) {
  preselected_features <- wilcox_table$Feature[1]
  cat("No features passed threshold. Keeping best single feature:", preselected_features, "\n")
}


## BALANCE SUBJECTS FOR CV: N_PER_CLASS + N_PER_CLASS
# Takes the subjects
# filter data_all to keep just data_balanced
# Cross-reference the features selected by Wilcoxon with those actually present after balancing

set.seed(SEED_MAIN)

subjects_all <- data_all %>% distinct(Subject, Faller)
fallers_pool    <- subjects_all %>% filter(Faller == 1)
nonfallers_pool <- subjects_all %>% filter(Faller == 0)

if (nrow(fallers_pool) < N_PER_CLASS) stop(paste0("Not enough fallers to sample ", N_PER_CLASS, "."))
if (nrow(nonfallers_pool) < N_PER_CLASS) stop(paste0("Not enough non-fallers to sample ", N_PER_CLASS, "."))

subjects_balanced <- bind_rows(
  fallers_pool    %>% slice_sample(n = N_PER_CLASS),
  nonfallers_pool %>% slice_sample(n = N_PER_CLASS)
)

data_balanced <- data_all %>% semi_join(subjects_balanced, by = "Subject")
subject_table <- data_balanced %>% distinct(Subject, Faller)

cat("\nBALANCED counts (used for CV):\n")
print(table(subject_table$Faller))
cat("Balanced N subjects:", nrow(subject_table), "\n")

# Keep only features that exist in balanced dataset
vars_bal <- setdiff(colnames(data_balanced %>% select(where(is.numeric))), "Faller")
preselected_features <- intersect(preselected_features, vars_bal)

if (length(preselected_features) == 0) stop("After balancing, none of the Wilcoxon-selected features are available.")
cat("\nUsing", length(preselected_features), "features in CV (selected on UNBALANCED data).\n")


## HELPERS (folding + preprocessing + metrics)

# avoid errors if one cell of the confusion matrix missing
safe_cm_get <- function(cm, true, pred) {
  if (true %in% rownames(cm) && pred %in% colnames(cm)) cm[true, pred] else 0
}

# assign fold 1..k separately for each class
make_subject_folds <- function(df_subjects, k, seed = 1) {
  set.seed(seed)
  folds <- rep(NA_integer_, nrow(df_subjects))
  idx0 <- which(df_subjects$Faller == 0)
  idx1 <- which(df_subjects$Faller == 1)
  folds[idx0] <- sample(rep(1:k, length.out = length(idx0)))
  folds[idx1] <- sample(rep(1:k, length.out = length(idx1)))
  folds
}

# useful for inner CV in glmnet, guarantees grouping by subject
make_foldid_by_subject <- function(subject_vec, y_vec, k = 5, seed = 1) {
  set.seed(seed)
  subj_df <- data.frame(Subject = subject_vec, y = y_vec) %>% distinct(Subject, y)
  
  fold_subj <- rep(NA_integer_, nrow(subj_df))
  idx0 <- which(subj_df$y == 0)
  idx1 <- which(subj_df$y == 1)
  
  fold_subj[idx0] <- sample(rep(1:k, length.out = length(idx0)))
  fold_subj[idx1] <- sample(rep(1:k, length.out = length(idx1)))
  
  map <- setNames(fold_subj, subj_df$Subject)
  as.integer(map[as.character(subject_vec)])
}

# average charge calculated on the train and also applied to the test
impute_mean_train_apply <- function(trainX, otherX) {
  mu <- colMeans(trainX, na.rm = TRUE)
  for (j in seq_len(ncol(trainX))) {
    if (anyNA(trainX[, j])) trainX[is.na(trainX[, j]), j] <- mu[j]
    if (anyNA(otherX[, j])) otherX[is.na(otherX[, j]), j] <- mu[j]
  }
  list(trainX = trainX, otherX = otherX)
}

# min-max scaling calculated on the train and applied to the test
minmax_train_apply <- function(trainX, otherX) {
  mn <- apply(trainX, 2, function(v) min(v, na.rm = TRUE))
  mx <- apply(trainX, 2, function(v) max(v, na.rm = TRUE))
  rng <- mx - mn
  rng[rng == 0 | is.na(rng)] <- 1
  
  trainX2 <- sweep(trainX, 2, mn, "-")
  otherX2 <- sweep(otherX, 2, mn, "-")
  trainX2 <- sweep(trainX2, 2, rng, "/")
  otherX2 <- sweep(otherX2, 2, rng, "/")
  
  list(trainX = trainX2, otherX = otherX2)
}


## INNER LAMBDA SELECTION (SUBJECT-GROUPED) — parallel over lambdas
# for a given alpha and an outer-train:
# - creates 5 folds at a subject level 
# -initial glmnet fit to obtain the lambda grid
# -reduce to a maximum of 40 lambda values
# - For each lambda:
#   - Perform 5 internal training/validations
#   - Calculate AUC on validation
#   - Averaging AUC over 5 folds
# - Select the lambda with the maximum average AUC
# this part is parallelized, so each lambda is evaluated by a different worker

inner_choose_lambda_parallel <- function(Xtrain_df, ytrain, subjects_train,
                                         alpha, inner_k = 5, seed = 1,
                                         n_lambda_keep = 40) {
  
  foldid <- make_foldid_by_subject(subjects_train, ytrain, k = inner_k, seed = seed)
  folds <- sort(unique(foldid))
  
  base_fit <- glmnet(x = as.matrix(Xtrain_df), y = ytrain,
                     family = "binomial", alpha = alpha)
  lambda_grid <- base_fit$lambda
  
  if (length(lambda_grid) > n_lambda_keep) {
    keep <- unique(round(seq(1, length(lambda_grid), length.out = n_lambda_keep)))
    lambda_grid <- lambda_grid[keep]
  }
  
  fold_cache <- lapply(folds, function(f) {
    idx_val <- which(foldid == f)
    idx_tr  <- which(foldid != f)
    list(
      Xtr = as.matrix(Xtrain_df[idx_tr, , drop = FALSE]),
      ytr = ytrain[idx_tr],
      Xva = as.matrix(Xtrain_df[idx_val, , drop = FALSE]),
      yva = ytrain[idx_val]
    )
  })
  
  auc_by_lambda <- foreach(
    li = seq_along(lambda_grid),
    .combine = c,
    .packages = c("glmnet", "pROC")
  ) %dopar% {
    
    lam <- lambda_grid[li]
    aucs <- c()
    
    for (fc in fold_cache) {
      fit <- glmnet(x = fc$Xtr, y = fc$ytr, family = "binomial", alpha = alpha, lambda = lam)
      prob <- as.numeric(predict(fit, newx = fc$Xva, type = "response", s = lam))
      
      if (length(unique(fc$yva)) < 2) {
        aucs <- c(aucs, NA_real_)
      } else {
        aucs <- c(aucs, as.numeric(pROC::auc(pROC::roc(fc$yva, prob, quiet = TRUE))))
      }
    }
    mean(aucs, na.rm = TRUE)
  }
  
  best_lambda <- if (all(is.na(auc_by_lambda))) max(lambda_grid) else lambda_grid[which.max(auc_by_lambda)]
  list(best_lambda = best_lambda)
}


## STORAGE

results <- tibble(
  alpha = numeric(),
  Accuracy = numeric(),
  sd_Accuracy = numeric(),
  Sensitivity = numeric(),
  sd_Sensitivity= numeric(),
  Specificity = numeric(),
  sd_Specificity = numeric(),
  BalancedAccuracy = numeric(),
  AUC = numeric(),
  sd_AUC = numeric(),
  MeanNonZero = numeric()
)

nonzero_freq_by_alpha <- vector("list", length(alphas))
names(nonzero_freq_by_alpha) <- as.character(alphas)


## OUTER LOOP (SEQUENTIAL) + INNER (PARALLEL)
# For each alpha, repeats 20 times:
# - create 5 layered external folds
# - for each fold:
#   - Split train/test by subjects
#   - Apply train-only preprocessing (impute+minmax)
#   - Chooses lambda with inner CV
#   - Fits the final model on the outer-train with that lambda
#   - Predicts on the outer-test (probability), then classes with threshold 0.5
#   - Computes accuracy, sensitivity, specificity, balanced accuracy, AUC
#   - Register how many features have non zero coefficient and which

for (a in alphas) {
  
  cat("\n==================== alpha =", a, "====================\n")
  
  acc_vec <- c(); sens_vec <- c(); spec_vec <- c(); balacc_vec <- c(); auc_vec <- c()
  nonzero_counts <- c()
  nz_freq <- list()
  
  for (rep in 1:num_repeats) {
    
    folds_subj <- make_subject_folds(subject_table, num_folds, seed = 1000 + rep)
    
    for (fold in 1:num_folds) {
      
      test_subjects  <- subject_table$Subject[folds_subj == fold]
      train_subjects <- subject_table$Subject[folds_subj != fold]
      
      train_df <- data_balanced %>% filter(Subject %in% train_subjects)
      test_df  <- data_balanced %>% filter(Subject %in% test_subjects)
      
      vars <- preselected_features
      
      Xtrain <- train_df %>% select(all_of(vars)) %>% as.matrix()
      Xtest  <- test_df  %>% select(all_of(vars)) %>% as.matrix()
      ytrain <- train_df$Faller
      ytest  <- test_df$Faller
      
      # TRAIN-only impute + scale
      imp <- impute_mean_train_apply(Xtrain, Xtest)
      Xtrain_m <- imp$trainX
      Xtest_m  <- imp$otherX
      
      sc <- minmax_train_apply(Xtrain_m, Xtest_m)
      Xtrain_s <- sc$trainX
      Xtest_s  <- sc$otherX
      
      Xtrain_df <- as.data.frame(Xtrain_s); names(Xtrain_df) <- vars
      Xtest_df  <- as.data.frame(Xtest_s);  names(Xtest_df)  <- vars
      
      # INNER lambda selection
      inner <- inner_choose_lambda_parallel(
        Xtrain_df = Xtrain_df,
        ytrain = ytrain,
        subjects_train = train_df$Subject,
        alpha = a,
        inner_k = inner_k,
        seed = 20000 + rep*100 + fold,
        n_lambda_keep = N_LAMBDA_KEEP
      )
      best_lambda <- inner$best_lambda
      
      # Fit final model on outer-train
      fit_outer <- glmnet(
        x = as.matrix(Xtrain_df), y = ytrain,
        family = "binomial",
        alpha = a,
        lambda = best_lambda
      )
      
      prob <- as.numeric(predict(fit_outer, newx = as.matrix(Xtest_df), type = "response", s = best_lambda))
      pred <- ifelse(prob > 0.5, 1, 0)
      
      cm <- table(True = ytest, Predicted = pred)
      TN <- safe_cm_get(cm, "0", "0")
      TP <- safe_cm_get(cm, "1", "1")
      FP <- safe_cm_get(cm, "0", "1")
      FN <- safe_cm_get(cm, "1", "0")
      
      total <- TN + TP + FP + FN
      accuracy    <- (TP + TN) / total
      sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
      specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), 0)
      bal_acc     <- (sensitivity + specificity) / 2
      
      auc_val <- if (length(unique(ytest)) < 2) NA_real_ else
        as.numeric(pROC::auc(pROC::roc(ytest, prob, quiet = TRUE)))
      
      acc_vec    <- c(acc_vec, accuracy)
      sens_vec   <- c(sens_vec, sensitivity)
      spec_vec   <- c(spec_vec, specificity)
      balacc_vec <- c(balacc_vec, bal_acc)
      auc_vec    <- c(auc_vec, auc_val)
      
      # non-zero coefficient stability
      b <- as.matrix(coef(fit_outer, s = best_lambda))
      nz <- rownames(b)[which(b[, 1] != 0)]
      nz <- setdiff(nz, "(Intercept)")
      nonzero_counts <- c(nonzero_counts, length(nz))
      for (v in nz) nz_freq[[v]] <- (nz_freq[[v]] %||% 0) + 1
    }
  }
  
  nonzero_freq_by_alpha[[as.character(a)]] <- nz_freq
  
  results <- results %>%
    add_row(
      alpha = a,
      Accuracy = mean(acc_vec, na.rm = TRUE),
      sd_Accuracy = sd(acc_vec, na.rm = TRUE),
      Sensitivity = mean(sens_vec, na.rm = TRUE),
      sd_Sensitivity = sd(sens_vec, na.rm = TRUE),
      Specificity = mean(spec_vec, na.rm = TRUE),
      sd_Specificity = sd(spec_vec, na.rm = TRUE),
      BalancedAccuracy = mean(balacc_vec, na.rm = TRUE),
      AUC = mean(auc_vec, na.rm = TRUE),
      sd_AUC = sd(auc_vec, na.rm = TRUE),
      MeanNonZero = mean(nonzero_counts, na.rm = TRUE)
    )
}


## RESULTS + BEST ALPHA

print(results)
best_alpha <- results$alpha[which.max(results$AUC)]
cat("\nBest alpha (highest mean OUTER AUC) =", best_alpha, "\n")


## NON-ZERO COEF STABILITY SUMMARY

nz_stability <- purrr::imap_dfr(nonzero_freq_by_alpha, function(freq_list, a_chr) {
  if (length(freq_list) == 0) {
    return(tibble(alpha = as.numeric(a_chr), Feature = character(), Count = integer()))
  }
  tibble(
    alpha = as.numeric(a_chr),
    Feature = names(freq_list),
    Count = as.integer(unlist(freq_list))
  )
})

total_outer_fits <- num_repeats * num_folds
nz_stability <- nz_stability %>%
  mutate(Percent = 100 * Count / total_outer_fits) %>%
  arrange(alpha, desc(Count))


## EXPORT

write_xlsx(
  list(
    "Wilcoxon_UNBAL_pvalues"       = wilcox_table,
    paste0("Wilcoxon_selected_p", p_thresh_global) = tibble(Feature = preselected_features),
    "CV_performance"              = results,
    "NonZeroCoef_stability"       = nz_stability
  ),
  path = out_path
)

cat("\nSaved:", out_path, "\n")

### ----------------------------------------------------
### 15) CLEANUP PARALLEL
### ----------------------------------------------------
parallel::stopCluster(cl)
doParallel::registerDoSEQ()
cat("\nDone. Cluster stopped.\n")
