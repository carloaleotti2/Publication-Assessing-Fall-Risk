
# SVM — FALLER vs NONFALLER (SWAY + GAIT + TUG)
# TRUE NESTED Repeated CV (LEAK-FREE preprocessing) + SUBJECT-GROUPED folds
#
# Pipeline:
# 1) Load SWAY/GAIT/TUG + FRAT label
# 2) Average features per Subject + prefix names 
# 3) Merge to subject-level dataset (unbalanced)
# 4) Global feature selection with Wilcoxon on unbalanced data 
# 5) Balance subjects 
# 6) Nested Repeated CV:
#   - Outer: 5 fold x 20 repeats (generalization estimate)
#   - Inner: 5-fold x 2 repeats (hyperparameter tuning), parallel grid search
# 7) Export results per kernel


rm(list = ls()); gc()


# LIBRARIES

library(readxl)
library(dplyr)
library(tidyr)
library(e1071)
library(pROC)
library(doParallel)
library(foreach)
library(writexl)
library(parallel)


# USER SETTINGS

SEED_MAIN <- 123

# Outer CV design: 5-fold repeated 20 times (100 outer evaluations)
OUTER_K <- 5
OUTER_REPEATS <- 20

# Inner CV design: 5-fold repeated 2 times (tuning only)
INNER_K <- 5
INNER_REPEATS <- 2   

# Balancing at subject level (after Wilcoxon, before CV)
N_PER_CLASS <- 28

# feature selection (GLOBAL, pre-CV, on unbalanced data)
WILCOX_P_THRESH <- 0.15     
WILCOX_TOP_N_CAP <- 100     
WILCOX_MIN_FEATURES <- 5    
DO_FDR <- FALSE            
FDR_METHOD <- "BH"

# Kernels to execute
RUN_LINEAR <- TRUE
RUN_RBF    <- TRUE
RUN_POLY   <- TRUE

# compact hyperparameter grids
C_grid <- c(0.25, 0.5, 1, 2, 4)

# RBF grid uses sigma; e1071 wants gamma, so later gamma = 1/(2*sigma^2) 
sigma_grid <- c(0.005, 0.01, 0.05)   

# Polynomial grid (degree fixed; gamma tested via scale_grid)
degree_grid <- 2
scale_grid  <- c(0.001, 0.01)        

# Paths
base_path <- "C:/Users/Carlo/Documents/Università varia/Tirocinio_Tesi/motori/"
out_dir   <- "C:/Users/Carlo/Documents/Università varia/Tirocinio_Tesi/Features/"


## LOAD DATA (SWAY + GAIT + TUG + FRAT)

# Load raw feature tables
sway <- read_excel(file.path(base_path, "SWAY_combined_results_T0.xlsx"))
gait <- read_excel(file.path(base_path, "GAIT_combined_results_T0.xlsx"))
tug  <- read_excel(file.path(base_path, "TUG_combined_results_T0.xlsx"))

# Load FRAT and construct subject-level label:
# - "true"/"false" -> TRUE/FALSE
# - aggregate by Subject using any(): if any record is TRUE => Faller TRUE
frat <- read_excel(file.path(base_path, "FRAT_combined_results_T0.xlsx")) %>%
  mutate(
    IlSoggetto_CadutoNegliUltimi12Mesi_ = tolower(IlSoggetto_CadutoNegliUltimi12Mesi_),
    Faller = case_when(
      IlSoggetto_CadutoNegliUltimi12Mesi_ == "true"  ~ TRUE,
      IlSoggetto_CadutoNegliUltimi12Mesi_ == "false" ~ FALSE,
      TRUE ~ NA
    )
  ) %>%
  group_by(Subject) %>%
  summarise(Faller = any(Faller, na.rm = TRUE), .groups = "drop")


## SUBJECT-AVERAGE EACH MODALITY + PREFIX
# Helper: for each modality, compute mean of numeric columns per Subject,
# then add prefix to feature names to avoid collisions after merging. 
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


## MERGE + LABEL (UNBALANCED DATASET, SUBJECT-LEVEL)
# merge label + all averaged modalities
# keep only subjects with known label
# convert label into factor with fixed levels

data_ml <- frat %>%
  left_join(sway_avg, by = "Subject") %>%
  left_join(gait_avg, by = "Subject") %>%
  left_join(tug_avg,  by = "Subject") %>%
  filter(!is.na(Faller)) %>%
  mutate(Faller = factor(Faller, levels = c(TRUE, FALSE), labels = c("Faller", "NonFaller")))

# Replace NaN -> NA (safe)
data_ml <- data_ml %>%
  mutate(across(where(is.numeric), ~ ifelse(is.nan(.x), NA, .x)))

cat("\nUnbalanced subjects:", n_distinct(data_ml$Subject), "\n")
print(table(data_ml$Faller))


## Feature selection Wilcoxon

# wilcox_select_global:
# - runs wilcoxon for each numeric feature
# - uses p-value or (optionally) FDR-adjusted p_adj as selection score
# - selects features with score <= threshold
# - ensures at least min_features (best p-values)
# - applies cap to avoid too many selected features 
wilcox_select_global <- function(df, y_col = "Faller",
                                 p_thresh = 0.10,
                                 top_n_cap = 60,
                                 min_features = 5,
                                 do_fdr = FALSE,
                                 fdr_method = "BH") {
  y <- df[[y_col]]
  stopifnot(is.factor(y), length(levels(y)) == 2)
  
  # Candidate predictors: numeric columns only (exclude Subject if numeric by mistake)
  num_cols <- names(df)[sapply(df, is.numeric)]
  num_cols <- setdiff(num_cols, c(y_col, "Subject"))
  
  res <- lapply(num_cols, function(v) {
    x <- df[[v]]
    x1 <- x[y == levels(y)[1]]
    x2 <- x[y == levels(y)[2]]
    if (sum(!is.na(x1)) < 2 || sum(!is.na(x2)) < 2) {
      return(data.frame(feature = v, p_value = NA_real_))
    }
    pv <- tryCatch(
      wilcox.test(x ~ y, data = df, exact = FALSE)$p.value,
      error = function(e) NA_real_
    )
    data.frame(feature = v, p_value = pv)
  })
  
  res_df <- bind_rows(res) %>%
    mutate(
      p_adj = if (do_fdr) p.adjust(p_value, method = fdr_method) else NA_real_,
      score = if (do_fdr) p_adj else p_value
    ) %>%
    arrange(score)
  
  keep <- res_df %>%
    filter(!is.na(score), score <= p_thresh) %>%
    pull(feature)
  
  if (length(keep) < min_features) {
    keep <- res_df %>%
      filter(!is.na(score)) %>%
      slice_head(n = min_features) %>%
      pull(feature)
  }
  
  if (!is.null(top_n_cap) && length(keep) > top_n_cap) {
    keep <- res_df %>%
      filter(feature %in% keep) %>%
      arrange(score) %>%
      slice_head(n = top_n_cap) %>%
      pull(feature)
  }
  
  list(features = keep, table = res_df)
}

set.seed(SEED_MAIN)
wilcox_out <- wilcox_select_global(
  df = data_ml,
  p_thresh = WILCOX_P_THRESH,
  top_n_cap = WILCOX_TOP_N_CAP,
  min_features = WILCOX_MIN_FEATURES,
  do_fdr = DO_FDR,
  fdr_method = FDR_METHOD
)

preselected_features <- wilcox_out$features
wilcox_table <- wilcox_out$table

cat("\nWilcoxon-selected features (GLOBAL, before balancing):", length(preselected_features), "\n")
print(preselected_features)

if (length(preselected_features) == 0) stop("No features selected by Wilcoxon. Adjust threshold/cap/min.")


## BALANCE AT SUBJECT LEVEL (N_PER_CLASS + N_PER_CLASS) — AFTER WILCOXON
# Sample N_PER_CLASS subjects per class and filter the dataset accordingly 
set.seed(SEED_MAIN)

subj_tbl <- data_ml %>% distinct(Subject, Faller)
subj_f <- subj_tbl %>% filter(Faller == "Faller") %>% pull(Subject)
subj_n <- subj_tbl %>% filter(Faller == "NonFaller") %>% pull(Subject)

if (length(subj_f) < N_PER_CLASS || length(subj_n) < N_PER_CLASS) {
  stop(paste0("Not enough subjects for balancing: Fallers=", length(subj_f),
              ", NonFallers=", length(subj_n)))
}

subj_keep <- c(sample(subj_f, N_PER_CLASS), sample(subj_n, N_PER_CLASS))
data_balanced <- data_ml %>% filter(Subject %in% subj_keep)

cat("\nBalanced subjects:", n_distinct(data_balanced$Subject), "\n")
cat("Rows (subject-level):", nrow(data_balanced), "\n")
print(table(data_balanced$Faller))


## HELPERS (folding + preprocessing + metrics)

# make_subject_folds_once: creates a stratified subject-level folds by assigning fold IDs separately in each class
make_subject_folds_once <- function(subj_df, k, seed) {
  set.seed(seed)
  folds <- rep(NA_integer_, nrow(subj_df))
  idx_f  <- which(subj_df$Faller == "Faller")
  idx_nf <- which(subj_df$Faller == "NonFaller")
  folds[idx_f]  <- sample(rep(1:k, length.out = length(idx_f)))
  folds[idx_nf] <- sample(rep(1:k, length.out = length(idx_nf)))
  subj_df$Fold <- folds
  subj_df
}

# impute using train means
impute_train_apply <- function(Xtr, Xte) {
  mu <- colMeans(Xtr, na.rm = TRUE)
  for (j in seq_len(ncol(Xtr))) {
    Xtr[is.na(Xtr[, j]), j] <- mu[j]
    Xte[is.na(Xte[, j]), j] <- mu[j]
  }
  list(Xtr = Xtr, Xte = Xte)
}

# scale using train mean/sd (z-score)
scale_train_apply <- function(Xtr, Xte) {
  m <- apply(Xtr, 2, mean, na.rm = TRUE)
  s <- apply(Xtr, 2, sd,   na.rm = TRUE)
  s[s == 0 | is.na(s)] <- 1
  Xtr <- sweep(Xtr, 2, m, "-"); Xte <- sweep(Xte, 2, m, "-")
  Xtr <- sweep(Xtr, 2, s, "/"); Xte <- sweep(Xte, 2, s, "/")
  list(Xtr = Xtr, Xte = Xte)
}

# AUC safe computation (returns NA if only one class present)
safe_auc <- function(y_true, prob) {
  if (length(unique(y_true)) < 2) return(NA_real_)
  as.numeric(pROC::auc(pROC::roc(y_true, prob, levels = c("NonFaller","Faller"), quiet = TRUE)))
}

# Standard metrics from predicted probability using thr=0.5
compute_metrics <- function(y_true, prob, thr = 0.5) {
  pred <- factor(ifelse(prob > thr, "Faller", "NonFaller"), levels = c("NonFaller","Faller"))
  y_true <- factor(y_true, levels = c("NonFaller","Faller"))
  cm <- table(True = y_true, Pred = pred)
  
  TN <- ifelse("NonFaller" %in% rownames(cm) && "NonFaller" %in% colnames(cm), cm["NonFaller","NonFaller"], 0)
  TP <- ifelse("Faller"    %in% rownames(cm) && "Faller"    %in% colnames(cm), cm["Faller","Faller"], 0)
  FP <- ifelse("NonFaller" %in% rownames(cm) && "Faller"    %in% colnames(cm), cm["NonFaller","Faller"], 0)
  FN <- ifelse("Faller"    %in% rownames(cm) && "NonFaller" %in% colnames(cm), cm["Faller","NonFaller"], 0)
  
  total <- TN + TP + FP + FN
  acc <- (TP + TN) / total
  sens <- ifelse((TP + FN) > 0, TP/(TP+FN), NA_real_)
  spec <- ifelse((TN + FP) > 0, TN/(TN+FP), NA_real_)
  balacc <- (sens + spec)/2
  auc <- safe_auc(y_true, prob)
  
  list(Accuracy=acc, Sensitivity=sens, Specificity=spec, BalancedAccuracy=balacc, AUC=auc)
}

# inner_cv_score:
# - given outer-train, kernel and hyperparameters
# - performs inner subject-grouped repeated CV
# - with preprocessing inside each inner split
# - returns mean AUC as tuning objective 
inner_cv_score <- function(train_df, vars, kernel, params, inner_k, inner_repeats, seed_base) {
  
  subj_df <- train_df %>% distinct(Subject, Faller)
  aucs <- c()
  
  for (rr in 1:inner_repeats) {
    subj_folds <- make_subject_folds_once(subj_df, k = inner_k, seed = seed_base + rr)
    
    for (ff in 1:inner_k) {
      te_subj <- subj_folds$Subject[subj_folds$Fold == ff]
      
      tr_df <- train_df %>% filter(!Subject %in% te_subj)
      va_df <- train_df %>% filter( Subject %in% te_subj)
      
      Xtr <- tr_df %>% select(all_of(vars)) %>% as.matrix()
      Xva <- va_df %>% select(all_of(vars)) %>% as.matrix()
      ytr <- tr_df$Faller
      yva <- va_df$Faller
      
      imp <- impute_train_apply(Xtr, Xva); Xtr <- imp$Xtr; Xva <- imp$Xte
      sc  <- scale_train_apply(Xtr, Xva);  Xtr <- sc$Xtr;  Xva <- sc$Xte
      
      if (kernel == "linear") {
        fit <- e1071::svm(x = Xtr, y = ytr, kernel = "linear",
                          cost = params$C, probability = TRUE, scale = FALSE)
      } else if (kernel == "rbf") {
        gamma_val <- 1 / (2 * (params$sigma^2))
        fit <- e1071::svm(x = Xtr, y = ytr, kernel = "radial",
                          cost = params$C, gamma = gamma_val,
                          probability = TRUE, scale = FALSE)
      } else { # poly
        fit <- e1071::svm(x = Xtr, y = ytr, kernel = "polynomial",
                          cost = params$C, degree = params$degree,
                          gamma = params$scale,
                          probability = TRUE, scale = FALSE)
      }
      
      pr <- predict(fit, Xva, probability = TRUE)
      prob <- attr(pr, "probabilities")[, "Faller"]
      aucs <- c(aucs, safe_auc(yva, prob))
    }
  }
  
  mean(aucs, na.rm = TRUE)
}


# PARALLEL BACKEND (for inner tuning)
# starts a cluster used by foreach %dopar% in the tuning grid evaluation
set.seed(SEED_MAIN)
n_cores <- max(1, parallel::detectCores() - 1)
cl <- parallel::makeCluster(n_cores)
doParallel::registerDoParallel(cl)

# Export needed functions/objects to workers
parallel::clusterExport(
  cl,
  varlist = c(
    "inner_cv_score",
    "make_subject_folds_once",
    "impute_train_apply",
    "scale_train_apply",
    "safe_auc",
    "compute_metrics"
  ),
  envir = .GlobalEnv
)
cat("\nUsing", n_cores, "cores for INNER tuning.\n")
cat("Workers:", foreach::getDoParWorkers(), "\n")


# NESTED SVM RUNNER (ALL MODALITIES, FIXED FEATURES)
# run_nested_svm:
# - Outer repeated CV loop
# - For each outer split: 
#   * inner parallel grid search selects best hyperparameters (max mean AUC)
#   * fit final model on outer-train with best params
#   * evaluate on outer-test -> store metrics and selected params
run_nested_svm <- function(df, vars, kernel = c("linear","rbf","poly"),
                           outer_k=5, outer_repeats=20,
                           inner_k=5, inner_repeats=2,
                           C_grid=c(1),
                           sigma_grid=NULL,
                           degree_grid=2,
                           scale_grid=NULL,
                           seed_main=123) {
  
  kernel <- match.arg(kernel)
  
  if (kernel == "linear") {
    grid <- expand.grid(C = C_grid)
  } else if (kernel == "rbf") {
    grid <- expand.grid(C = C_grid, sigma = sigma_grid)
  } else {
    grid <- expand.grid(C = C_grid, degree = degree_grid, scale = scale_grid)
  }
  
  subj_df_all <- df %>% distinct(Subject, Faller)
  out_rows <- list()
  outer_counter <- 0
  
  for (rr in 1:outer_repeats) {
    
    subj_folds <- make_subject_folds_once(subj_df_all, k = outer_k, seed = seed_main + rr)
    
    for (ff in 1:outer_k) {
      
      outer_counter <- outer_counter + 1
      
      te_subj <- subj_folds$Subject[subj_folds$Fold == ff]
      train_df <- df %>% filter(!Subject %in% te_subj)
      test_df  <- df %>% filter( Subject %in% te_subj)
      
      seed_base_inner <- 100000 + outer_counter*10
      
      inner_auc <- foreach(
        g = 1:nrow(grid),
        .combine  = c,
        .packages = c("dplyr","e1071","pROC")
      ) %dopar% {
        params <- as.list(grid[g, , drop = FALSE])
        inner_cv_score(train_df, vars, kernel, params,
                       inner_k = inner_k, inner_repeats = inner_repeats,
                       seed_base = seed_base_inner + g*1000)
      }
      
      best_g <- which.max(inner_auc)
      best_params <- as.list(grid[best_g, , drop = FALSE])
      
      Xtr <- train_df %>% select(all_of(vars)) %>% as.matrix()
      Xte <- test_df  %>% select(all_of(vars)) %>% as.matrix()
      ytr <- train_df$Faller
      yte <- test_df$Faller
      
      imp <- impute_train_apply(Xtr, Xte); Xtr <- imp$Xtr; Xte <- imp$Xte
      sc  <- scale_train_apply(Xtr, Xte);  Xtr <- sc$Xtr;  Xte <- sc$Xte
      
      if (kernel == "linear") {
        fit <- e1071::svm(x = Xtr, y = ytr, kernel = "linear",
                          cost = best_params$C, probability = TRUE, scale = FALSE)
      } else if (kernel == "rbf") {
        gamma_val <- 1 / (2 * (best_params$sigma^2))
        fit <- e1071::svm(x = Xtr, y = ytr, kernel = "radial",
                          cost = best_params$C, gamma = gamma_val,
                          probability = TRUE, scale = FALSE)
      } else {
        fit <- e1071::svm(x = Xtr, y = ytr, kernel = "polynomial",
                          cost = best_params$C, degree = best_params$degree,
                          gamma = best_params$scale,
                          probability = TRUE, scale = FALSE)
      }
      
      pr <- predict(fit, Xte, probability = TRUE)
      prob <- attr(pr, "probabilities")[, "Faller"]
      
      m <- compute_metrics(y_true = yte, prob = prob, thr = 0.5)
      
      out_rows[[outer_counter]] <- data.frame(
        OuterRepeat = rr,
        OuterFold   = ff,
        N_Test_Subj = length(unique(test_df$Subject)),
        NFeat       = length(vars),
        Features    = paste(vars, collapse = ";"),
        Best_C      = best_params$C,
        Best_sigma  = ifelse(kernel == "rbf", best_params$sigma, NA_real_),
        Best_degree = ifelse(kernel == "poly", best_params$degree, NA_real_),
        Best_scale  = ifelse(kernel == "poly", best_params$scale, NA_real_),
        Accuracy    = m$Accuracy,
        Sensitivity = m$Sensitivity,
        Specificity = m$Specificity,
        BalancedAcc = m$BalancedAccuracy,
        AUC         = m$AUC
      )
      
      if (outer_counter %% 10 == 0) {
        cat("Completed outer eval:", outer_counter, "of", outer_k*outer_repeats, "(", kernel, ")\n")
      }
    }
  }
  
  out_df <- bind_rows(out_rows)
  
  summary_df <- out_df %>%
    summarise(
      Accuracy_mean    = mean(Accuracy, na.rm = TRUE),
      Accuracy_sd      = sd(Accuracy, na.rm = TRUE),
      Sensitivity_mean = mean(Sensitivity, na.rm = TRUE),
      Sensitivity_sd   = sd(Sensitivity, na.rm = TRUE),
      Specificity_mean = mean(Specificity, na.rm = TRUE),
      Specificity_sd   = sd(Specificity, na.rm = TRUE),
      BalancedAcc_mean = mean(BalancedAcc, na.rm = TRUE),
      BalancedAcc_sd   = sd(BalancedAcc, na.rm = TRUE),
      AUC_mean         = mean(AUC, na.rm = TRUE),
      AUC_sd           = sd(AUC, na.rm = TRUE)
    )
  
  list(per_fold = out_df, summary = summary_df, grid = grid)
}


## RUN NESTED SVMs (ALL MODALITIES)
# execute selected kernels; each returns per-fold results + summary 
res_list <- list()

if (RUN_LINEAR) {
  cat("\n==================== NESTED SVM LINEAR (ALL) ====================\n")
  res_list$linear <- run_nested_svm(
    df = data_balanced,
    vars = preselected_features,
    kernel = "linear",
    outer_k = OUTER_K, outer_repeats = OUTER_REPEATS,
    inner_k = INNER_K, inner_repeats = INNER_REPEATS,
    C_grid = C_grid,
    seed_main = SEED_MAIN
  )
  print(res_list$linear$summary)
}

if (RUN_POLY) {
  cat("\n==================== NESTED SVM POLY (ALL) ====================\n")
  res_list$poly <- run_nested_svm(
    df = data_balanced,
    vars = preselected_features,
    kernel = "poly",
    outer_k = OUTER_K, outer_repeats = OUTER_REPEATS,
    inner_k = INNER_K, inner_repeats = INNER_REPEATS,
    C_grid = C_grid,
    degree_grid = degree_grid,
    scale_grid = scale_grid,
    seed_main = SEED_MAIN
  )
  print(res_list$poly$summary)
}

if (RUN_RBF) {
  cat("\n==================== NESTED SVM RBF (ALL) ====================\n")
  res_list$rbf <- run_nested_svm(
    df = data_balanced,
    vars = preselected_features,
    kernel = "rbf",
    outer_k = OUTER_K, outer_repeats = OUTER_REPEATS,
    inner_k = INNER_K, inner_repeats = INNER_REPEATS,
    C_grid = C_grid,
    sigma_grid = sigma_grid,
    seed_main = SEED_MAIN
  )
  print(res_list$rbf$summary)
}


## STOP CLUSTER
# Shutdown parallel backend
stopCluster(cl)
registerDoSEQ()


## EXPORT (one xlsx per kernel) + Wilcoxon table

common_sheets <- list(
  "WILCOXON_table_unbalanced" = wilcox_table,
  "WILCOXON_features_used" = data.frame(Feature = preselected_features)
)

if (!is.null(res_list$linear)) {
  write_xlsx(
    c(common_sheets,
      list("per_fold" = res_list$linear$per_fold,
           "summary"  = res_list$linear$summary)),
    path = file.path(out_dir, paste0("SVM_ALL_Linear_Wilcoxon_p", WILCOX_P_THRESH, "_Nested.xlsx"))
  )
}

if (!is.null(res_list$poly)) {
  write_xlsx(
    c(common_sheets,
      list("per_fold" = res_list$poly$per_fold,
           "summary"  = res_list$poly$summary)),
    path = file.path(out_dir, paste0("SVM_ALL_Poly_Wilcoxon_p", WILCOX_P_THRESH, "_Nested.xlsx"))
  )
}

if (!is.null(res_list$rbf)) {
  write_xlsx(
    c(common_sheets,
      list("per_fold" = res_list$rbf$per_fold,
           "summary"  = res_list$rbf$summary)),
    path = file.path(out_dir, paste0("SVM_ALL_RBF_Wilcoxon_p", WILCOX_P_THRESH, "_Nested.xlsx"))
  )
}

cat("\nDone.\n")
