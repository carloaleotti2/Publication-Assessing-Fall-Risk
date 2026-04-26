################################################################################
# LIGHT NESTED (REPEATED) CV — RANDOM FOREST via ranger (REPRODUCIBLE)
# SWAY + GAIT + TUG (all averaged per subject, then merged)
#
# REPRODUCIBILITY FIXES:
# 1) RNGkind("L'Ecuyer-CMRG") + set.seed(SEED_MAIN) at top
# 2) deterministic ordering before balancing (arrange(Subject))
# 3) clusterSetRNGStream(cl, SEED_MAIN) after makeCluster()
# 4) caret::trainControl(seeds=...) so tuning is deterministic in parallel
# 5) ranger seed in both tuning (caret) and final refit
# 6) on.exit cleanup (cluster always closed even if errors)
################################################################################

rm(list = ls()); gc()

## LIBRARIES
library(readxl)
library(dplyr)
library(caret)
library(pROC)
library(ranger)

library(doParallel)
library(parallel)
library(tibble)

## GLOBAL RNG (IMPORTANT)
RNGkind("L'Ecuyer-CMRG")
SEED_MAIN <- 123
set.seed(SEED_MAIN)

## USER KNOBS
OUTER_K <- 5
OUTER_REPEATS <- 20

INNER_K <- 5
INNER_REPEATS <- 2

TUNE_TREES  <- 300
FINAL_TREES <- 1000

N_CORES <- max(1, parallel::detectCores() - 1)

N_PER_CLASS <- 28

## LOAD DATA
base_path <- "C:/Users/Carlo/Documents/Università varia/Tirocinio_Tesi/motori/"

sway <- read_excel(file.path(base_path, "SWAY_combined_results_T0.xlsx"))
gait <- read_excel(file.path(base_path, "GAIT_combined_results_T0.xlsx"))
tug  <- read_excel(file.path(base_path, "TUG_combined_results_T0.xlsx"))

frat <- read_excel(file.path(base_path, "FRAT_combined_results_T0.xlsx")) %>%
  mutate(
    IlSoggetto_CadutoNegliUltimi12Mesi_ = tolower(IlSoggetto_CadutoNegliUltimi12Mesi_),
    Faller = case_when(
      IlSoggetto_CadutoNegliUltimi12Mesi_ == "true"  ~ TRUE,
      IlSoggetto_CadutoNegliUltimi12Mesi_ == "false" ~ FALSE,
      TRUE ~ NA
    )
  ) %>%
  dplyr::select(Subject, Faller) %>%
  mutate(Subject = as.character(Subject)) %>%
  arrange(Subject)

## HELPER: AVERAGE ALL TRIALS PER SUBJECT + PREFIX FEATURES
avg_subject_prefix <- function(df, prefix) {
  out <- df %>%
    mutate(Subject = as.character(Subject)) %>%
    group_by(Subject) %>%
    summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
    arrange(Subject)
  
  nm <- names(out)
  nm[nm != "Subject"] <- paste0(prefix, "__", nm[nm != "Subject"])
  names(out) <- nm
  out
}

sway_avg <- avg_subject_prefix(sway, "SWAY")
gait_avg <- avg_subject_prefix(gait, "GAIT")
tug_avg  <- avg_subject_prefix(tug,  "TUG")

## MERGE MODALITIES (subject-level) + LABEL
data_merged <- frat %>%
  left_join(sway_avg, by = "Subject") %>%
  left_join(gait_avg, by = "Subject") %>%
  left_join(tug_avg,  by = "Subject") %>%
  filter(!is.na(Faller)) %>%
  arrange(Subject) %>%
  mutate(
    Faller = factor(Faller, levels = c(TRUE, FALSE), labels = c("Faller", "NonFaller"))
  ) %>%
  mutate(across(where(is.numeric), ~ ifelse(is.nan(.x), NA, .x)))

cat("\nAvailable subjects after merge:\n")
print(table(data_merged$Faller))
cat("N subjects total:", nrow(data_merged), "\n")

## BALANCE (N_PER_CLASS + N_PER_CLASS) — DETERMINISTIC
set.seed(SEED_MAIN)

# stabilize before sampling
data_merged <- data_merged %>% arrange(Subject)

fallers_pool    <- data_merged %>% filter(Faller == "Faller")
nonfallers_pool <- data_merged %>% filter(Faller == "NonFaller")

if (nrow(fallers_pool) < N_PER_CLASS) stop("Not enough fallers to sample.")
if (nrow(nonfallers_pool) < N_PER_CLASS) stop("Not enough non-fallers to sample.")

fallers    <- fallers_pool    %>% slice_sample(n = N_PER_CLASS)
nonfallers <- nonfallers_pool %>% slice_sample(n = N_PER_CLASS)

data_balanced <- bind_rows(fallers, nonfallers) %>%
  arrange(Faller, Subject)  # stable

cat("\nBalanced subjects:\n")
print(table(data_balanced$Faller))

# Keep Subject for deterministic ordering/debug; drop for modeling
data_balanced_model <- data_balanced %>% dplyr::select(-Subject)

## HELPER: MEDIAN IMPUTE using outer-train statistics (leak-free)
median_impute <- function(train_x, test_x) {
  med <- sapply(train_x, function(col) {
    m <- median(col, na.rm = TRUE)
    if (is.na(m) || is.infinite(m)) 0 else m
  })
  
  for (nm in names(med)) {
    train_x[[nm]][is.na(train_x[[nm]])] <- med[[nm]]
    test_x[[nm]][is.na(test_x[[nm]])]  <- med[[nm]]
  }
  
  list(train = train_x, test = test_x, medians = med)
}

## OUTER FOLDS (reproducible)
set.seed(SEED_MAIN)
outer_folds <- caret::createMultiFolds(
  y = data_balanced_model$Faller,
  k = OUTER_K,
  times = OUTER_REPEATS
)

## PARALLEL BACKEND (REPRODUCIBLE)
cl <- parallel::makeCluster(N_CORES)
parallel::clusterSetRNGStream(cl, SEED_MAIN)  # <-- crucial
doParallel::registerDoParallel(cl)

on.exit({
  try(parallel::stopCluster(cl), silent = TRUE)
  try(doParallel::registerDoSEQ(), silent = TRUE)
}, add = TRUE)

## STORAGE
outer_results <- tibble(
  outer_split = names(outer_folds),
  AUC = NA_real_,
  Accuracy = NA_real_,
  Sensitivity = NA_real_,
  Specificity = NA_real_,
  BestTune = NA_character_
)

imp_list <- vector("list", length(outer_folds))
names(imp_list) <- names(outer_folds)

## OUTER LOOP (eval) + INNER LOOP (tuning) — REPRODUCIBLE
for (i in seq_along(outer_folds)) {
  
  train_idx <- outer_folds[[i]]
  test_idx  <- setdiff(seq_len(nrow(data_balanced_model)), train_idx)
  
  outer_train <- data_balanced_model[train_idx, , drop = FALSE]
  outer_test  <- data_balanced_model[test_idx,  , drop = FALSE]
  
  # ---- Inner indices (deterministic)
  set.seed(5000 + i)
  inner_index <- caret::createMultiFolds(outer_train$Faller, k = INNER_K, times = INNER_REPEATS)
  
  # ---- Tuning grid (deterministic)
  p <- ncol(outer_train) - 1
  mtry_grid <- unique(pmax(1, pmin(p, round(c(sqrt(p), p/3, p/2)))))
  
  tune_grid <- expand.grid(
    mtry = mtry_grid,
    splitrule = "gini",
    min.node.size = c(1, 5)
  )
  
  # ---- caret reproducibility: seeds list (required for parallel)
  n_resamp <- length(inner_index)    # INNER_K * INNER_REPEATS
  n_grid   <- nrow(tune_grid)
  
  set.seed(900000 + i)
  seeds_list <- vector(mode = "list", length = n_resamp + 1)
  for (r in 1:n_resamp) {
    seeds_list[[r]] <- sample.int(1e7, n_grid)
  }
  seeds_list[[n_resamp + 1]] <- sample.int(1e7, 1)
  
  inner_ctrl <- caret::trainControl(
    method = "repeatedcv",
    number = INNER_K,
    repeats = INNER_REPEATS,
    classProbs = TRUE,
    summaryFunction = caret::twoClassSummary,
    allowParallel = TRUE,
    index = inner_index,
    seeds = seeds_list
  )
  
  # ---- Inner tuning (seeded) + ranger seed
  set.seed(1000 + i)
  fit_inner <- caret::train(
    Faller ~ .,
    data = outer_train,
    method = "ranger",
    metric = "ROC",
    trControl = inner_ctrl,
    tuneGrid = tune_grid,
    num.trees = TUNE_TREES,
    importance = "impurity",
    preProcess = c("medianImpute"),
    num.threads = 1,
    seed = 10000 + i
  )
  
  best <- fit_inner$bestTune
  
  # ---- Leak-free imputation for final refit and test eval (outer_train medians)
  x_train <- outer_train %>% dplyr::select(-Faller)
  x_test  <- outer_test  %>% dplyr::select(-Faller)
  impd <- median_impute(x_train, x_test)
  
  outer_train_imp <- dplyr::bind_cols(impd$train, Faller = outer_train$Faller)
  outer_test_imp  <- dplyr::bind_cols(impd$test,  Faller = outer_test$Faller)
  
  # ---- Refit on OUTER TRAIN with more trees (seeded)
  fit_final <- ranger::ranger(
    Faller ~ .,
    data = outer_train_imp,
    probability = TRUE,
    num.trees = FINAL_TREES,
    mtry = best$mtry,
    min.node.size = best$min.node.size,
    splitrule = as.character(best$splitrule),
    importance = "impurity",
    num.threads = 1,
    seed = 20000 + i
  )
  
  prob_test <- predict(fit_final, data = outer_test_imp)$predictions[, "Faller"]
  pred_test <- factor(ifelse(prob_test >= 0.5, "Faller", "NonFaller"),
                      levels = levels(data_balanced_model$Faller))
  
  roc_obj <- pROC::roc(
    response = outer_test_imp$Faller,
    predictor = prob_test,
    levels = c("NonFaller", "Faller"),
    direction = "<",
    quiet = TRUE
  )
  
  cm <- caret::confusionMatrix(pred_test, outer_test_imp$Faller, positive = "Faller")
  
  outer_results$AUC[i]         <- as.numeric(pROC::auc(roc_obj))
  outer_results$Accuracy[i]    <- unname(cm$overall["Accuracy"])
  outer_results$Sensitivity[i] <- unname(cm$byClass["Sensitivity"])
  outer_results$Specificity[i] <- unname(cm$byClass["Specificity"])
  outer_results$BestTune[i]    <- paste(names(best), best, sep = "=", collapse = "; ")
  
  vi <- fit_final$variable.importance
  imp_list[[i]] <- tibble(Feature = names(vi), Overall = as.numeric(vi))
  
  if (i %% 10 == 0) cat("Completed outer split", i, "of", length(outer_folds), "\n")
}

## SUMMARY PERFORMANCE
perf_summary <- outer_results %>%
  summarise(
    n_splits = n(),
    AUC_mean = mean(AUC, na.rm = TRUE),
    AUC_sd   = sd(AUC, na.rm = TRUE),
    ACC_mean = mean(Accuracy, na.rm = TRUE),
    ACC_sd   = sd(Accuracy, na.rm = TRUE),
    SEN_mean = mean(Sensitivity, na.rm = TRUE),
    SEN_sd   = sd(Sensitivity, na.rm = TRUE),
    SPE_mean = mean(Specificity, na.rm = TRUE),
    SPE_sd   = sd(Specificity, na.rm = TRUE)
  )

print(perf_summary)
print(outer_results)

## FEATURE IMPORTANCE
imp_all <- bind_rows(imp_list) %>%
  group_by(Feature) %>%
  summarise(
    mean_importance = mean(Overall, na.rm = TRUE),
    sd_importance   = sd(Overall, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_importance), Feature)

top20_vars_all <- imp_all$Feature[1:20]
print(top20_vars_all)

# Optional: per-modality top 20
top20_sway <- imp_all %>% filter(grepl("^SWAY__", Feature)) %>% slice_head(n = 20)
top20_gait <- imp_all %>% filter(grepl("^GAIT__", Feature)) %>% slice_head(n = 20)
top20_tug  <- imp_all %>% filter(grepl("^TUG__",  Feature)) %>% slice_head(n = 20)

cat("\nTop 20 overall:\n"); print(head(imp_all, 60))
cat("\nTop 20 SWAY:\n");    print(top20_sway)
cat("\nTop 20 GAIT:\n");    print(top20_gait)
cat("\nTop 20 TUG:\n");     print(top20_tug)

# Keep compatibility with your workflow
var_RF <- imp_all$Feature
# print(var_RF) # can be long


