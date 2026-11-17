library(tidyverse)
library(tidymodels)
library(data.table)
library(randomForest)
library(factoextra)
library(micer)
library(cowplot)
library(factoextra)
library(bundle)
library(permimp)
library(FactoMineR) 
library(vegan)


# ==================================== MODELING WITH TIDYMODELS ====================================
INPUT_DIR <- ""     # Path to CSV files
OUT_DIR <- ""       # Path to save results
RDS_PATH = ""       # Path to save the results as an R object into a file


#Load all tables
unetT <- read_csv(paste0(INPUT_DIR,"UNet_Train.csv"))[,c(4, 9:104)] |> mutate(label = as.factor(label))
unetV <- read_csv(paste0(INPUT_DIR,"UNet_Test.csv"))[,c(4, 9:104)] |> mutate(label = as.factor(label))
mamT <- read_csv(paste0(INPUT_DIR,"MambaUNet_Train.csv"))[,c(4,9:104)] |> mutate(label = as.factor(label))
mamV <- read_csv(paste0(INPUT_DIR,"MambaUNet_Test.csv"))[,c(4, 9:104)] |> mutate(label = as.factor(label))
mamuT <- read_csv(paste0(INPUT_DIR,"MambaUNetC_Train.csv"))[,c(4, 9:104)] |> mutate(label = as.factor(label))
mamuV <- read_csv(paste0(INPUT_DIR,"MambaUNetC_Test.csv"))[,c(4, 9:104)] |> mutate(label = as.factor(label))
swinT <- read_csv(paste0(INPUT_DIR,"SwinUNet_Train.csv"))[,c(4, 9:104)] |> mutate(label = as.factor(label))
swinV <- read_csv(paste0(INPUT_DIR,"SwinUNet_Test.csv"))[,c(4, 9:104)] |> mutate(label = as.factor(label))
swinuT <- read_csv(paste0(INPUT_DIR,"SwinUNetC_Train.csv"))[,c(4, 9:104)] |> mutate(label = as.factor(label))
swinuV <- read_csv(paste0(INPUT_DIR,"SwinUNetC_Test.csv"))[,c(4, 9:104)] |> mutate(label = as.factor(label))

testBool <- data.frame(test = unetT$label == mamT$label)

unetT <- read_csv(paste0(INPUT_DIR,"UNet_Train.csv"))[,c(4, 9:104)] |> mutate(label = as.factor(label))
unetV <- read_csv(paste0(INPUT_DIR,"UNet_Test.csv"))[,c(4, 9:104)] |> mutate(label = as.factor(label))
mamT <- read_csv(paste0(INPUT_DIR,"MambaUNet_Train.csv"))[,c(9:104)]
mamV <- read_csv(paste0(INPUT_DIR,"MambaUNet_Test.csv"))[,c(9:104)]
mamuT <- read_csv(paste0(INPUT_DIR,"MambaUNetC_Train.csv"))[,c(9:104)]
mamuV <- read_csv(paste0(INPUT_DIR,"MambaUNetC_Test.csv"))[,c(9:104)]
swinT <- read_csv(paste0(INPUT_DIR,"SwinUNet_Train.csv"))[,c(9:104)]
swinV <- read_csv(paste0(INPUT_DIR,"SwinUNet_Test.csv"))[,c(9:104)]
swinuT <- read_csv(paste0(INPUT_DIR,"SwinUNetC_Train.csv"))[,c(9:104)]
swinuV <- read_csv(paste0(INPUT_DIR,"SwinUNetC_Test.csv"))[,c(9:104)]

#List Vars
uVars <- c("label", names(unetT)[2:97])
mVars <- c("label", names(mamT))
muVars <- c("label", names(mamuT))
sVars <- c("label", names(swinT))
suVars <- c("label", names(swinuT))

roleV <- c("outcome", rep("predictor", 96))

#Merge all training tables
allTrain <- bind_cols(unetT, mamT, mamuT, swinT, swinuT)
set.seed(42)
allTrain <- allTrain |> group_by(label) |> sample_n(1000, replace=FALSE) |> ungroup()

#Merge all test tables
allTest <- bind_cols(unetV, mamV, mamuV, swinV, swinuV)
set.seed(43)
allTest <- allTest|> sample_n(10000, replace=FALSE)

#Define folds
set.seed(123)
folds <- vfold_cv(allTrain,
                  v = 10,
                  repeats=1,
                  strata = label)

#Define recipes
recipeN_U <- recipe(allTrain, vars = uVars, roles=roleV) |>
  step_range(all_predictors())

recipePCA_U <- recipe(allTrain, vars = uVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_pca(all_predictors(), num_comp=15)

recipekPCA_U <- recipe(allTrain, vars = uVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_kpca(all_predictors(), num_comp=15)

recipeICA_U <- recipe(allTrain, vars = uVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_ica(all_predictors(), num_comp=15)

recipeN_M <- recipe(allTrain, vars = mVars, roles=roleV) |>
  step_range(all_predictors())

recipePCA_M <- recipe(allTrain, vars = mVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_pca(all_predictors(), num_comp=15)

recipekPCA_M <- recipe(allTrain, vars = mVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_kpca(all_predictors(), num_comp=15)

recipeICA_M <- recipe(allTrain, vars = mVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_ica(all_predictors(), num_comp=15)

recipeN_MU <- recipe(allTrain, vars = muVars, roles=roleV) |>
  step_range(all_predictors())

recipePCA_MU <- recipe(allTrain, vars = muVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_pca(all_predictors(), num_comp=15)

recipekPCA_MU <- recipe(allTrain, vars = muVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_kpca(all_predictors(), num_comp=15)

recipeICA_MU <- recipe(allTrain, vars = muVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_ica(all_predictors(), num_comp=15)

recipeN_S <- recipe(allTrain, vars = sVars, roles=roleV) |>
  step_range(all_predictors())

recipePCA_S <- recipe(allTrain, vars = sVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_pca(all_predictors(), num_comp=15)

recipekPCA_S <- recipe(allTrain, vars = sVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_kpca(all_predictors(), num_comp=15)

recipeICA_S <- recipe(allTrain, vars = sVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_ica(all_predictors(), num_comp=15)

recipeN_SU <- recipe(allTrain, vars = suVars, roles=roleV) |>
  step_range(all_predictors())

recipePCA_SU <- recipe(allTrain, vars = suVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_pca(all_predictors(), num_comp=15)

recipekPCA_SU <- recipe(allTrain, vars = suVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_kpca(all_predictors(), num_comp=15)

recipeICA_SU <- recipe(allTrain, vars = suVars, roles=roleV) |>
  step_range(all_predictors()) |>
  step_ica(all_predictors(), num_comp=15)

#Define models
rfM <- rand_forest(mtry=tune(),
                   min_n=tune(),
                   trees=500) |>
  set_engine("ranger") |>
  set_mode("classification")


svmM  <- svm_rbf(cost = tune(),
                 rbf_sigma = tune()) |>
  set_engine("kernlab") |>
  set_mode("classification")

#Define Workflowset
myWFS <-
  workflow_set(
    preproc = list(unet_norm = recipeN_U,
                   unet_pca = recipePCA_U,
                   unet_kpca = recipekPCA_U,
                   unet_ica = recipeICA_U,
                   mamba_norm = recipeN_M,
                   mamba_pca = recipePCA_M,
                   mamba_kpca = recipekPCA_M,
                   mamba_ica = recipeICA_M,
                   mambau_norm = recipeN_MU,
                   mambau_pca = recipePCA_MU,
                   mambau_kpca = recipekPCA_MU,
                   mambau_ica = recipeICA_MU,
                   swin_norm = recipeN_S,
                   swin_pca = recipePCA_S,
                   swin_kpca = recipekPCA_S,
                   swin_ica = recipeICA_S,
                   swinu_norm = recipeN_SU,
                   swinu_pca = recipePCA_SU,
                   swinu_kpca = recipekPCA_SU,
                   swinu_ica = recipeICA_SU),
    models = list(RF = rfM,
                  SVM = svmM)
  )

#Define metrics
myMetrics <- metric_set(f_meas)

#Tune Hyperparameters
future::plan(future::multisession, workers = min(10, parallel::detectCores() - 1))

grid_ctrl <-
  control_grid(
    verbose=TRUE,
    save_pred = TRUE,
    parallel_over = "resamples",
    save_workflow = TRUE
  )

grid_results <-
  myWFS |>
  workflow_map(
    seed = 42,
    resamples = folds,
    grid = 25,
    control = grid_ctrl,
    metrics=myMetrics
  )

future::plan(future::sequential)

saveRDS(grid_results, paste0(RDS_PATH, "savedWorkflow.rds"))
saveRDS(myWFS, paste0(RDS_PATH, "savedWorkflowPre.rds"))

autoplot(
  grid_results,
  rank_metric = "f_meas",
  metric = "f_meas",
  select_best = TRUE)+
  geom_text(aes(y = mean - .01,
                label = wflow_id),
            angle = 90,
            hjust = 1) +
  lims(y=c(0.6, 1.0))+
  theme(legend.position = "none")

# ==================================== ACCURACY ASSESSMENT (AVERAGED) ====================================

#Obtain assessment metrics
assessFunc <- function(test = allTest, fs = "UNet", recipe="norm", alg="rf"){
  mResults <- grid_results |>
    extract_workflow_set_result(str_glue('{fs}_{recipe}_{alg}')) |>
    select_best(metric="f_meas")
  
  set.seed(42)
  mModel <-
    grid_results |>
    extract_workflow(id=str_glue('{fs}_{recipe}_{alg}')) |>
    finalize_workflow(mResults) |>
    fit(data = allTrain)
  
  testPred <- mModel |>
    predict(test) |>
    bind_cols(test)
  
  saveRDS(testPred |> select(label, .pred_class),
          str_glue("{OUT_DIR}preds/{fs}_{recipe}_{alg}.rds"))
  
  metCI <- miceCI(reps=1000,
                  lowPercentile=0.025,
                  highPercentile=0.975,
                  reference=testPred$label,
                  prediction=testPred$.pred_class) |>
    mutate(fs = fs, recipe = recipe, alg = alg)
  
  return(metCI)
}

unet_n_rf <- assessFunc(test = allTest, fs = "unet", recipe="norm", alg="RF")
unet_pca_rf  <- assessFunc(test = allTest, fs = "unet", recipe="pca", alg="RF")
unet_kpca_rf  <- assessFunc(test = allTest, fs = "unet", recipe="kpca", alg="RF")
unet_ica_rf  <- assessFunc(test = allTest, fs = "unet", recipe="ica", alg="RF")
unet_n_svm  <- assessFunc(test = allTest, fs = "unet", recipe="norm", alg="SVM")
unet_pca_svm  <- assessFunc(test = allTest, fs = "unet", recipe="pca", alg="SVM")
unet_kpca_svm  <- assessFunc(test = allTest, fs = "unet", recipe="kpca", alg="SVM")
unet_ica_svm  <- assessFunc(test = allTest, fs = "unet", recipe="ica", alg="SVM")

mamba_n_rf <- assessFunc(test = allTest, fs = "mamba", recipe="norm", alg="RF")
mamba_pca_rf  <- assessFunc(test = allTest, fs = "mamba", recipe="pca", alg="RF")
mamba_kpca_rf  <- assessFunc(test = allTest, fs = "mamba", recipe="kpca", alg="RF")
mamba_ica_rf  <- assessFunc(test = allTest, fs = "mamba", recipe="ica", alg="RF")
mamba_n_svm  <- assessFunc(test = allTest, fs = "mamba", recipe="norm", alg="SVM")
mamba_pca_svm  <- assessFunc(test = allTest, fs = "mamba", recipe="pca", alg="SVM")
mamba_kpca_svm  <- assessFunc(test = allTest, fs = "mamba", recipe="kpca", alg="SVM")
mamba_ica_svm  <- assessFunc(test = allTest, fs = "mamba", recipe="ica", alg="SVM")

mambau_n_rf <- assessFunc(test = allTest, fs = "mambau", recipe="norm", alg="RF")
mambau_pca_rf  <- assessFunc(test = allTest, fs = "mambau", recipe="pca", alg="RF")
mambau_kpca_rf  <- assessFunc(test = allTest, fs = "mambau", recipe="kpca", alg="RF")
mambau_ica_rf  <- assessFunc(test = allTest, fs = "mambau", recipe="ica", alg="RF")
mambau_n_svm  <- assessFunc(test = allTest, fs = "mambau", recipe="norm", alg="SVM")
mambau_pca_svm  <- assessFunc(test = allTest, fs = "mambau", recipe="pca", alg="SVM")
mambau_kpca_svm  <- assessFunc(test = allTest, fs = "mambau", recipe="kpca", alg="SVM")
mambau_ica_svm  <- assessFunc(test = allTest, fs = "mambau", recipe="ica", alg="SVM")

swin_n_rf <- assessFunc(test = allTest, fs = "swin", recipe="norm", alg="RF")
swin_pca_rf  <- assessFunc(test = allTest, fs = "swin", recipe="pca", alg="RF")
swin_kpca_rf  <- assessFunc(test = allTest, fs = "swin", recipe="kpca", alg="RF")
swin_ica_rf  <- assessFunc(test = allTest, fs = "swin", recipe="ica", alg="RF")
swin_n_svm  <- assessFunc(test = allTest, fs = "swin", recipe="norm", alg="SVM")
swin_pca_svm  <- assessFunc(test = allTest, fs = "swin", recipe="pca", alg="SVM")
swin_kpca_svm  <- assessFunc(test = allTest, fs = "swin", recipe="kpca", alg="SVM")
swin_ica_svm  <- assessFunc(test = allTest, fs = "swin", recipe="ica", alg="SVM")

swinu_n_rf <- assessFunc(test = allTest, fs = "swinu", recipe="norm", alg="RF")
swinu_pca_rf  <- assessFunc(test = allTest, fs = "swinu", recipe="pca", alg="RF")
swinu_kpca_rf  <- assessFunc(test = allTest, fs = "swinu", recipe="kpca", alg="RF")
swinu_ica_rf  <- assessFunc(test = allTest, fs = "swinu", recipe="ica", alg="RF")
swinu_n_svm  <- assessFunc(test = allTest, fs = "swinu", recipe="norm", alg="SVM")
swinu_pca_svm  <- assessFunc(test = allTest, fs = "swinu", recipe="pca", alg="SVM")
swinu_kpca_svm  <- assessFunc(test = allTest, fs = "swinu", recipe="kpca", alg="SVM")
swinu_ica_svm  <- assessFunc(test = allTest, fs = "swinu", recipe="ica", alg="SVM")

allMets <- bind_rows(unet_n_rf,
                     unet_pca_rf,
                     unet_kpca_rf,
                     unet_ica_rf,
                     unet_n_svm,
                     unet_pca_svm,
                     unet_kpca_svm,
                     unet_ica_svm,
                     mamba_n_rf,
                     mamba_pca_rf,
                     mamba_kpca_rf,
                     mamba_ica_rf,
                     mamba_n_svm,
                     mamba_pca_svm,
                     mamba_kpca_svm,
                     mamba_ica_svm,
                     mambau_n_rf,
                     mambau_pca_rf,
                     mambau_kpca_rf,
                     mambau_ica_rf,
                     mambau_n_svm,
                     mambau_pca_svm,
                     mambau_kpca_svm,
                     mambau_ica_svm,
                     swin_n_rf,
                     swin_pca_rf,
                     swin_kpca_rf,
                     swin_ica_rf,
                     swin_n_svm,
                     swin_pca_svm,
                     swin_kpca_svm,
                     swin_ica_svm,
                     swinu_n_rf,
                     swinu_pca_rf,
                     swinu_kpca_rf,
                     swinu_ica_rf,
                     swinu_n_svm,
                     swinu_pca_svm,
                     swinu_kpca_svm,
                     swinu_ica_svm)

write.csv(allMets, paste0(OUT_DIR, "metResultsRev.csv"))

metsToUse <- c("overallAccuracy", "MICE", "macroF1", "macroPA", "macroUA")

mets2 <- read_csv(paste0(OUT_DIR, "metResultsRev.csv"))[,2:9] |>
  mutate_if(is.character, as.factor) |>
  filter(metric %in% metsToUse)

mets2$metric <- forcats::fct_relevel(mets2$metric, "overallAccuracy", "MICE", "macroF1", "macroPA", "macroUA")
mets2$recipe <- forcats::fct_relevel(mets2$recipe, "norm", "pca", "kpca", "ica")

mets2 <- mets2 |> mutate(fs_alg = str_glue('{fs}_{alg}'))
mets2$fs_alg <- as.factor(mets2$fs_alg)

mets2$metric <- forcats::fct_recode(mets2$metric,
                                    OA = "overallAccuracy",
                                    MICE = "MICE",
                                    F1 = "macroF1",
                                    PA = "macroPA",
                                    UA = "macroUA")

mets2$recipe <- forcats::fct_recode(mets2$recipe,
                                    Original = "norm",
                                    PCA = "pca",
                                    kPCA= "kpca",
                                    ICA = "ica")

mets2$fs_alg <- forcats::fct_relevel(mets2$fs_alg, "unet_RF", "unet_SVM", "swin_RF", "swin_SVM",
                                     "swinu_RF", "swinu_SVM", "mamba_RF", "mamba_SVM",
                                     "mambau_RF", "mambau_SVM")

mets2$fs_alg <- forcats::fct_recode(mets2$fs_alg,
                                    "UNet (RF)" = "unet_RF",
                                    "UNet (SVM)" = "unet_SVM",
                                    "SwinUNet (RF)" = "swin_RF",
                                    "SwinUNet (SVM)" = "swin_SVM",
                                    "SwinUNetC (RF)" = "swinu_RF",
                                    "SwinUNetC (SVM)" = "swinu_SVM",
                                    "MambaUNet (RF)" = "mamba_RF",
                                    "MambaUNet (SVM)" = "mamba_SVM",
                                    "MambaUNetC (RF)" = "mambau_RF",
                                    "MambaUNetC (SVM)" = "mambau_SVM")

mets2 |>
  ggplot(aes(x=fs_alg, y=mean))+
  geom_errorbar(aes(ymin=low.ci, ymax=high.ci))+
  geom_point(color="salmon")+
  facet_grid(metric~recipe)+
  theme_bw(base_size=12)+
  scale_y_continuous(limits=c(0.6, 1.0), breaks=seq(0.6, 1.0, by=.1), labels = c("0.60", "0.70", "0.80", "0.90", "1.00"))+
  labs(x="Feature Space (Algorithm)", y="Metric Value")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, color="black"))+
  theme(axis.text.y = element_text(color="black"))

# ==================================== ACCURACY ASSESSMENT (PER-CLASS BOOTSTRAP) ====================================

#Per-class precision, recall, and F1 with bootstrap CIs (same settings as miceCI)
classCI <- function(reference, prediction, reps=1000, lowPercentile=0.025, highPercentile=0.975){
  reference  <- as.factor(reference)
  prediction <- factor(prediction, levels = levels(reference))
  lv <- levels(reference)
  k  <- length(lv)
  n  <- length(reference)
  r  <- as.integer(reference)
  p  <- as.integer(prediction)
  
  #Confusion matrix: rows = reference, cols = prediction
  getMets <- function(idx){
    cm <- matrix(tabulate((r[idx] - 1L) * k + p[idx], nbins = k * k),
                 nrow = k, ncol = k, byrow = TRUE)
    tp <- diag(cm)
    precision <- tp / colSums(cm)
    recall    <- tp / rowSums(cm)
    f1        <- 2 * precision * recall / (precision + recall)
    c(precision, recall, f1)
  }
  
  est  <- getMets(seq_len(n))
  boot <- vapply(seq_len(reps),
                 \(i) getMets(sample.int(n, n, replace = TRUE)),
                 numeric(3 * k))
  
  data.frame(class    = rep(lv, 3),
             n_ref    = rep(as.vector(table(reference)), 3),
             metric   = rep(c("precision", "recall", "f1"), each = k),
             estimate = est,
             mean     = rowMeans(boot, na.rm = TRUE),
             median   = apply(boot, 1, median, na.rm = TRUE),
             low.ci   = apply(boot, 1, quantile, probs = lowPercentile, na.rm = TRUE),
             high.ci  = apply(boot, 1, quantile, probs = highPercentile, na.rm = TRUE),
             row.names = NULL)
}

#Obtain per-class metrics
classFunc <- function(test = allTest, fs = "UNet", recipe="norm", alg="rf"){
  
  predFile <- str_glue("{OUT_DIR}preds/{fs}_{recipe}_{alg}.rds")
  
  testPred <- readRDS(predFile)
  
  set.seed(42)
  clsCI <- classCI(reps=1000,
                   lowPercentile=0.025,
                   highPercentile=0.975,
                   reference=testPred$label,
                   prediction=testPred$.pred_class) |>
    mutate(fs = fs, recipe = recipe, alg = alg)
  
  return(clsCI)
}

dir.create(file.path(OUT_DIR, "preds"), showWarnings = FALSE)

unet_n_rf_cls <- classFunc(test = allTest, fs = "unet", recipe="norm", alg="RF")
unet_pca_rf_cls  <- classFunc(test = allTest, fs = "unet", recipe="pca", alg="RF")
unet_kpca_rf_cls  <- classFunc(test = allTest, fs = "unet", recipe="kpca", alg="RF")
unet_ica_rf_cls  <- classFunc(test = allTest, fs = "unet", recipe="ica", alg="RF")
unet_n_svm_cls  <- classFunc(test = allTest, fs = "unet", recipe="norm", alg="SVM")
unet_pca_svm_cls  <- classFunc(test = allTest, fs = "unet", recipe="pca", alg="SVM")
unet_kpca_svm_cls  <- classFunc(test = allTest, fs = "unet", recipe="kpca", alg="SVM")
unet_ica_svm_cls  <- classFunc(test = allTest, fs = "unet", recipe="ica", alg="SVM")

mamba_n_rf_cls <- classFunc(test = allTest, fs = "mamba", recipe="norm", alg="RF")
mamba_pca_rf_cls  <- classFunc(test = allTest, fs = "mamba", recipe="pca", alg="RF")
mamba_kpca_rf_cls  <- classFunc(test = allTest, fs = "mamba", recipe="kpca", alg="RF")
mamba_ica_rf_cls  <- classFunc(test = allTest, fs = "mamba", recipe="ica", alg="RF")
mamba_n_svm_cls  <- classFunc(test = allTest, fs = "mamba", recipe="norm", alg="SVM")
mamba_pca_svm_cls  <- classFunc(test = allTest, fs = "mamba", recipe="pca", alg="SVM")
mamba_kpca_svm_cls  <- classFunc(test = allTest, fs = "mamba", recipe="kpca", alg="SVM")
mamba_ica_svm_cls  <- classFunc(test = allTest, fs = "mamba", recipe="ica", alg="SVM")

mambau_n_rf_cls <- classFunc(test = allTest, fs = "mambau", recipe="norm", alg="RF")
mambau_pca_rf_cls  <- classFunc(test = allTest, fs = "mambau", recipe="pca", alg="RF")
mambau_kpca_rf_cls  <- classFunc(test = allTest, fs = "mambau", recipe="kpca", alg="RF")
mambau_ica_rf_cls  <- classFunc(test = allTest, fs = "mambau", recipe="ica", alg="RF")
mambau_n_svm_cls  <- classFunc(test = allTest, fs = "mambau", recipe="norm", alg="SVM")
mambau_pca_svm_cls  <- classFunc(test = allTest, fs = "mambau", recipe="pca", alg="SVM")
mambau_kpca_svm_cls  <- classFunc(test = allTest, fs = "mambau", recipe="kpca", alg="SVM")
mambau_ica_svm_cls  <- classFunc(test = allTest, fs = "mambau", recipe="ica", alg="SVM")

swin_n_rf_cls <- classFunc(test = allTest, fs = "swin", recipe="norm", alg="RF")
swin_pca_rf_cls  <- classFunc(test = allTest, fs = "swin", recipe="pca", alg="RF")
swin_kpca_rf_cls  <- classFunc(test = allTest, fs = "swin", recipe="kpca", alg="RF")
swin_ica_rf_cls  <- classFunc(test = allTest, fs = "swin", recipe="ica", alg="RF")
swin_n_svm_cls  <- classFunc(test = allTest, fs = "swin", recipe="norm", alg="SVM")
swin_pca_svm_cls  <- classFunc(test = allTest, fs = "swin", recipe="pca", alg="SVM")
swin_kpca_svm_cls  <- classFunc(test = allTest, fs = "swin", recipe="kpca", alg="SVM")
swin_ica_svm_cls  <- classFunc(test = allTest, fs = "swin", recipe="ica", alg="SVM")

swinu_n_rf_cls <- classFunc(test = allTest, fs = "swinu", recipe="norm", alg="RF")
swinu_pca_rf_cls  <- classFunc(test = allTest, fs = "swinu", recipe="pca", alg="RF")
swinu_kpca_rf_cls  <- classFunc(test = allTest, fs = "swinu", recipe="kpca", alg="RF")
swinu_ica_rf_cls  <- classFunc(test = allTest, fs = "swinu", recipe="ica", alg="RF")
swinu_n_svm_cls  <- classFunc(test = allTest, fs = "swinu", recipe="norm", alg="SVM")
swinu_pca_svm_cls  <- classFunc(test = allTest, fs = "swinu", recipe="pca", alg="SVM")
swinu_kpca_svm_cls  <- classFunc(test = allTest, fs = "swinu", recipe="kpca", alg="SVM")
swinu_ica_svm_cls  <- classFunc(test = allTest, fs = "swinu", recipe="ica", alg="SVM")

allClassMets <- bind_rows(unet_n_rf_cls,
                          unet_pca_rf_cls,
                          unet_kpca_rf_cls,
                          unet_ica_rf_cls,
                          unet_n_svm_cls,
                          unet_pca_svm_cls,
                          unet_kpca_svm_cls,
                          unet_ica_svm_cls,
                          mamba_n_rf_cls,
                          mamba_pca_rf_cls,
                          mamba_kpca_rf_cls,
                          mamba_ica_rf_cls,
                          mamba_n_svm_cls,
                          mamba_pca_svm_cls,
                          mamba_kpca_svm_cls,
                          mamba_ica_svm_cls,
                          mambau_n_rf_cls,
                          mambau_pca_rf_cls,
                          mambau_kpca_rf_cls,
                          mambau_ica_rf_cls,
                          mambau_n_svm_cls,
                          mambau_pca_svm_cls,
                          mambau_kpca_svm_cls,
                          mambau_ica_svm_cls,
                          swin_n_rf_cls,
                          swin_pca_rf_cls,
                          swin_kpca_rf_cls,
                          swin_ica_rf_cls,
                          swin_n_svm_cls,
                          swin_pca_svm_cls,
                          swin_kpca_svm_cls,
                          swin_ica_svm_cls,
                          swinu_n_rf_cls,
                          swinu_pca_rf_cls,
                          swinu_kpca_rf_cls,
                          swinu_ica_rf_cls,
                          swinu_n_svm_cls,
                          swinu_pca_svm_cls,
                          swinu_kpca_svm_cls,
                          swinu_ica_svm_cls)

write.csv(allClassMets, paste0(OUT_DIR, "classMetResultsRev.csv"))

classMets2 <- read_csv(paste0(OUT_DIR, "classMetResultsRev.csv"))[,2:12] |>
  mutate_if(is.character, as.factor) |>
  mutate(class = as.factor(class))

classMets2$metric <- forcats::fct_relevel(classMets2$metric, "f1", "precision", "recall")
classMets2$recipe <- forcats::fct_relevel(classMets2$recipe, "norm", "pca", "kpca", "ica")

classMets2 <- classMets2 |> mutate(fs_alg = str_glue('{fs}_{alg}'))
classMets2$fs_alg <- as.factor(classMets2$fs_alg)

classMets2$metric <- forcats::fct_recode(classMets2$metric,
                                         F1 = "f1",
                                         Precision = "precision",
                                         Recall = "recall")

classMets2$recipe <- forcats::fct_recode(classMets2$recipe,
                                         Original = "norm",
                                         PCA = "pca",
                                         kPCA= "kpca",
                                         ICA = "ica")

classMets2$fs_alg <- forcats::fct_relevel(classMets2$fs_alg, "unet_RF", "unet_SVM", "swin_RF", "swin_SVM",
                                          "swinu_RF", "swinu_SVM", "mamba_RF", "mamba_SVM",
                                          "mambau_RF", "mambau_SVM")

classMets2$fs_alg <- forcats::fct_recode(classMets2$fs_alg,
                                         "UNet (RF)" = "unet_RF",
                                         "UNet (SVM)" = "unet_SVM",
                                         "SwinUNet (RF)" = "swin_RF",
                                         "SwinUNet (SVM)" = "swin_SVM",
                                         "SwinUNetC (RF)" = "swinu_RF",
                                         "SwinUNetC (SVM)" = "swinu_SVM",
                                         "MambaUNet (RF)" = "mamba_RF",
                                         "MambaUNet (SVM)" = "mamba_SVM",
                                         "MambaUNetC (RF)" = "mambau_RF",
                                         "MambaUNetC (SVM)" = "mambau_SVM")

classMets2 |>
  ggplot(aes(x=fs_alg, y=mean, color=class))+
  geom_errorbar(aes(ymin=low.ci, ymax=high.ci), width=0.4, position=position_dodge(width=0.6))+
  geom_point(position=position_dodge(width=0.6))+
  facet_grid(metric~recipe)+
  theme_bw(base_size=12)+
  labs(x="Feature Space (Algorithm)", y="Metric Value", color="Class")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, color="black"))+
  theme(axis.text.y = element_text(color="black"))

# ==================================== ACCURACY ASSESSMENT (PER-CLASS) ====================================

#Per-class precision, recall, and F1
classMets <- function(reference, prediction){
  reference  <- as.factor(reference)
  prediction <- factor(prediction, levels = levels(reference))
  cm <- table(reference, prediction)   #rows = reference, cols = prediction
  tp <- diag(cm)
  precision <- tp / colSums(cm)
  recall    <- tp / rowSums(cm)
  f1        <- 2 * precision * recall / (precision + recall)
  data.frame(class     = levels(reference),
             n_ref     = as.vector(rowSums(cm)),
             precision = as.vector(precision),
             recall    = as.vector(recall),
             f1        = as.vector(f1),
             row.names = NULL)
}

#Obtain per-class metrics (uses saved predictions if they exist, otherwise refits and saves them)
classFunc <- function(test = allTest, fs = "UNet", recipe="norm", alg="rf"){
  predFile <- str_glue("{OUT_DIR}preds/{fs}_{recipe}_{alg}.rds")
  
  testPred <- readRDS(predFile)
  clsMets <- classMets(reference=testPred$label,
                       prediction=testPred$.pred_class) |>
    mutate(fs = fs, recipe = recipe, alg = alg)

  return(clsMets)
}

unet_n_rf_cls <- classFunc(test = allTest, fs = "unet", recipe="norm", alg="RF")
unet_pca_rf_cls  <- classFunc(test = allTest, fs = "unet", recipe="pca", alg="RF")
unet_kpca_rf_cls  <- classFunc(test = allTest, fs = "unet", recipe="kpca", alg="RF")
unet_ica_rf_cls  <- classFunc(test = allTest, fs = "unet", recipe="ica", alg="RF")
unet_n_svm_cls  <- classFunc(test = allTest, fs = "unet", recipe="norm", alg="SVM")
unet_pca_svm_cls  <- classFunc(test = allTest, fs = "unet", recipe="pca", alg="SVM")
unet_kpca_svm_cls  <- classFunc(test = allTest, fs = "unet", recipe="kpca", alg="SVM")
unet_ica_svm_cls  <- classFunc(test = allTest, fs = "unet", recipe="ica", alg="SVM")

mamba_n_rf_cls <- classFunc(test = allTest, fs = "mamba", recipe="norm", alg="RF")
mamba_pca_rf_cls  <- classFunc(test = allTest, fs = "mamba", recipe="pca", alg="RF")
mamba_kpca_rf_cls  <- classFunc(test = allTest, fs = "mamba", recipe="kpca", alg="RF")
mamba_ica_rf_cls  <- classFunc(test = allTest, fs = "mamba", recipe="ica", alg="RF")
mamba_n_svm_cls  <- classFunc(test = allTest, fs = "mamba", recipe="norm", alg="SVM")
mamba_pca_svm_cls  <- classFunc(test = allTest, fs = "mamba", recipe="pca", alg="SVM")
mamba_kpca_svm_cls  <- classFunc(test = allTest, fs = "mamba", recipe="kpca", alg="SVM")
mamba_ica_svm_cls  <- classFunc(test = allTest, fs = "mamba", recipe="ica", alg="SVM")

mambau_n_rf_cls <- classFunc(test = allTest, fs = "mambau", recipe="norm", alg="RF")
mambau_pca_rf_cls  <- classFunc(test = allTest, fs = "mambau", recipe="pca", alg="RF")
mambau_kpca_rf_cls  <- classFunc(test = allTest, fs = "mambau", recipe="kpca", alg="RF")
mambau_ica_rf_cls  <- classFunc(test = allTest, fs = "mambau", recipe="ica", alg="RF")
mambau_n_svm_cls  <- classFunc(test = allTest, fs = "mambau", recipe="norm", alg="SVM")
mambau_pca_svm_cls  <- classFunc(test = allTest, fs = "mambau", recipe="pca", alg="SVM")
mambau_kpca_svm_cls  <- classFunc(test = allTest, fs = "mambau", recipe="kpca", alg="SVM")
mambau_ica_svm_cls  <- classFunc(test = allTest, fs = "mambau", recipe="ica", alg="SVM")

swin_n_rf_cls <- classFunc(test = allTest, fs = "swin", recipe="norm", alg="RF")
swin_pca_rf_cls  <- classFunc(test = allTest, fs = "swin", recipe="pca", alg="RF")
swin_kpca_rf_cls  <- classFunc(test = allTest, fs = "swin", recipe="kpca", alg="RF")
swin_ica_rf_cls  <- classFunc(test = allTest, fs = "swin", recipe="ica", alg="RF")
swin_n_svm_cls  <- classFunc(test = allTest, fs = "swin", recipe="norm", alg="SVM")
swin_pca_svm_cls  <- classFunc(test = allTest, fs = "swin", recipe="pca", alg="SVM")
swin_kpca_svm_cls  <- classFunc(test = allTest, fs = "swin", recipe="kpca", alg="SVM")
swin_ica_svm_cls  <- classFunc(test = allTest, fs = "swin", recipe="ica", alg="SVM")

swinu_n_rf_cls <- classFunc(test = allTest, fs = "swinu", recipe="norm", alg="RF")
swinu_pca_rf_cls  <- classFunc(test = allTest, fs = "swinu", recipe="pca", alg="RF")
swinu_kpca_rf_cls  <- classFunc(test = allTest, fs = "swinu", recipe="kpca", alg="RF")
swinu_ica_rf_cls  <- classFunc(test = allTest, fs = "swinu", recipe="ica", alg="RF")
swinu_n_svm_cls  <- classFunc(test = allTest, fs = "swinu", recipe="norm", alg="SVM")
swinu_pca_svm_cls  <- classFunc(test = allTest, fs = "swinu", recipe="pca", alg="SVM")
swinu_kpca_svm_cls  <- classFunc(test = allTest, fs = "swinu", recipe="kpca", alg="SVM")
swinu_ica_svm_cls  <- classFunc(test = allTest, fs = "swinu", recipe="ica", alg="SVM")

allClassMets <- bind_rows(unet_n_rf_cls,
                          unet_pca_rf_cls,
                          unet_kpca_rf_cls,
                          unet_ica_rf_cls,
                          unet_n_svm_cls,
                          unet_pca_svm_cls,
                          unet_kpca_svm_cls,
                          unet_ica_svm_cls,
                          mamba_n_rf_cls,
                          mamba_pca_rf_cls,
                          mamba_kpca_rf_cls,
                          mamba_ica_rf_cls,
                          mamba_n_svm_cls,
                          mamba_pca_svm_cls,
                          mamba_kpca_svm_cls,
                          mamba_ica_svm_cls,
                          mambau_n_rf_cls,
                          mambau_pca_rf_cls,
                          mambau_kpca_rf_cls,
                          mambau_ica_rf_cls,
                          mambau_n_svm_cls,
                          mambau_pca_svm_cls,
                          mambau_kpca_svm_cls,
                          mambau_ica_svm_cls,
                          swin_n_rf_cls,
                          swin_pca_rf_cls,
                          swin_kpca_rf_cls,
                          swin_ica_rf_cls,
                          swin_n_svm_cls,
                          swin_pca_svm_cls,
                          swin_kpca_svm_cls,
                          swin_ica_svm_cls,
                          swinu_n_rf_cls,
                          swinu_pca_rf_cls,
                          swinu_kpca_rf_cls,
                          swinu_ica_rf_cls,
                          swinu_n_svm_cls,
                          swinu_pca_svm_cls,
                          swinu_kpca_svm_cls,
                          swinu_ica_svm_cls)

write.csv(allClassMets, paste0(OUT_DIR, "classMetResultsRev.csv"))

#Number of test samples per class (for the table caption)
allClassMets |> distinct(class, n_ref)

#Table: one row per recipe / feature space / algorithm; Precision, Recall, F1 for each class
classTable <- allClassMets |>
  mutate(fs = factor(fs,
                     levels = c("unet", "swin", "swinu", "mamba", "mambau"),
                     labels = c("UNet", "SwinUNet", "SwinUNetC", "MambaUNet", "MambaUNetC")),
         recipe = factor(recipe,
                         levels = c("norm", "pca", "kpca", "ica"),
                         labels = c("Original", "PCA", "kPCA", "ICA")),
         across(c(precision, recall, f1), \(x) round(x, 3))) |>
  select(-n_ref) |>
  pivot_wider(names_from = class,
              values_from = c(precision, recall, f1),
              names_vary = "slowest",
              names_glue = "{class}_{.value}") |>
  arrange(recipe, fs, alg) |>
  rename(Recipe = recipe, `Feature Space` = fs, Algorithm = alg) |>
  relocate(Recipe, `Feature Space`, Algorithm)

print(classTable, n = Inf, width = Inf)

write.csv(classTable, paste0(OUT_DIR, "classMetTableRev.csv", row.names = FALSE))
# ==================================== PCA ANALYSIS ====================================

#Load all tables
set.seed(42)
unetT <- read_csv(paste0(INPUT_DIR,"UNet_Train.csv"))[,c(4, 9:104)] |> group_by(label) |> sample_n(1000, replace=FALSE) |> ungroup() |> select(-label)
mamT <- read_csv(paste0(INPUT_DIR,"MambaUNet_Train.csv"))[,c(4, 9:104)] |> group_by(label) |> sample_n(1000, replace=FALSE) |> ungroup() |> select(-label)
mamuT <- read_csv(paste0(INPUT_DIR,"MambaUNetC_Train.csv"))[,c(4,9:104)]  |> group_by(label) |> sample_n(1000, replace=FALSE) |> ungroup() |> select(-label)
swinT <- read_csv(paste0(INPUT_DIR,"SwinUNet_Train.csv"))[,c(4, 9:104)]  |> group_by(label) |> sample_n(1000, replace=FALSE) |> ungroup() |> select(-label)
swinuT <- read_csv(paste0(INPUT_DIR,"SwinUNetC_Train.csv"))[,c(4, 9:104)]  |> group_by(label) |> sample_n(1000, replace=FALSE) |> ungroup() |> select(-label)

pcaUnet <- prcomp(unetT, scale = TRUE)
pcaMam <- prcomp(mamT, scale = TRUE)
pcaMamu <- prcomp(mamuT, scale = TRUE)
pcaSwin <- prcomp(swinT, scale = TRUE)
pcaSwinu <- prcomp(swinuT, scale = TRUE)

#Merge all training tables
pPcaUnet <- fviz_screeplot(pcaUnet, addlabels = FALSE, ncp=15)+
  geom_text(aes(label = paste0(round(eig, 1), "%")),
            size = 2.5, angle = 45, hjust = -0.1, vjust = 0.5, nudge_y = 0.5) +
  theme_bw(base_size=12)+
  scale_y_continuous(expand = c(0, 0), limits=c(0, 43), breaks=seq(0, 45, by = 5), labels=scales::label_percent(scale=1, accuracy=.1))+
  scale_x_discrete(expand = expansion(add = c(0.4, 1))) +
  labs(x="PC", y="Percent of Variance Explained")+
  ggtitle("(a) UNet")+
  theme(axis.text.x = element_text(color="black"))+
  theme(axis.text.y = element_text(color="black"))

pPcaMam <- fviz_screeplot(pcaMam, addlabels = FALSE, ncp=15)+
  geom_text(aes(label = paste0(round(eig, 1), "%")),
            size = 2.5, angle = 45, hjust = -0.1, vjust = 0.5, nudge_y = 0.5) +
  theme_bw(base_size=12)+
  scale_y_continuous(expand = c(0, 0), limits=c(0, 43), breaks=seq(0, 45, by = 5), labels=scales::label_percent(scale=1, accuracy=.1))+
  scale_x_discrete(expand = expansion(add = c(0.4, 1))) +
  labs(x="PC", y="Percent of Variance Explained")+
  ggtitle("(d) MambaUNet")+
  theme(axis.text.x = element_text(color="black"))+
  theme(axis.text.y = element_text(color="black"))

pPcaMamU <- fviz_screeplot(pcaMamu, addlabels = FALSE, ncp=15)+
  geom_text(aes(label = paste0(round(eig, 1), "%")),
            size = 2.5, angle = 45, hjust = -0.1, vjust = 0.5, nudge_y = 0.5) +
  theme_bw(base_size=12)+
  scale_y_continuous(expand = c(0, 0), limits=c(0, 43), breaks=seq(0, 45, by = 5), labels=scales::label_percent(scale=1, accuracy=.1))+
  scale_x_discrete(expand = expansion(add = c(0.4, 1))) +
  labs(x="PC", y="Percent of Variance Explained")+
  ggtitle("(e) MambaUNetC")+
  theme(axis.text.x = element_text(color="black"))+
  theme(axis.text.y = element_text(color="black"))

pPcaSwin <-fviz_screeplot(pcaSwin, addlabels = FALSE, ncp=15)+
  geom_text(aes(label = paste0(round(eig, 1), "%")),
            size = 2.5, angle = 45, hjust = -0.1, vjust = 0.5, nudge_y = 0.5) +
  theme_bw(base_size=12)+
  scale_y_continuous(expand = c(0, 0), limits=c(0, 43), breaks=seq(0, 45, by = 5), labels=scales::label_percent(scale=1, accuracy=.1))+
  scale_x_discrete(expand = expansion(add = c(0.4, 1))) +
  labs(x="PC", y="Percent of Variance Explained")+
  ggtitle("(b) SwinUNet")+
  theme(axis.text.x = element_text(color="black"))+
  theme(axis.text.y = element_text(color="black"))

pPcaSwinU <- fviz_screeplot(pcaSwinu, addlabels = FALSE, ncp=15)+
  geom_text(aes(label = paste0(round(eig, 1), "%")),
            size = 2.5, angle = 45, hjust = -0.1, vjust = 0.5, nudge_y = 0.5) +
  theme_bw(base_size=12)+
  scale_y_continuous(expand = c(0, 0), limits=c(0, 43), breaks=seq(0, 45, by = 5), labels=scales::label_percent(scale=1, accuracy=.1))+
  scale_x_discrete(expand = expansion(add = c(0.45, 1))) +
  labs(x="PC", y="Percent of Variance Explained")+
  ggtitle("(c) SwinUNetC")+
  theme(axis.text.x = element_text(color="black"))+
  theme(axis.text.y = element_text(color="black"))

plot_grid(pPcaUnet, pPcaSwin, pPcaSwinU, pPcaMam, pPcaMamU, nrows=3)

# ==================================== KPCA ANALYSIS ====================================
library(kernlab)

doKPCA <- function(data){
  kpc <- kpca(~., data = data, kernel = "rbfdot")
  eigs <- eig(kpc)
  pca_like <- list()
  pca_like$sdev <- sqrt(eigs)
  pca_like$center <- FALSE
  pca_like$scale <- FALSE
  class(pca_like) <- "prcomp"
  return(pca_like)}

pcaUnet <- doKPCA(unetT)
pcaMam <- doKPCA(mamT)
pcaMamu <- doKPCA(mamuT)
pcaSwin <- doKPCA(swinT)
pcaSwinu <- doKPCA(swinuT)

pPcaUnet <- fviz_screeplot(pcaUnet, addlabels = FALSE, ncp=15)+
  geom_text(aes(label = paste0(round(eig, 1), "%")),
            size = 2.5, angle = 45, hjust = -0.1, vjust = 0.5, nudge_y = 0.5) +
  theme_bw(base_size=12)+
  scale_y_continuous(expand = c(0, 0), limits=c(0, 32), breaks=seq(0, 45, by = 5), labels=scales::label_percent(scale=1, accuracy=.1))+
  scale_x_discrete(expand = expansion(add = c(0.45, 1))) +
  labs(x="kPC", y="Percent of Variance Explained")+
  ggtitle("(a) UNet")+
  theme(axis.text.x = element_text(color="black"))+
  theme(axis.text.y = element_text(color="black"))

pPcaMam <- fviz_screeplot(pcaMam, addlabels = FALSE, ncp=15)+
  geom_text(aes(label = paste0(round(eig, 1), "%")),
            size = 2.5, angle = 45, hjust = -0.1, vjust = 0.5, nudge_y = 0.5) +
  theme_bw(base_size=12)+
  scale_y_continuous(expand = c(0, 0), limits=c(0, 32), breaks=seq(0, 45, by = 5), labels=scales::label_percent(scale=1, accuracy=.1))+
  scale_x_discrete(expand = expansion(add = c(0.45, 1))) +
  labs(x="kPC", y="Percent of Variance Explained")+
  ggtitle("(d) MambaUNet")+
  theme(axis.text.x = element_text(color="black"))+
  theme(axis.text.y = element_text(color="black"))

pPcaMamU <- fviz_screeplot(pcaMamu, addlabels = FALSE, ncp=15)+
  geom_text(aes(label = paste0(round(eig, 1), "%")),
            size = 2.5, angle = 45, hjust = -0.1, vjust = 0.5, nudge_y = 0.5) +
  theme_bw(base_size=12)+
  scale_y_continuous(expand = c(0, 0), limits=c(0, 32), breaks=seq(0, 45, by = 5), labels=scales::label_percent(scale=1, accuracy=.1))+
  scale_x_discrete(expand = expansion(add = c(0.45, 1))) +
  labs(x="kPC", y="Percent of Variance Explained")+
  ggtitle("(e) MambaUNetC")+
  theme(axis.text.x = element_text(color="black"))+
  theme(axis.text.y = element_text(color="black"))

pPcaSwin <-fviz_screeplot(pcaSwin, addlabels = FALSE, ncp=15)+
  geom_text(aes(label = paste0(round(eig, 1), "%")),
            size = 2.5, angle = 45, hjust = -0.1, vjust = 0.5, nudge_y = 0.5) +
  theme_bw(base_size=12)+
  scale_y_continuous(expand = c(0, 0), limits=c(0, 32), breaks=seq(0, 45, by = 5), labels=scales::label_percent(scale=1, accuracy=.1))+
  scale_x_discrete(expand = expansion(add = c(0.45, 1))) +
  labs(x="kPC", y="Percent of Variance Explained")+
  ggtitle("(b) SwinUNet")+
  theme(axis.text.x = element_text(color="black"))+
  theme(axis.text.y = element_text(color="black"))

pPcaSwinU <- fviz_screeplot(pcaSwinu, addlabels = FALSE, ncp=15)+
  geom_text(aes(label = paste0(round(eig, 1), "%")),
            size = 2.5, angle = 45, hjust = -0.1, vjust = 0.5, nudge_y = 0.5) +
  theme_bw(base_size=12)+
  scale_y_continuous(expand = c(0, 0), limits=c(0, 32), breaks=seq(0, 45, by = 5), labels=scales::label_percent(scale=1, accuracy=.1))+
  scale_x_discrete(expand = expansion(add = c(0.45, 1))) +
  labs(x="kPC", y="Percent of Variance Explained")+
  ggtitle("(c) SwinUNetC")+
  theme(axis.text.x = element_text(color="black"))+
  theme(axis.text.y = element_text(color="black"))

plot_grid(pPcaUnet, pPcaSwin, pPcaSwinU, pPcaMam, pPcaMamU, nrows=3)

# ==================================== MARGINAL VARIABLE IMPORTANCE ====================================

set.seed(42)
unetT <- read_csv(paste0(INPUT_DIR,"UNet_Train.csv"))[,c(4, 9:104)] |> mutate(label = as.factor(label))
unetV <- read_csv(paste0(INPUT_DIR,"UNet_Test.csv"))[,c(4, 9:104)] |> mutate(label = as.factor(label))
mamT <- read_csv(paste0(INPUT_DIR,"MambaUNet_Train.csv"))[,c(9:104)]
mamV <- read_csv(paste0(INPUT_DIR,"MambaUNet_Test.csv"))[,c(9:104)]
mamuT <- read_csv(paste0(INPUT_DIR,"MambaUNetC_Train.csv"))[,c(9:104)]
mamuV <- read_csv(paste0(INPUT_DIR,"MambaUNetC_Test.csv"))[,c(9:104)]
swinT <- read_csv(paste0(INPUT_DIR,"SwinUNet_Train.csv"))[,c(9:104)]
swinV <- read_csv(paste0(INPUT_DIR,"SwinUNet_Test.csv"))[,c(9:104)]
swinuT <- read_csv(paste0(INPUT_DIR,"SwinUNetC_Train.csv"))[,c(9:104)]
swinuV <- read_csv(paste0(INPUT_DIR,"SwinUNetC_Test.csv"))[,c(9:104)]

#List Vars
uVars <- c("label", names(unetT)[2:97])
mVars <- c("label", names(mamT))
muVars <- c("label", names(mamuT))
sVars <- c("label", names(swinT))
suVars <- c("label", names(swinuT))

roleV <- c("outcome", rep("predictor", 96))

#Merge all training tables
allTrain <- bind_cols(unetT, mamT, mamuT, swinT, swinuT)
set.seed(42)
allTrain <- allTrain |> group_by(label) |> sample_n(1000, replace=FALSE) |> ungroup() |> data.frame()

mAll <- randomForest(y=allTrain[,1],
                     x=allTrain[,2:481],
                     ntree=1000,
                     importance=T,
                     confusion=T,
                     err.rate=T,
                     keep.forest = TRUE,
                     keep.inbag = TRUE)

impReUC <- permimp(mAll,
                   progressBar = TRUE,
                   conditional=FALSE,
                   do_check=FALSE)

plot(impReUC,
     type = "box",
     horizontal = TRUE,
     margin = c(3, 9, 3, 1),
     nVar=25)

impMeans <- data.frame(names = names(impReUC$perTree),
                       mns = impReUC$values,
                       nk=481-rank(impReUC$values),
                       grp = str_extract(names(impReUC$perTree), "^[^_]+"))

impMeans |> ggplot(aes(x=grp, y=nk))+
  geom_boxplot(fill="gray")+
  theme_bw(base_size=14)+
  labs(x="Feature Space", y="Variable Importance Rank")+
  theme(axis.text.x = element_text(color="black"))+
  theme(axis.text.y = element_text(color="black"))

# ==================================== PAIRWISE MANTEL TESTS AMONG DECODER FEATURE MAPS ====================================

library(vegan)
library(data.table)
library(tidyverse)

labSet <- read_csv(paste0(INPUT_DIR,"UNet_Train.csv"))[,4] |> mutate(label = as.factor(label)) |> mutate(idSeq = seq(1,640000, 1))

unetT <- read_csv(paste0(INPUT_DIR,"UNet_Train.csv"))[,c(9:104)]
mamT <- read_csv(paste0(INPUT_DIR,"MambaUNet_Train.csv"))[,c(9:104)]
mamuT <- read_csv(paste0(INPUT_DIR,"MambaUNetC_Train.csv"))[,c(9:104)]
swinT <- read_csv(paste0(INPUT_DIR,"SwinUNet_Train.csv"))[,c(9:104)]
swinuT <- read_csv(paste0(INPUT_DIR,"SwinUNetC_Train.csv"))[,c(9:104)]

mergedData <- bind_cols(labSet,unetT, mamT, mamuT, swinT, swinuT)

#Settings 
input_file   <- mergedData
class_col    <- "label"
image_col    <- NULL     # NULL if you have no image column
models       <- c("UNet", "SwinUNet", "SwinUNetC", "MambaUNet", "MambaUNetC")  # column prefixes
n_per_class  <- 500
n_resamples  <- 10
n_perm       <- 999            # 999 is enough for 10 tests with Holm (min p = 0.001 < 0.05/10)
base_seed    <- 20261112
n_cores      <- max(1L, parallel::detectCores() - 1L)
restrict_within_image <- FALSE # TRUE = permute only within source images (needs image_col)
out_prefix   <- "mantel"

#Helpers

#Stratified sample of row indices: n pixels per class, without replacement.
sample_pixels <- function(cls, n, seed) {
  set.seed(seed)
  by_class <- split(seq_along(cls), cls)
  idx <- lapply(names(by_class), function(k) {
    i <- by_class[[k]]
    if (length(i) < n) {
      warning(sprintf("Class '%s' has only %d pixels; using all of them.", k, length(i)))
      return(i)
    }
    i[sample.int(length(i), n)]
  })
  sort(unlist(idx, use.names = FALSE))
}

#Z-score each channel over the sampled pixels and return the
#lower triangle of the n x n cross-product (Gram) matrix as a dist object.
gram_dist <- function(X) {
  X <- as.matrix(X)
  sds <- apply(X, 2, sd)
  keep <- is.finite(sds) & sds > 0
  if (any(!keep)) message(sprintf("    dropped %d constant channel(s)", sum(!keep)))
  X <- scale(X[, keep, drop = FALSE], center = TRUE, scale = TRUE)
  G <- tcrossprod(X)              # n x n, = X %*% t(X)
  structure(as.dist(G), n_channels = sum(keep), n_dropped = sum(!keep))
}

#Load data
dt <- as.data.table(mergedData)

feature_cols <- setNames(lapply(models, function(m) {
  cols <- grep(paste0("^", m, "_"), names(dt), value = TRUE)
  if (length(cols) == 0) stop("No feature columns found for model prefix '", m, "_'")
  cols
}), models)

cat("Rows:", nrow(dt), "\n")
cat("Classes:", length(unique(dt[[class_col]])), "\n")
cat("Channels per model:", paste(models, lengths(feature_cols), sep = "=", collapse = ", "), "\n")

if (restrict_within_image && is.null(image_col))
  stop("restrict_within_image = TRUE requires image_col")

pairs <- combn(models, 2)
n_tests <- ncol(pairs)
cat("Pairwise tests per resample:", n_tests, "\n\n")

#Resampling loop 
results  <- vector("list", n_resamples)
channels <- vector("list", n_resamples)
samples  <- vector("list", n_resamples)

for (r in seq_len(n_resamples)) {
  seed <- base_seed + r
  rows <- sample_pixels(dt[[class_col]], n_per_class, seed)
  samples[[r]] <- rows
  cat(sprintf("Resample %d/%d (seed %d): %d pixels\n", r, n_resamples, seed, length(rows)))
  
  grams <- setNames(vector("list", length(models)), models)
  for (m in models) {
    cat("  Gram matrix:", m, "\n")
    grams[[m]] <- gram_dist(dt[rows, feature_cols[[m]], with = FALSE])
  }
  channels[[r]] <- data.table(resample = r, model = models,
                              n_channels = sapply(grams, attr, "n_channels"),
                              n_dropped  = sapply(grams, attr, "n_dropped"))
  
  strata <- if (restrict_within_image) factor(dt[[image_col]][rows]) else NULL
  
  res_r <- rbindlist(lapply(seq_len(n_tests), function(k) {
    a <- pairs[1, k]; b <- pairs[2, k]
    cat(sprintf("  Mantel: %s vs %s\n", a, b))
    set.seed(base_seed + 1000L * r + k)        # reproducible permutations
    mt <- mantel(grams[[a]], grams[[b]],
                 method = "pearson",
                 permutations = n_perm,
                 strata = strata,
                 parallel = n_cores)
    data.table(resample = r, seed = seed, model_1 = a, model_2 = b,
               n_pixels = length(rows), mantel_r = unname(mt$statistic),
               p_value = mt$signif)
  }))
  
  # Holm correction across the pairwise comparisons in this resample.
  # The family of hypotheses is the set of model pairs; resamples are
  # replicate draws used to assess sampling sensitivity, not new hypotheses.
  res_r[, p_holm := p.adjust(p_value, method = "holm")]
  results[[r]] <- res_r
  
  rm(grams); invisible(gc())
}

all_res  <- rbindlist(results)
chan_tab <- rbindlist(channels)

#Summary across resamples
summary_tab <- all_res[, .(
  mean_r       = mean(mantel_r),
  sd_r         = sd(mantel_r),
  min_r        = min(mantel_r),
  max_r        = max(mantel_r),
  max_p_holm   = max(p_holm),               # conservative: worst draw
  n_sig_holm   = sum(p_holm < 0.05),
  n_resamples  = .N
), by = .(model_1, model_2)][order(-mean_r)]

print(summary_tab, digits = 3)

#Save outputs
fwrite(all_res,     paste0(out_prefix, "_all_resamples.csv"))
fwrite(summary_tab, paste0(out_prefix, "_summary.csv"))
fwrite(chan_tab,    paste0(out_prefix, "_channels.csv"))
saveRDS(list(sampled_rows = samples, settings = list(
  input_file = input_file, models = models, n_per_class = n_per_class,
  n_resamples = n_resamples, n_perm = n_perm, base_seed = base_seed,
  restrict_within_image = restrict_within_image)),
  paste0(out_prefix, "_samples_and_settings.rds"))
writeLines(capture.output(sessionInfo()), paste0(out_prefix, "_sessionInfo.txt"))

cat("\nDone. Outputs written with prefix '", out_prefix, "'.\n", sep = "")

