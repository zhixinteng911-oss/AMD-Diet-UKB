library(survival)
library(data.table)
BASE <- "."
dir.create(file.path(BASE, "output"), showWarnings=FALSE, recursive=TRUE)
M2 <- c("sex_binary","Energy_Total_z","tdi_normal_z","EduYears","smoker_current",
        "smoker_former","PA_METs","alcohol_frequent","AMD_PRS_z")
all <- fread(paste0(BASE, "/input/analysis_master_dataset.csv"), data.table=FALSE)
train <- fread(paste0(BASE, "/input/train_A_random.csv"), data.table=FALSE)
qbreaks <- quantile(train$PANDA_score, probs=c(0,0.25,0.5,0.75,1), na.rm=TRUE, type=7)
qbreaks[1] <- -Inf
qbreaks[5] <- Inf
cat("training-set PANDA quartile cut-points:", paste(round(qbreaks,2), collapse=", "), "\n")
results <- list()
for(scheme in c("A_random","B_centre","C_time")) {
  test_path <- paste0(BASE, "/input/test_", scheme, ".csv")
  if(!file.exists(test_path)) {
    cat("skip", scheme, "(split file not found)\n")
    next
  }
  cat("\nscheme:", scheme, "\n")
  test <- fread(test_path, data.table=FALSE)
  cat(sprintf("test: n=%d, events=%d\n", nrow(test), sum(test$event_amd)))
  f <- as.formula(paste0("Surv(age_at_index, age_at_exit, event_amd) ~ PANDA_score_pct + ", paste(M2, collapse="+")))
  c <- coxph(f, data=test)
  hr <- exp(coef(c)["PANDA_score_pct"])
  ci <- exp(confint(c)["PANDA_score_pct",])
  p <- summary(c)$coefficients["PANDA_score_pct","Pr(>|z|)"]
  cat(sprintf("  per 1 pct: HR=%.3f (%.3f-%.3f), P=%.6f\n", hr, ci[1], ci[2], p))
  test$PANDA_q <- cut(test$PANDA_score, breaks=qbreaks, labels=c("Q1","Q2","Q3","Q4"), include.lowest=TRUE)
  cat("  quartile distribution:\n")
  print(table(test$PANDA_q, useNA="ifany"))
  fq <- as.formula(paste0("Surv(age_at_index, age_at_exit, event_amd) ~ PANDA_q + ", paste(M2, collapse="+")))
  cq <- coxph(fq, data=test)
  sq <- summary(cq)
  cat("  Q1-Q4:\n")
  for(q in c("Q2","Q3","Q4")) {
    cname <- paste0("PANDA_q", q)
    if(cname %in% names(coef(cq))) {
      hrq <- exp(coef(cq)[cname])
      ciq <- exp(confint(cq)[cname,])
      pq <- sq$coefficients[cname,"Pr(>|z|)"]
      cat(sprintf("    %s vs Q1: HR=%.3f (%.3f-%.3f), P=%.4f\n", q, hrq, ciq[1], ciq[2], pq))
    }
  }
  ph <- cox.zph(c)
  cat(sprintf("  PH test P=%.3f\n", ph$table["PANDA_score_pct","p"]))
  results[[scheme]] <- data.frame(
    scheme=scheme, n=nrow(test), events=sum(test$event_amd),
    per20_HR=hr, per20_CIlo=ci[1], per20_CIhi=ci[2], per20_P=p,
    PH_P=ph$table["PANDA_score_pct","p"]
  )
}
cat("\nfull sample\n")
f_all <- as.formula(paste0("Surv(age_at_index, age_at_exit, event_amd) ~ PANDA_score_pct + ", paste(M2, collapse="+")))
c_all <- coxph(f_all, data=all)
hr_all <- exp(coef(c_all)["PANDA_score_pct"])
ci_all <- exp(confint(c_all)["PANDA_score_pct",])
p_all <- summary(c_all)$coefficients["PANDA_score_pct","Pr(>|z|)"]
cat(sprintf("  per 1 pct: HR=%.3f (%.3f-%.3f), P=%.6f\n", hr_all, ci_all[1], ci_all[2], p_all))
res_df <- do.call(rbind, results)
res_df <- rbind(res_df, data.frame(scheme="Full", n=nrow(all), events=sum(all$event_amd),
                                     per20_HR=hr_all, per20_CIlo=ci_all[1], per20_CIhi=ci_all[2], per20_P=p_all, PH_P=NA))
write.csv(res_df, paste0(BASE, "/output/validation_cox_summary.csv"), row.names=FALSE)
print(res_df)
