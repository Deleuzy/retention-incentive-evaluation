# Independent cross-check in R of three numbers produced by the Python pipeline.
#
#  1. Experimental benchmark: difference in means and Welch 95% CI (stats::t.test),
#     read straight from the raw Stata file, not from anything Python wrote.
#  2. IPW estimate of the ATT on the PSID and CPS samples, propensity score fitted with
#     stats::glm using the same specification as src/rie/estimators.py::ps_design.
#  3. Stratified Cox hazard ratio for months 0 to 12 in the SIMULATED pilot data
#     (survival::coxph, Efron ties, the same default as lifelines).
#
# Uses only packages that ship with R (foreign, survival) plus jsonlite.
# Run from the repository root:  Rscript r/crosscheck.R

suppressPackageStartupMessages({
  library(foreign)
  library(survival)
  library(jsonlite)
})

raw <- function(f) {
  d <- read.dta(file.path("data", "raw", f))
  d$data_id <- NULL
  d
}

nsw <- raw("nsw_dw.dta")
tt <- t.test(re78 ~ factor(treat, levels = c(1, 0)), data = nsw, var.equal = FALSE)
benchmark <- unname(tt$estimate[1] - tt$estimate[2])

ipw_att <- function(controls_file) {
  d <- rbind(nsw[nsw$treat == 1, ], raw(controls_file))
  d$u74 <- as.numeric(d$re74 == 0)
  d$u75 <- as.numeric(d$re75 == 0)
  d$r4 <- d$re74 / 1000
  d$r5 <- d$re75 / 1000
  fit <- glm(
    treat ~ age + I(age^2) + education + I(education^2) + black + hispanic + married +
      nodegree + r4 + r5 + I(r4^2) + I(r5^2) + u74 + u75 + I(education * r4),
    family = binomial(), data = d, control = glm.control(maxit = 100, epsilon = 1e-12)
  )
  p <- pmin(pmax(fitted(fit), 1e-6), 1 - 1e-6)
  w <- p / (1 - p)
  co <- d$treat == 0
  mean(d$re78[!co]) - sum(w[co] * d$re78[co]) / sum(w[co])
}

pilot <- read.csv(file.path("design", "simulated_pilot_data.csv"))
pilot$ev12 <- ifelse(pilot$months <= 12, pilot$event, 0)
pilot$t12 <- pmin(pilot$months, 12)
cx <- coxph(Surv(t12, ev12) ~ treat + strata(stratum), data = pilot, ties = "efron")

out <- list(
  r_version = R.version.string,
  benchmark = benchmark,
  benchmark_welch_ci = unname(as.numeric(tt$conf.int)),
  ipw_psid = ipw_att("psid_controls.dta"),
  ipw_cps = ipw_att("cps_controls.dta"),
  cox_hr_months_0_12_simulated = unname(exp(coef(cx))["treat"])
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 10, pretty = TRUE),
           file.path("r", "r_crosscheck.json"))
cat(toJSON(out, auto_unbox = TRUE, digits = 6, pretty = TRUE), "\n")
