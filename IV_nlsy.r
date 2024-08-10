## Code for the project 'Semiparametric IV Regression without Exclusion Restrictions' by Wayne Gao and Rui Wang
## It is replicating the empirical results in the Application: Family background variables

## packages
library(haven)
library(sandwich)
library(randomForest)
library(e1071)
library(neuralnet)

rm(list=ls())

## read data
data_nlsy <- read_dta("NLSY79.dta")

## parameters of CI
alpha <- 0.05           # significance level
q <- qnorm(1-alpha/2)   # quantile

## dependent variable
Y <- log(data_nlsy$EARNINGS)

## features
S <- data_nlsy$S               # years of schoolings
female <- data_nlsy$FEMALE     # whether an individual is female
black <- data_nlsy$ETHBLACK    # whether an individual is black
marr <- data_nlsy$MARRIED      # whether an individual is married
exp<- data_nlsy$EXP            # working experience
hour <- data_nlsy$HOURS        # working hours 

## family background variables
SM <- data_nlsy$SM             # mothers' education
SF <- data_nlsy$SF             # fathers' education
SMF <- (SM + SF)/2             
sib <- data_nlsy$SIBLINGS      # number of siblings

## region indicators
regnc <- data_nlsy$REGNC
regne <- data_nlsy$REGNE
regs <- data_nlsy$REGS


# get summary statistics
V <- cbind(Y, S, female, black, marr, exp, hour, regnc, regne, regs, SMF, sib)
s <- cbind(colMeans(V), apply(V,2,sd), apply(V,2,min), apply(V,2,max))
round(s, 3)


## get all variables
n <- length(Y)
W_2 <- cbind(rep(1,n), S, female, black, marr, exp, hour, regnc, regne, regs, SMF, sib) # all regressors
d_1<-ncol(W_2)

#build two formulas
formX <- S ~ female + black + marr + exp + hour + regnc + regne + regs + SMF + sib
formY <- Y ~ female + black + marr + exp + hour + regnc + regne + regs + SMF + sib

vall <-cbind(Y, S, female, black, marr, exp, hour, regnc, regne, regs, SMF, sib)


## First step nonparametric estimation
## First-step estimator 1: support vector machine
modelX <-svm(formX)
Xhat_2 <- fitted(modelX)

modelY <- svm(formY)
Yhat_2 <- fitted(modelY)


## First-step estimator 2: random forest estimation
set.seed(123)
modelX <- randomForest(formX, ntree=500)
Xhat_2 <- predict(modelX, vall)

modelY <- randomForest(formY, ntree=500)
Yhat_2 <- predict(modelY, vall)


## First-step estimator 3: neural network
maxs <- apply(vall, 2, max)
mins <- apply(vall, 2, min)
vall_norm <- as.data.frame(scale(vall, center = mins, scale = maxs - mins))

set.seed(123)
modelX_nn <- neuralnet(formX, data = vall_norm, algorithm = "rprop+", hidden=c(5), threshold=0.1, stepmax = 1e+06)
modelY_nn <- neuralnet(formY, data = vall_norm, algorithm = "rprop+", hidden=c(5), threshold=0.1, stepmax = 1e+06)

Xhat_2 <- predict(modelX_nn, vall_norm)*(max(S) - min(S)) + min(S)
Yhat_2 <- predict(modelY_nn, vall_norm)*(max(Y) - min(Y)) + min(Y)


## Second step estimation
## Estimation I: use Y as a dependent variable
What_2 <- cbind(rep(1,n), Xhat_2, female, black, marr, exp, hour, regnc, regne, regs, SMF, sib)  # all regressor
Z_2 <- cbind(female, black, marr, exp, hour, regnc, regne, regs, SMF, sib)         # instrument

reg1_2 <- lm(Y ~ Xhat_2+Z_2)
est1_2 <- reg1_2$coefficients

# standard error
H_2 <- solve(t(What_2) %*% What_2 /n)      # E[W_i'W_i]^(-1)
e1sq_2 <- (Y-W_2 %*% est1_2)^2             # squre of error
V1_2 <- t(What_2) %*% (What_2 * (e1sq_2 %*% rep(1,d_1))) /n 
var1_2 <- H_2 %*% V1_2 %*% H_2 /n          # estimated variance of theta_hat
sd1_2 <- sqrt(diag(var1_2))                # standard error
CI1_2 <- cbind(est1_2 - q * sd1_2, est1_2+ q * sd1_2)  # confidence interval


## Estimation II: estimate E[Y|Z] nonparametrically
reg2_2 <- lm(Yhat_2 ~ Xhat_2+Z_2)
est2_2 <- reg2_2$coefficients

# standard error
e2sq_2 <- (Y-W_2 %*% est2_2)^2          # squre of error
V2_2 <- t(What_2) %*% (What_2 * (e2sq_2 %*% rep(1,d_1))) /n 
var2_2 <- H_2 %*% V2_2 %*% H_2 /n      # estimated variance
sd2_2 <- sqrt(diag(var2_2))            # standard error
CI2_2 <- cbind(est2_2 - q * sd2_2, est2_2+ q * sd2_2)  # confidence interval


## Estimation III: discretized estimator
exp_ind <- as.numeric(exp <= quantile(exp,probs=1/2))       # indicator for exp>median(exp)
hour_ind <- as.numeric(hour <= quantile(hour, probs=1/2))   # indicator for working hours
SMF_ind <- as.numeric(SMF <= quantile(SMF, probs=1/2))      # indicator of parents' education
sib_ind <- as.numeric(sib <= quantile(sib, probs=1/2))      # indicator for number of siblings

Zdis <- cbind(female, black, marr, exp_ind, hour_ind, regnc, regne, regs, SMF_ind, sib_ind)
ndis <-ncol(Zdis)

# Initialize 3D array

D<-array(0, dim = c(n, ndis, ndis))
# # Fill the array
for (i in 1: ndis-1) {
  for (j in (i+1): ndis) {
    D[ , i, j] <- Zdis[ , i] * Zdis[ , j]
  }
}

# Reshape the array
D <- array(D, dim = c(n, ndis*ndis))
Dm <- colMeans(D)
DD <- D[, Dm != 0]

est_dis <- solve(t(W_2) %*% DD %*% solve(t(DD) %*% DD) %*% t(DD) %*% W_2) %*% t(W_2) %*% DD %*% solve(t(DD) %*% DD) %*% t(DD) %*% Y

# standard error
H_dis <- t(W_2) %*% DD %*% solve(t(DD) %*% DD) %*% t(DD) %*% W_2 / n
O_dis <- t(DD) %*% (DD * ((Y - W_2 %*% est_dis) %*% rep(1, ncol(DD)))^2) / n
V_dis <- t(W_2) %*% DD %*% solve(t(DD) %*% DD) %*% O_dis %*% solve(t(DD) %*% DD) %*% t(DD) %*% W_2  
var_dis <- solve(H_dis) %*% V_dis %*% solve(H_dis) / n     # estimated variance
sd_dis <- sqrt(diag(var_dis))                              # estimated standard deviation
CI_dis<- cbind(est_dis - q*sd_dis, est_dis + q*sd_dis)     # estimated CI


## OLS with instruments
WW_2 <- cbind(S, female, black, marr, exp, hour,  regnc, regne, regs, SMF, sib) # all regressors
reg3_2 <- lm(Y ~ WW_2)
est3_2 <- reg3_2$coefficients 

# standard error of OLS
H3_2 <- solve(t(W_2) %*% W_2 /n)                             # E[W_i'W_i]
e3sq_2 <- (Y - W_2 %*% est3_2)^2
V3_2 <- t(W_2) %*% (W_2 *  (e3sq_2 %*% rep(1,d_1)))/n        # E[e_i^2W_i'W_i]
var3_2 <- H3_2 %*% V3_2 %*% H3_2 /n                          # estimated asymptotic variance
sd3_2 <- sqrt(diag(var3_2))
CI3_2<- cbind(est3_2 - q*sd3_2, est3_2 + q*sd3_2)            # estimated CI


## OLS without instruments 
WW2_2 <- cbind(S, female, black, marr, exp, hour, regnc, regne, regs)  # all regressors
W2_2 <-cbind(rep(1,n), S, female, black, marr, exp, hour, regnc, regne, regs)
reg4_2 <- lm(Y ~ WW2_2)
est4_2 <- reg4_2$coefficients

# standard error of OLS
H4_2 <- solve(t(W2_2) %*% W2_2 /n)                                  # E[W_i'W_i]
e4sq_2 <- (Y - W2_2 %*% est4_2)^2
V4_2 <- t(W2_2) %*% (W2_2 *  (e4sq_2 %*% rep(1,ncol(W2_2))))/n      # E[e_i^2W_i'W_i]
var4_2 <- H4_2 %*% V4_2 %*% H4_2 /n                                 # estimated asymptotic variance
sd4_2 <- sqrt(diag(var4_2))
CI4_2<- cbind(est4_2 - q*sd4_2, est4_2 + q*sd4_2)


## 2sls with two instrument
# Create all instruments and regressors
Zall_2 <- cbind(rep(1,n), female, black, marr, exp, hour, regnc, regne, regs, SMF, sib)
Xall <- cbind(rep(1,n), S, female, black, marr, exp, hour, regnc, regne, regs)
est5_2<- solve(t(Xall) %*% Zall_2 %*% solve(t(Zall_2) %*% Zall_2) %*% t(Zall_2) %*% Xall) %*% t(Xall) %*% Zall_2 %*% solve(t(Zall_2) %*% Zall_2) %*% t(Zall_2) %*% Y

# Calculate H5 (2sls standard error)
H5_2 <- t(Xall) %*% Zall_2 %*% solve(t(Zall_2) %*% Zall_2) %*% t(Zall_2) %*% Xall / n
O5_2 <- t(Zall_2) %*% (Zall_2 * ((Y - Xall %*% est5_2) %*% rep(1,ncol(Zall_2)))^2) / n
V5_2 <- t(Xall) %*% Zall_2 %*% solve(t(Zall_2) %*% Zall_2) %*% O5_2 %*% solve(t(Zall_2) %*% Zall_2) %*% t(Zall_2) %*% Xall
var5_2 <- solve(H5_2) %*% V5_2 %*% solve(H5_2) / n
sd5_2 <- sqrt(diag(var5_2))                           # estimated standard error
CI5_2<- cbind(est5_2 - q*sd5_2, est5_2 + q*sd5_2)     # confidence interval


## 2sls with parents' education
# Create all instruments and regressors
Zall_par <- cbind(rep(1,n), female, black, marr, exp, hour, regnc, regne, regs, SMF)
est6_2<- solve(t(Xall) %*% Zall_par %*% solve(t(Zall_par) %*% Zall_par) %*% t(Zall_par) %*% Xall) %*% t(Xall) %*% Zall_par %*% solve(t(Zall_par) %*% Zall_par) %*% t(Zall_par) %*% Y

# Calculate H5 (2sls standard error)
H6_2 <- t(Xall) %*% Zall_par %*% solve(t(Zall_par) %*% Zall_par) %*% t(Zall_par) %*% Xall / n
O6_2 <- t(Zall_par) %*% (Zall_par * ((Y - Xall %*% est6_2) %*% rep(1,ncol(Zall_par)))^2) / n
V6_2 <- t(Xall) %*% Zall_par %*% solve(t(Zall_par) %*% Zall_par) %*% O6_2 %*% solve(t(Zall_par) %*% Zall_par) %*% t(Zall_par) %*% Xall
var6_2 <- solve(H6_2) %*% V6_2 %*% solve(H6_2) / n
sd6_2 <- sqrt(diag(var6_2))                            # estimated standard error
CI6_2<- cbind(est6_2 - q*sd6_2, est6_2 + q*sd6_2)      # confidence interval


## 2sls with number of siblings
# Create all instruments and regressors
Zall_sib <- cbind(rep(1,n), female, black, marr, exp, hour, regnc, regne, regs, sib)
est7_2<- solve(t(Xall) %*% Zall_sib %*% solve(t(Zall_sib) %*% Zall_sib) %*% t(Zall_sib) %*% Xall) %*% t(Xall) %*% Zall_sib %*% solve(t(Zall_sib) %*% Zall_sib) %*% t(Zall_sib) %*% Y

# Calculate H5 (2sls standard error)
H7_2 <- t(Xall) %*% Zall_sib %*% solve(t(Zall_sib) %*% Zall_sib) %*% t(Zall_sib) %*% Xall / n
O7_2 <- t(Zall_sib) %*% (Zall_sib * ((Y - Xall %*% est7_2) %*% rep(1,ncol(Zall_sib)))^2) / n
V7_2 <- t(Xall) %*% Zall_sib %*% solve(t(Zall_sib) %*% Zall_sib) %*% O7_2 %*% solve(t(Zall_sib) %*% Zall_sib) %*% t(Zall_sib) %*% Xall
var7_2 <- solve(H7_2) %*% V7_2 %*% solve(H7_2) / n
sd7_2 <- sqrt(diag(var7_2))                             # estimated standard error
CI7_2<- cbind(est7_2 - q*sd7_2, est7_2 + q*sd7_2)       # confidence interval

