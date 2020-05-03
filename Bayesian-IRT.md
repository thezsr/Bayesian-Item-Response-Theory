Bayesian IRT
================
Reginald Ziedzor
4/29/2020

# Bayesian Item Response Theory using Stan

This is an introduction to using Stan for item score analysis. We are
using the LSAT dataset from the `ltm` package in R.

``` r
data(LSAT, package = "ltm")
```

We can look at the first few observations of the LSAT dataset

``` r
head(LSAT)
```

    ##   Item 1 Item 2 Item 3 Item 4 Item 5
    ## 1      0      0      0      0      0
    ## 2      0      0      0      0      0
    ## 3      0      0      0      0      0
    ## 4      0      0      0      0      1
    ## 5      0      0      0      0      1
    ## 6      0      0      0      0      1

## Running Stan Code

``` r
library(rstan)
```

    ## Loading required package: StanHeaders

    ## Loading required package: ggplot2

    ## rstan (Version 2.19.3, GitRev: 2e1f913d3ca3)

    ## For execution on a local, multicore CPU with excess RAM we recommend calling
    ## options(mc.cores = parallel::detectCores()).
    ## To avoid recompilation of unchanged Stan programs, we recommend calling
    ## rstan_options(auto_write = TRUE)

    ## For improved execution time, we recommend calling
    ## Sys.setenv(LOCAL_CPPFLAGS = '-march=corei7 -mtune=corei7')
    ## although this causes Stan to throw an error on a few processors.

``` r
options(mc.cores = parallel::detectCores()-1)
rstan_options(auto_write = TRUE)
```

Using `rstan` to run the Stan code.

``` stan
data {
  int N_I; //number of items
  int N; //number of examinees
  int<lower = 0, upper = 1> Y[N, N_I]; //observed data
}

parameters {
  vector[N] theta; //ability levels of examinees
  vector[N_I] diff; //item difficulty
  real<lower = 0> disc; //ensuring item discrimination parameter is nonnegative
  real<lower = 0> sigma_disc;
  real mu_diff;
}

model {
  real p_ij; //probability of correctly responding to an item
  
  for (j in 1:N) {
    for(i in 1:N_I) {
      
      p_ij = disc*(theta[j] - diff[i]);
      
      Y[j,i] ~ bernoulli_logit(p_ij); 
    }
  }
  //Priors
  mu_diff ~ normal(0,2);
  theta ~ std_normal();
  diff ~ normal(mu_diff,5);
  disc ~ normal(0,sigma_disc);
  sigma_disc ~ cauchy(0,2); 
}
```

Posterior Sampling

``` r
mod <- rstan::sampling(IRT_1PL, chains = 3, data = list(N = nrow(LSAT), N_I = ncol(LSAT), Y = LSAT))
```

Posterior Estimates

``` r
print(mod, pars = c("disc", "diff"))
```

    ## Inference for Stan model: abeca9ec42ff7652768a70fe559c2763.
    ## 3 chains, each with iter=2000; warmup=1000; thin=1; 
    ## post-warmup draws per chain=1000, total post-warmup draws=3000.
    ## 
    ##          mean se_mean   sd  2.5%   25%   50%   75% 97.5% n_eff Rhat
    ## disc     0.72    0.00 0.07  0.58  0.68  0.73  0.77  0.86   521    1
    ## diff[1] -3.78    0.02 0.37 -4.62 -4.00 -3.74 -3.52 -3.16   560    1
    ## diff[2] -1.38    0.01 0.16 -1.72 -1.48 -1.37 -1.27 -1.10   696    1
    ## diff[3] -0.33    0.00 0.10 -0.53 -0.40 -0.33 -0.26 -0.14  2404    1
    ## diff[4] -1.81    0.01 0.19 -2.22 -1.92 -1.79 -1.67 -1.48   666    1
    ## diff[5] -2.90    0.01 0.29 -3.55 -3.07 -2.86 -2.71 -2.42   571    1
    ## 
    ## Samples were drawn using NUTS(diag_e) at Sat May 02 20:59:48 2020.
    ## For each parameter, n_eff is a crude measure of effective sample size,
    ## and Rhat is the potential scale reduction factor on split chains (at 
    ## convergence, Rhat=1).

Using the `ltm` package to obtain maximum likelihood estimates

``` r
library(ltm)
```

    ## Loading required package: MASS

    ## Loading required package: msm

    ## Loading required package: polycor

``` r
model_rasch <- rasch(data = LSAT, IRT.param = TRUE)
summary(model_rasch)
```

    ## 
    ## Call:
    ## rasch(data = LSAT, IRT.param = TRUE)
    ## 
    ## Model Summary:
    ##    log.Lik      AIC      BIC
    ##  -2466.938 4945.875 4975.322
    ## 
    ## Coefficients:
    ##                 value std.err   z.vals
    ## Dffclt.Item 1 -3.6153  0.3266 -11.0680
    ## Dffclt.Item 2 -1.3224  0.1422  -9.3009
    ## Dffclt.Item 3 -0.3176  0.0977  -3.2518
    ## Dffclt.Item 4 -1.7301  0.1691 -10.2290
    ## Dffclt.Item 5 -2.7802  0.2510 -11.0743
    ## Dscrmn         0.7551  0.0694  10.8757
    ## 
    ## Integration:
    ## method: Gauss-Hermite
    ## quadrature points: 21 
    ## 
    ## Optimization:
    ## Convergence: 0 
    ## max(|grad|): 2.9e-05 
    ## quasi-Newton: BFGS
