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

``` r
mod <- rstan::sampling(IRT_1PL, chains = 3, data = list(N = nrow(LSAT), N_I = ncol(LSAT), Y = LSAT))
```

``` r
print(mod, pars = c("disc", "diff"))
```

    ## Inference for Stan model: abeca9ec42ff7652768a70fe559c2763.
    ## 3 chains, each with iter=2000; warmup=1000; thin=1; 
    ## post-warmup draws per chain=1000, total post-warmup draws=3000.
    ## 
    ##          mean se_mean   sd  2.5%   25%   50%   75% 97.5% n_eff Rhat
    ## disc     0.72    0.00 0.07  0.58  0.67  0.72  0.77  0.86   363    1
    ## diff[1] -3.81    0.02 0.39 -4.69 -4.04 -3.77 -3.53 -3.16   381    1
    ## diff[2] -1.39    0.01 0.16 -1.76 -1.49 -1.38 -1.28 -1.11   476    1
    ## diff[3] -0.34    0.00 0.11 -0.55 -0.41 -0.33 -0.27 -0.13  1969    1
    ## diff[4] -1.82    0.01 0.19 -2.24 -1.93 -1.80 -1.68 -1.49   428    1
    ## diff[5] -2.92    0.01 0.30 -3.59 -3.09 -2.89 -2.72 -2.42   415    1
    ## 
    ## Samples were drawn using NUTS(diag_e) at Thu Apr 30 00:47:19 2020.
    ## For each parameter, n_eff is a crude measure of effective sample size,
    ## and Rhat is the potential scale reduction factor on split chains (at 
    ## convergence, Rhat=1).
