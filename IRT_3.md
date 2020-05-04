3PL
================
Reginald Ziedzor
5/2/2020

# Three Parameter Logistic Model

In this example, we will still be using the `LSAT` dataset from the
`ltm` package.

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
library(ltm)
```

    ## Loading required package: MASS

    ## Loading required package: msm

    ## Loading required package: polycor

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

The code for the 3PL in Stan

``` stan
data {
  int N_I; //number of items
  int N; //number of examinees
  int<lower = 0, upper = 1> Y[N, N_I]; //observed data
}

parameters {
  vector[N] theta; //ability levels of examinees
  vector[N_I] diff; //item difficulty
  vector<lower = 0>[N_I] disc; //ensuring item discrimination parameter is nonnegative
  real<lower = 0> sigma_disc;
  vector<lower = 0, upper = 1>[N_I] guess;
  real mu_diff;
}

model {
  real p_ij; //probability of correctly responding to an item
  real eta;
  
  for (j in 1:N) {
    for(i in 1:N_I) {
      
      eta = inv_logit(disc[i]*(theta[j] - diff[i]));
      p_ij = guess[i] + (1 - guess[i])*eta;
      
      Y[j,i] ~ bernoulli(p_ij); 
    }
  }
  //Priors
  mu_diff ~ normal(0,2);
  theta ~ std_normal();
  diff ~ normal(mu_diff,5);
  disc ~ normal(0,sigma_disc);
  sigma_disc ~ cauchy(0,2); 
  guess ~ beta(5,23); 
}
```

``` r
options(mc.cores = parallel::detectCores()-1)
rstan_options(auto_write = TRUE)
```

Posterior Sampling

``` r
stan_3pl <- rstan::sampling(IRT_3PL, chains = 3, data = list(N = nrow(LSAT), N_I = ncol(LSAT), Y = LSAT))
```

Posterior Estimates

``` r
print(stan_3pl, pars = c("disc", "diff", "guess"))
```

    ## Inference for Stan model: 3e84f7b255c820c7d3d75aaf180fd6c6.
    ## 3 chains, each with iter=2000; warmup=1000; thin=1; 
    ## post-warmup draws per chain=1000, total post-warmup draws=3000.
    ## 
    ##           mean se_mean   sd  2.5%   25%   50%   75% 97.5% n_eff Rhat
    ## disc[1]   0.79    0.01 0.28  0.34  0.60  0.77  0.96  1.45   782 1.01
    ## disc[2]   0.79    0.01 0.24  0.37  0.63  0.77  0.92  1.34   713 1.00
    ## disc[3]   1.12    0.02 0.42  0.51  0.85  1.05  1.31  2.20   442 1.01
    ## disc[4]   0.73    0.01 0.24  0.33  0.56  0.71  0.87  1.27   774 1.00
    ## disc[5]   0.66    0.01 0.23  0.27  0.50  0.64  0.79  1.15   924 1.00
    ## diff[1]  -3.59    0.06 1.33 -6.89 -4.06 -3.28 -2.72 -2.02   466 1.02
    ## diff[2]  -0.93    0.01 0.40 -1.85 -1.13 -0.90 -0.67 -0.28  1116 1.00
    ## diff[3]   0.19    0.01 0.24 -0.19  0.03  0.16  0.33  0.71  1448 1.00
    ## diff[4]  -1.51    0.02 0.56 -2.88 -1.79 -1.41 -1.15 -0.70   901 1.00
    ## diff[5]  -3.10    0.04 1.21 -6.32 -3.54 -2.82 -2.32 -1.67   809 1.01
    ## guess[1]  0.18    0.00 0.07  0.06  0.13  0.17  0.23  0.35  5873 1.00
    ## guess[2]  0.18    0.00 0.07  0.07  0.13  0.18  0.22  0.34  3571 1.00
    ## guess[3]  0.17    0.00 0.07  0.06  0.12  0.16  0.21  0.31  1653 1.00
    ## guess[4]  0.18    0.00 0.07  0.07  0.13  0.17  0.23  0.34  4515 1.00
    ## guess[5]  0.18    0.00 0.08  0.06  0.12  0.17  0.23  0.35  8285 1.00
    ## 
    ## Samples were drawn using NUTS(diag_e) at Mon May 04 12:11:31 2020.
    ## For each parameter, n_eff is a crude measure of effective sample size,
    ## and Rhat is the potential scale reduction factor on split chains (at 
    ## convergence, Rhat=1).
