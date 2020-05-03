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
    ## Samples were drawn using NUTS(diag_e) at Sun May 03 01:00:27 2020.
    ## For each parameter, n_eff is a crude measure of effective sample size,
    ## and Rhat is the potential scale reduction factor on split chains (at 
    ## convergence, Rhat=1).

``` r
model_3pl <- ltm::tpm(data = LSAT, type = "latent.trait", IRT.param = TRUE)
model_3pl
```

    ## 
    ## Call:
    ## ltm::tpm(data = LSAT, type = "latent.trait", IRT.param = TRUE)
    ## 
    ## Coefficients:
    ##         Gussng  Dffclt  Dscrmn
    ## Item 1   0.037  -3.296   0.829
    ## Item 2   0.078  -1.145   0.760
    ## Item 3   0.012  -0.249   0.902
    ## Item 4   0.035  -1.766   0.701
    ## Item 5   0.053  -2.990   0.666
    ## 
    ## Log.Lik: -2466.66

Using `brms` for 3

``` r
library(brms)
```

    ## Loading required package: Rcpp

    ## Loading 'brms' package (version 2.12.0). Useful instructions
    ## can be found by typing help('brms'). A more detailed introduction
    ## to the package is available through vignette('brms_overview').

    ## 
    ## Attaching package: 'brms'

    ## The following object is masked from 'package:rstan':
    ## 
    ##     loo

    ## The following object is masked from 'package:stats':
    ## 
    ##     ar

``` r
formula_3pl <- brms::bf(
response2 ~ gamma + (1 - gamma) * inv_logit(beta + exp(logalpha) * theta),
nl = TRUE,
theta ~ 0 + (1 | person),
beta ~ 1 + (1 |i| item),
logalpha ~ 1 + (1 |i| item),
logitgamma ~ 1 + (1 |i| item),
nlf(gamma ~ inv_logit(logitgamma)),
family = brmsfamily("bernoulli", link = "identity"))
```

Converting LSAT data from wide to long in order to use `brms`

``` r
ID <- 1:nrow(LSAT)
LSAT1 <- data.frame(LSAT, ID)
LSAT_long <- reshape(LSAT1, varying = c("Item.1", "Item.2", "Item.3", "Item.4", "Item.5"), 
                     idvar = "ID",direction = "long")
LSAT_long <- reshape(LSAT1, varying = c("Item.1", "Item.2", "Item.3", "Item.4", "Item.5"), 
                     idvar = "ID",direction = "long")
colnames(LSAT_long) <- c("person", "item", "response2")
```

``` r
brms::brm(formula = formula_3pl, data = LSAT_long)
```

    ## Compiling the C++ model

    ## Start sampling

    ## Warning: There were 259 divergent transitions after warmup. Increasing adapt_delta above 0.8 may help. See
    ## http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup

    ## Warning: Examine the pairs() plot to diagnose sampling problems

    ## Warning: The largest R-hat is 2.54, indicating chains have not mixed.
    ## Running the chains for more iterations may help. See
    ## http://mc-stan.org/misc/warnings.html#r-hat

    ## Warning: Bulk Effective Samples Size (ESS) is too low, indicating posterior means and medians may be unreliable.
    ## Running the chains for more iterations may help. See
    ## http://mc-stan.org/misc/warnings.html#bulk-ess

    ## Warning: Tail Effective Samples Size (ESS) is too low, indicating posterior variances and tail quantiles may be unreliable.
    ## Running the chains for more iterations may help. See
    ## http://mc-stan.org/misc/warnings.html#tail-ess

    ## Warning: Parts of the model have not converged (some Rhats are > 1.05). Be
    ## careful when analysing the results! We recommend running more iterations and/or
    ## setting stronger priors.

    ## Warning: There were 259 divergent transitions after warmup. Increasing
    ## adapt_delta above 0.8 may help. See http://mc-stan.org/misc/
    ## warnings.html#divergent-transitions-after-warmup

    ##  Family: bernoulli 
    ##   Links: mu = identity 
    ## Formula: response2 ~ gamma + (1 - gamma) * inv_logit(beta + exp(logalpha) * theta) 
    ##          theta ~ 0 + (1 | person)
    ##          beta ~ 1 + (1 | i | item)
    ##          logalpha ~ 1 + (1 | i | item)
    ##          logitgamma ~ 1 + (1 | i | item)
    ##          gamma ~ inv_logit(logitgamma)
    ##    Data: LSAT_long (Number of observations: 5000) 
    ## Samples: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
    ##          total post-warmup samples = 4000
    ## 
    ## Group-Level Effects: 
    ## ~person (Number of levels: 1000) 
    ##                     Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
    ## sd(theta_Intercept)    10.77     11.95     0.35    40.80 1.01     2024     1600
    ## 
    ## ~item (Number of levels: 5) 
    ##                                              Estimate Est.Error l-95% CI
    ## sd(beta_Intercept)                               3.42      5.90     0.59
    ## sd(logalpha_Intercept)                           8.00     10.80     0.03
    ## sd(logitgamma_Intercept)                         8.59     12.10     0.39
    ## cor(beta_Intercept,logalpha_Intercept)          -0.02      0.50    -0.86
    ## cor(beta_Intercept,logitgamma_Intercept)        -0.01      0.50    -0.87
    ## cor(logalpha_Intercept,logitgamma_Intercept)     0.01      0.48    -0.85
    ##                                              u-95% CI Rhat Bulk_ESS Tail_ESS
    ## sd(beta_Intercept)                              20.90 1.36       10       28
    ## sd(logalpha_Intercept)                          34.00 1.46        8       46
    ## sd(logitgamma_Intercept)                        38.55 1.20       14     1587
    ## cor(beta_Intercept,logalpha_Intercept)           0.86 1.00     2610     2379
    ## cor(beta_Intercept,logitgamma_Intercept)         0.88 1.01      509     2229
    ## cor(logalpha_Intercept,logitgamma_Intercept)     0.87 1.01     2666     2506
    ## 
    ## Population-Level Effects: 
    ##                         Estimate  Est.Error    l-95% CI u-95% CI Rhat Bulk_ESS
    ## beta_Intercept       -1080461.54 2156278.89 -6212463.69     2.56 1.61        7
    ## logalpha_Intercept    -562536.96  578685.82 -2091391.00    -0.41 2.26        5
    ## logitgamma_Intercept  -490936.18  608465.07 -2044990.40     2.03 2.54        5
    ##                      Tail_ESS
    ## beta_Intercept             14
    ## logalpha_Intercept         16
    ## logitgamma_Intercept       13
    ## 
    ## Samples were drawn using sampling(NUTS). For each parameter, Bulk_ESS
    ## and Tail_ESS are effective sample size measures, and Rhat is the potential
    ## scale reduction factor on split chains (at convergence, Rhat = 1).
