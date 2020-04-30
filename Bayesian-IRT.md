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
library(cmdstanr)
```

    ## This is cmdstanr version 0.0.0.9000

    ## - Online documentation and vignettes at mc-stan.org/cmdstanr

    ## - CmdStan path set to: C:/Users/Senyo/Documents/.cmdstanr/cmdstan

    ## - Use set_cmdstan_path() to change the path

Using `cmdstanr` to run the Stan code.

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.
