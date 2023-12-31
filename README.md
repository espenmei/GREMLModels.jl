# GREMLModels.jl
**GREMLModels.jl** is a Julia package for fitting variance component models that are structured according to known, dense, relationship matrices.
The intended use case is the analysis of genetic and environmental components of variance for quantitative traits, when genetic relationship matrices are computed from genome-wide SNP data.

## Models
**GREMLModels.jl** can fit models of the form
$$\boldsymbol{y} \sim N(\bf{X} \boldsymbol{\beta}, \bf{V}).$$

In this model $\boldsymbol{\beta}$ is the fixed effects of covariates $\bf{X}$ and $\bf{V}$ is the covariance matrix of the conditional responses. $\bf{V}$ can be modelled with the following structure

$$\bf{V} = \sum_{i=1}^q \delta_i \bf{R}_i$$

where $\delta_i$ are variance component parameters and $\bf{R}_i$ are symmetric matrices that are provided by the user.

The variance component parameters can be defined as functions of the vector of parameters $\bf{\theta}$ that are optimized. This may for instance be needed if $\delta_i$ is a covariance parameter that is bounded by the values of other parameters. This was the motivation for creating this package.

## Installation
The package can be installed from github with
``` julia
(@v1.6) pkg> add https://github.com/espenmei/GREMLModels.jl
```

## Example 1
Here is an example with simulated data showing how the package may be used to fit two variance components and two covariates for the means. The model for the means can be defined using the `@formula` macro together with a `DataFrame` storing the covariates and the response variable. The model for the covariance can be defined by providing a vector with relationship matrices.
```julia
using GREMLModels, LinearAlgebra, DataFrames, StatsModels

n, m = 1000, 2000
v = randn(n, m)
R1 = Symmetric(v * v' / m)
R2 = Diagonal(ones(n))
V = 2R1 + 4R2
y = cholesky(V).L * randn(n)
dat = DataFrame(y = y, x = rand(n), z = rand(n))
r = [R1, R2]
m = fit(GREMLModel, @formula(y ~ 1 + x + z), dat, r)
```
This should give an output reasonably similar to this
```julia
logLik     -2 logLik  AIC        AICc       BIC
-2314.5590 4629.1180  4639.1180  4639.1784  4663.6568  

 Variance component parameters:
Comp.   Est.    Std. Error
θ₁      2.0943  -
θ₂      4.0674  -

 Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)  -0.179071     0.203213  -0.88    0.3782
x            -0.0029327    0.26284   -0.01    0.9911
z             0.435445     0.274538   1.59    0.1127
────────────────────────────────────────────────────
```

Not all arguments are available when fitting a model this way. Sometimes it may be useful to first define a model with the relevant arguments and in a next step to fit that model.

```julia
y = dat[!,:y]
X = Matrix(hcat(ones(n), dat[!,2:3]))
dat = GREMLData(y, X, r)
m2 = GREMLModel(dat, [1.0, 1.0], [0.0, 0.0])
fit!(m2)
```
This first creates an object of type `GREMLData` that is used to hold the data input to a `GREMLModel`. Then a model, represented by an object of type `GREMLModel` is created from a constructor that accepts a `GREMLData`, a vector of initial values for the variance component parameters `[1.0, 1.0]` and a vector of lower bounds for those parameters `[0.0, 0.0]`. And in the last line that model is fitted `fit!(m2)`. The `!` signals that the function modifies the model, which is a convention in julia.

## Example 2
Consider the model for responses obtained from offspring

$$\boldsymbol{y} = \bf{X} \boldsymbol{\beta} + \boldsymbol{m} + \boldsymbol{p} + \boldsymbol{o} + \boldsymbol{\epsilon},$$

where $\boldsymbol{o}$ are direct additive effects of offspring own genes, $\boldsymbol{m}$ and $\boldsymbol{p}$ are indirect effects of maternal and paternal genes and $\boldsymbol{\epsilon}$ errors. Parameters of the model are the variances and covariances among the genetic effects and the error variance. Among individuals the genetic effects are correlated according to their genetic relatedness. Conditional on the covariates, this model has covariance matrix

```math
\bf{V} = \sigma_o^2\bf{A}_{mm} + \sigma_m^2\bf{A}_{pp} + \sigma_p^2\bf{A}_{oo} + \\
\sigma_{mp}(\bf{A}_{mp}+\bf{A}_{pm}) + \sigma_{mo}(\bf{A}_{mo} + \bf{A}_{om}) + \\
\sigma_{po}(\bf{A}_{po}+\bf{A}_{op}) + \sigma_\epsilon^2\bf{I}.
```

The genetic relatedness matrices $\bf{A}_{ij}$ are blocks of the genetic relatedness matrix of all individuals when structured accordingly

```math
\bf{A} = \begin{bmatrix} \bf{A}_{mm} & \bf{A}_{mp} & \bf{A}_{mo}\\
\bf{A}_{pm} & \bf{A}_{pp} & \bf{A}_{po}\\
\bf{A}_{om} & \bf{A}_{op} & \bf{A}_{oo} \end{bmatrix}.
```

In order to fit the model, the first step is to compute the gneetic relationship matrices. The genetic relationships will be estimated from a simulated dataset of 10000 SNPs from 6000 parent-offspring trios stored in *plink* format. The julia package `SnpArrays` is used to read snp-data and compute genetic relationships.

```julia
using VCModels, LinearAlgebra, DataFrames, StatsModels, SnpArrays

gpath = joinpath(@__DIR__, "data", "trio")
trio = SnpData(gpath)
A = 2 * grm(trio.snparray)
```

Because the snp-data was arranged according to families within roles, the genetic relationship matrix can be partitioned as

```julia
K = div(trio.people, 3)
mid = 1:K
pid = K+1:2K
oid = 2K+1:3*K

Amm = A[mid, mid]
App = A[pid, pid]
Aoo = A[oid, oid]
Dpm = A[pid, mid] + A[mid, pid]
Dom = A[oid, mid] + A[mid, oid]
Dop = A[oid, pid] + A[pid, oid] 
```

The off-diagonal blocks are added to compute the expressions within parenthesis in the covariance expression above. These relationship matrices are collected in a vector as before

```julia
R = Diagonal(ones(K))
r = [Amm, App, Aoo, Dpm, Dom, Dop, R]
```

It is also necessary to define a matrix of covariates for the means, which is just a constant here

```julia
X = ones(K)
```
The dataset also contains some simulated responses

```julia
y = parse.(Float64, trio.person_info[oid, :phenotype])
```

A challenge with this model is that some of the variance terms represent covariances, which puts restrictions on which values the parameters can take. If we collect the genetic parameters in a covariance matrix

$$\bf{G} = \begin{bmatrix} \sigma^2_m & \sigma_{mp} & \sigma_{mo}\\
\sigma_{pm} & \sigma^2_p & \sigma_{po}\\
\sigma_{om} & \sigma_{op} & \sigma^2_o \end{bmatrix},$$

this should be positive semidefinite. One way to enforce this is to let the parameters of the model be the elements of the lower-triangular Cholesky factorization $\bf{L}$ of the covariance matrix for genetic effects

$$\bf{L} \bf{L}' = \bf{G}.$$

If $\boldsymbol{\theta}$ is a vector of unique parameters of the model, $\bf{L}$ can be described as

$$\bf{L} = \begin{bmatrix} \theta_1 & 0 & 0\\
\theta_2 & \theta_4 & 0\\
\theta_3 & \theta_5 & \theta_6 \end{bmatrix}.$$

Then

$$\bf{G} = \begin{bmatrix} \sigma^2_m & \sigma_{mp} & \sigma_{mo}\\
\sigma_{pm} & \sigma^2_p & \sigma_{po}\\
\sigma_{om} & \sigma_{op} & \sigma^2_o \end{bmatrix} =
\begin{bmatrix} \theta^2_1 & \theta_1\theta_2 & \theta_1\theta_3\\
\theta_2\theta_1 & \theta^2_2 + \theta^2_4 & \theta_2\theta_3 + \theta_4\theta_5\\
\theta_3\theta_1 & \theta_3\theta_2 + \theta_5\theta_4 & \theta^2_3 + \theta^2_5 + \theta^2_6\end{bmatrix}$$

This can be implemented by overriding the default `VCModels.transform!` method. This is used to define the $q$ dimensional vector of variance component parameters $\delta$ as a function of the $q$ dimensional vector of parameters that are optimized $\theta$. In this case, this could be implemented as

```julia
function GREMLModels.transform!(δ::Vector, θ::Vector)
    δ[1] = θ[1]^2
    δ[2] = θ[2]^2 + θ[4]^2
    δ[3] = θ[3]^2 + θ[5]^2 + θ[6]^2
    δ[4] = θ[2] * θ[1]
    δ[5] = θ[3] * θ[1]
    δ[6] = θ[3] * θ[2] + θ[5] * θ[4]
    δ[7] = θ[7]
end
```

The last element is the residual variance. We can try to fit this model by initializing L to identity and setting the lower bound for the diagonal elements to zero

```julia
dat = GREMLData(y, X, r)
m1 = GREMLModel(dat, [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0], [0.0, -Inf, -Inf, 0.0, -Inf, 0.0, 0.0])
fit!(m1)
```

This should give an output similar to

```julia
logLik      -2 logLik   AIC         AICc        BIC
-17772.7032 35545.4065  35561.4065  35561.4305  35615.0026

 Variance component parameters:
Comp.   Est.    Std. Error
θ₁      1.9214  -
θ₂      0.1848  -
θ₃      1.1346  -
θ₄      -1.6814 -
θ₅      -1.2530 -
θ₆      0.4947  -
θ₇      9.3309  -

 Fixed-effects parameters:
────────────────────────────────────────
      Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────
x1  2.07909   0.0448462  46.36    <1e-99
────────────────────────────────────────
```

Note that the variance components reported in the output are not the variance components in the model, they are the elements of $\bf{L}$. To obtain the variance components we could do

```julia
m1.δ
7-element Vector{Float64}:
 3.6919273370876997
 2.861211241117614
 3.101860361178375
 0.35508630372540173
 2.1799820994344636
 2.3164093450653436
 9.330854714731743
```
