module GREMLModels

using Base: Ryu
using DataFrames: DataFrame
using Distributions
using FiniteDiff
using LinearAlgebra
using NLopt
using StatsAPI
using StatsBase
using StatsModels

# Things to override
import StatsBase: fit, fit! # Can now call fit, fit! directly
import LinearAlgebra: dot, logabsdet

export
    # Structs/constructors 
    GREMLData,
    GREMLModel,
    GREMLOpt,
    # Computations
    update!,
    setθ!,
    updateΛ!,
    updateμ!,
    objective,
    fit,
    fit!,
    hessian,
    hessian!,
    jacobian,
    jacobian!,
    jac,
    transform,
    # Utilities
    fixef,
    fixef!,
    ranef,
    ranef!,
    vcov,
    vcovvc,
    vcovvctr,
    stderror,
    stderrorvc,
    # Implements
    coef,
    deviance,
    dof,
    dof_residual,
    loglikelihood,
    modelmatrix,
    nobs,
    response,
    isnested

# Order matters!
include("optimization.jl")
include("gremlmodel.jl")
#include("algorithms.jl")
include("calc.jl")
include("linalg.jl")
#include("fileIO.jl")

end # module