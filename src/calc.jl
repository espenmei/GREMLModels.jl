function hessian(m::GREMLModel)
    q = m.data.dims.q
    hessian!(zeros(eltype(m.θ), q, q), m)
end

function hessian!(H::Matrix, m::GREMLModel)
    θ_opt = copy(m.θ)
    FiniteDiff.finite_difference_hessian!(H, x -> objective(update!(m, x)), θ_opt)
    update!(m, θ_opt)
    H
end

function jacobian(m::GREMLModel)
    FiniteDiff.finite_difference_jacobian(transform, copy(m.θ))
end