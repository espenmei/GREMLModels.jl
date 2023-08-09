function hessian(m::GREMLModel)
    q = m.data.dims.q
    hessian!(zeros(eltype(m.θ), q, q), m)
end

function hessian!(H::Matrix, m::GREMLModel)
    function obj(x::Vector)
        val = objective(update!(m_tmp, x))
        #showiter(m_tmp.opt)
        val
    end
    m_tmp = deepcopy(m) # Finitediff kødder med med m under vurdering, så lag en kopi av alt og la den kødde der
    # Bedre å lage en ny modell?
    # Det er mulig FD har en funksjon som ikke overskriver
    FiniteDiff.finite_difference_hessian!(H, obj, copy(m.θ))
    H
end

function jacobian(m::GREMLModel)
    FiniteDiff.finite_difference_jacobian(transform, copy(m.θ))
end

# Dette er latterlig. Krev heller at transform tar en vector så slipper du å lage en kopi av hele modellen
# Det er ikke alltid at L vil være q x q, feks ved sum av paremetere.
#function jacobian(m::VCModel)
#    q = m.data.dims.q
#    jacobian!(zeros(eltype(m.θ), q, q), m)
#end

#function jacobian!(J::Matrix, m::VCModel)
#    function f(θ::Vector)
#        update!(m_tmp, θ)
#        transform(m_tmp)
#    end
#    m_tmp = deepcopy(m)
#    J .= FiniteDiff.finite_difference_jacobian(f, copy(m.θ))
#    #FiniteDiff.finite_difference_jacobian!(J, f, copy(m.θ))
#    J
#end