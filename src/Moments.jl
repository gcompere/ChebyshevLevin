"""
Numerical integration against supplied Chebyshev moments. This backend depends
on the shared Chebyshev utilities and SpecialFunctions. It knows no Kerr
geometry or radial domain and does not solve a Levin ODE. `MomentProblem` is the generic boundary;
`inverse_phase_weights` supplies one moment family. Physics and solver selection
belong to callers.
"""
module Moments

# Direct dependency: expint must retain Complex{BigFloat} guard precision here.
# Relying on another package's transitive import would not declare this API use;
# replacing it with Float64 expint would discard the cancellation margin below.
import SpecialFunctions
using ..ChebyshevLevin: chebyshev_gauss_nodes, from_chebyshev_coordinate, chebyshev_value
export MomentProblem, MomentEstimate, moment_estimate, integrate_moments, inverse_phase_weights
const DEFAULT_ORDERS = (8,12,16,24,32,48,64)
const PRECISION_LOCK = ReentrantLock()

"""
    MomentProblem(amplitude, (left,right), moments; weight_norm, orders=DEFAULT_ORDERS)

Represent `integral amplitude(x)*weight(x) dx` on a finite interval.
`moments(n)` returns `n+1` integrals of
`T_k(2*(x-left)/(right-left)-1)*weight(x)`, k=0:n, including the Jacobian.
`weight_norm` bounds `integral abs(weight(x)) dx`. Only the smooth amplitude is
sampled: the weight and its oscillations enter through the supplied moments.
Callbacks must return finite values convertible to ComplexF64. Errors in the
amplitude or moment callbacks are not estimated by this interface.
"""
struct MomentProblem{F,W,O}
    amplitude::F
    interval::Tuple{Float64,Float64}
    moments::W
    weight_norm::Float64
    orders::O
end

function MomentProblem(amplitude, interval::Tuple{Real,Real}, moments;
                       weight_norm::Real, orders=DEFAULT_ORDERS)
    left,right=Float64.(interval)
    isfinite(left) && isfinite(right) && left<right || throw(ArgumentError("Invalid moment interval."))
    isfinite(weight_norm) && 0<=weight_norm<=floatmax(Float64) ||
        throw(ArgumentError("Expected a finite nonnegative weight norm."))
    degrees=Tuple(orders)
    length(degrees)>=2 && all(n->n isa Integer && 0<=n<=256,degrees) &&
        all(diff(collect(degrees)).>0) || throw(ArgumentError("Supply at least two increasing integer degrees in 0:256."))
    return MomentProblem(amplitude,(left,right),moments,Float64(weight_norm),degrees)
end

"""
A numerical integral and its refinement state, independent of physical domains.
The error estimate is the maximum of degree change, sampled interpolation
residual times `weight_norm`, and a roundoff indicator. The sampled residual is
not a supremum bound: narrow unsampled features can escape it. Callback errors
are separate. No auxiliary endpoint solution or physical fallback is stored.
"""
struct MomentEstimate
    value::ComplexF64
    low_value::ComplexF64
    error_estimate::Float64
    tolerance::Float64
    low_order::Int
    high_order::Int
    tolerance_met::Bool
    degree_index::Int
    interpolation_error::Float64
    roundoff_error::Float64
    probes::Vector{ComplexF64}
end

"""
    inverse_phase_weights(beta, b, order; precision_bits=256)

Return `integral_0^b T_k(2x/b-1)*exp(-im*beta/x) dx`, k=0:order, for real beta.
Set u=x/b and v=1/u. For z=i*beta/b,
`integral_0^1 u^j exp(-z/u) du = integral_1^Inf exp(-z*v)/v^(j+2) dv = E_(j+2)(z)`.
The monomial moment on [0,b] is therefore `b^(j+1)*E_(j+2)(z)`; for a polynomial
in u the integral is `b*sum(p[j+1]*E_(j+2)(z))`. At beta=0 use `1/(j+1)`.
The shifted Chebyshev recurrence is `P_(k+1)=(4u-2)P_k-P_(k-1)`.

Independent exponential-integral calls avoid unstable upward moment recurrence.
Shifted-polynomial coefficient 1-norms grow as T_k(3), approximately
`(3+sqrt(8))^k/2`, costing about 2.55*k bits in cancellation. The retained rule
`max(precision_bits,8*order+128)` exceeds that growth and leaves a margin above
53-bit output precision. This is conservative working precision, not an error
certificate for SpecialFunctions; tests compare with higher precision. Round
only final weights to ComplexF64. The lock serializes this module's precision
scopes; unrelated BigFloat users must manage their own precision scopes.
"""
function inverse_phase_weights(beta::Real,b::Real,order::Integer;precision_bits::Integer=256)
    isfinite(Float64(beta)) && isfinite(Float64(b)) && Float64(b)>0 ||
        throw(ArgumentError("Expected finite beta and positive b representable in Float64."))
    0<=order<=256 && precision_bits>=128 || throw(ArgumentError("Invalid degree or guard precision."))
    lock(PRECISION_LOCK) do
        setprecision(BigFloat,max(precision_bits,8order+128)) do
            z=Complex{BigFloat}(0,BigFloat(beta)/BigFloat(b))
            moments=[iszero(z) ? Complex{BigFloat}(inv(BigFloat(k+1))) :
                     SpecialFunctions.expint(k+2,z) for k in 0:order]
            weights=ComplexF64[]
            previous=BigFloat[1]
            push!(weights,ComplexF64(BigFloat(b)*moments[1]))
            order==0 && return weights
            current=BigFloat[-1,2]
            for k in 1:order
                push!(weights,ComplexF64(BigFloat(b)*sum(current[j]*moments[j] for j in eachindex(current))))
                k==order && break
                following=zeros(BigFloat,k+2)
                for j in eachindex(current)
                    following[j]-=2current[j]
                    following[j+1]+=4current[j]
                end
                following[1:length(previous)].-=previous
                previous,current=current,following
            end
            return weights
        end
    end
end

function chebyshev_coefficients(amplitude,interval,order)
    node_count=order+1
    angles=[pi*(j+.5)/node_count for j in 0:order]
    # Interior roots avoid endpoints. This direct cosine transform interpolates
    # the amplitude only; halving coefficient 0 gives the full-constant convention
    # shared with Clenshaw. The oscillatory weight enters through the moments.
    nodes=chebyshev_gauss_nodes(order)
    values=ComplexF64[amplitude(from_chebyshev_coordinate(t,interval)) for t in nodes]
    all(isfinite,values) || throw(ArgumentError("Nonfinite smooth amplitude."))
    coefficients=[2/node_count*sum(values[j]*cos(k*angles[j]) for j in 1:node_count) for k in 0:order]
    coefficients[1]/=2
    return coefficients
end

# Eighteen fixed, symmetric interior probes are reused between degrees. The
# offset (approximately pi/10) avoids interpolation-root and midpoint coincidences
# across the default ladder. Eighteen probes are an empirical cost/coverage choice,
# not a sampling theorem; narrow features or aliasing can escape this check.
probe_nodes() = [(j+.314159)/(17+.628318) for j in 0:17]

function estimate_order(problem,order,probes)
    coefficients=chebyshev_coefficients(problem.amplitude,problem.interval,order)
    weights=problem.moments(order)
    length(weights)==order+1 && all(isfinite,weights) || throw(ArgumentError("Invalid supplied moments."))
    terms=coefficients.*weights
    value=sum(terms)
    residual=problem.weight_norm*maximum(abs(probes[j]-chebyshev_value(coefficients,2t-1))
                                        for (j,t) in enumerate(probe_nodes()))
    # Term magnitudes expose cancellation hidden by abs(value). 32 eps is the
    # retained empirical safety margin for coefficient/sum rounding at default
    # degrees 8..64, not a rigorous forward-error bound or a callback error bound.
    roundoff=32eps(Float64)*sum(abs,terms)
    isfinite(value) && isfinite(residual) && isfinite(roundoff) || throw(ArgumentError("Nonfinite moment estimate."))
    return (;value,residual,roundoff)
end

"""
    moment_estimate(problem; atol=1e-12, rtol=1e-8, previous=nothing)

Compare the first two degrees, or advance one degree with `previous` from this
same problem. Reuse the previous integral and independent probes. No physical
fallback is chosen. Use `integrate_moments` for a complete adaptive calculation.

For amplitude H, weight w, interpolant p_N = sum(c_k*T_k), supplied moments
mu_k = integral(T_k*w), and Q_N = sum(c_k*mu_k), exact moments give the bound
`abs(integral(H*w)-Q_N) <= W * norm(H-p_N, Inf)`, provided
`W = weight_norm >= integral(abs(w))`. The caller supplies W; it cannot be
inferred from the finite moment vector. This bound concerns interpolation only.

The implementation instead uses the sampled diagnostic
`max(abs(Q_N-Q_M), W*maximum(abs(H(x_j)-p_N(x_j))), 32*eps(Float64)*sum(abs(c_k*mu_k)))`.
Here M is the previous degree and N the current degree. Successive-degree
agreement is useful for a smooth amplitude whose Chebyshev coefficients decay,
but both degrees can share an error. The 18 probe coordinates are
`x_j = left + (right-left)*(j+0.314159)/(17+0.628318)`, j=0:17.
They lie strictly inside the interval, are symmetric and avoid the default
interpolation roots and midpoint. Their number and offset are empirical choices
for an inexpensive check, not a rigorous bound on the unsampled supremum.

The absolute product sum measures sensitivity to cancellation, unlike abs(Q_N).
The factor 32 is the retained empirical roundoff margin for the default degrees
8..64. It is not a degree-dependent error bound and does not certify coefficient
transforms or supplied moments, especially for custom higher degrees. Taking the
maximum combines three warning indicators; these are not separately bounded
error contributions to be added. Unresolved narrow features or aliasing can fool
both degree comparison and probes. Amplitude/phase evaluation errors and moment
callback errors are not independently bounded by any of these terms.

`tolerance_met` means only that this diagnostic is at most
`max(atol, rtol*abs(Q_N))`. It certifies neither the true quadrature error nor the
accuracy of a physical waveform assembled by a caller.
"""
function moment_estimate(problem::MomentProblem;atol=1e-12,rtol=1e-8,previous=nothing)
    isfinite(atol) && atol>=0 && isfinite(rtol) && rtol>=0 || throw(ArgumentError("Invalid moment tolerances."))
    if previous===nothing
        left,right=problem.interval
        probes=ComplexF64[problem.amplitude(left+(right-left)*t) for t in probe_nodes()]
        all(isfinite,probes) || throw(ArgumentError("Nonfinite validation probes."))
        degree_index=2
        low_value=estimate_order(problem,problem.orders[1],probes).value
    else
        previous.degree_index<length(problem.orders) || throw(ArgumentError("Moment degree ladder is exhausted."))
        probes=previous.probes
        degree_index=previous.degree_index+1
        low_value=previous.value
    end
    high=estimate_order(problem,problem.orders[degree_index],probes)
    error_estimate=max(abs(high.value-low_value),high.residual,high.roundoff)
    tolerance=max(atol,rtol*abs(high.value))
    return MomentEstimate(high.value,low_value,error_estimate,tolerance,
        problem.orders[degree_index-1],problem.orders[degree_index],error_estimate<=tolerance,
        degree_index,high.residual,high.roundoff,probes)
end

"""
    integrate_moments(problem; atol=1e-12, rtol=1e-8, max_refinements=length(problem.orders)-2)

Refine until the empirical check passes or the degree/refinement limit is reached.
Return a `MomentEstimate`; exhausted work can leave `tolerance_met=false`.
"""
function integrate_moments(problem::MomentProblem;atol=1e-12,rtol=1e-8,
                           max_refinements::Integer=length(problem.orders)-2)
    max_refinements>=0 || throw(ArgumentError("max_refinements must be nonnegative."))
    estimate=moment_estimate(problem;atol,rtol)
    for _ in 1:min(max_refinements,length(problem.orders)-2)
        estimate.tolerance_met && break
        estimate=moment_estimate(problem;atol,rtol,previous=estimate)
    end
    return estimate
end

end # module Moments
