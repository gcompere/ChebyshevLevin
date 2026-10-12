using Test

# A different weight and shifted interval exercise the supplied-moment
# interface without Kerr physics or a Levin equation.
using ChebyshevLevin.Moments

@testset "Generic supplied-moment integration" begin
    left,right=-0.3,0.7
    midpoint,half_width=(left+right)/2,(right-left)/2
    plain_moment(k)=isodd(k) ? 0. : (right-left)/(1-k*k)
    # Weight 1+x: x=midpoint+half_width*t and t*T_k=(T_(k+1)+T_|k-1|)/2.
    moments(order)=[(1+midpoint)*plain_moment(k)+
        half_width*(plain_moment(k+1)+plain_moment(abs(k-1)))/2 for k in 0:order]
    rate=2+3im
    problem=MomentProblem(x->exp(rate*x),(left,right),moments;
        weight_norm=(right-left)*(1+midpoint),orders=(2,4,8,16,24))
    primitive(x)=exp(rate*x)*((1+x)/rate-1/rate^2)
    first_estimate=integrate_moments(problem;atol=1e-12,rtol=0.,max_refinements=0)
    result=integrate_moments(problem;atol=1e-12,rtol=0.)
    @test !first_estimate.tolerance_met
    @test result.tolerance_met
    @test result.value ≈ primitive(right)-primitive(left) atol=1e-12 rtol=0
    refined=moment_estimate(problem;atol=1e-12,rtol=0.,previous=first_estimate)
    @test refined.probes === first_estimate.probes
    @test refined.low_value == first_estimate.value
    @test refined.high_order == 8
    @test_throws ArgumentError integrate_moments(problem;max_refinements=-1)
    @test_throws ArgumentError moment_estimate(problem;atol=-1.)
    @test_throws ArgumentError MomentProblem(identity,(1.,0.),moments;weight_norm=1.)
    @test_throws ArgumentError MomentProblem(identity,(0.,1.),moments;weight_norm=-1.)
    bad=MomentProblem(identity,(0.,1.),order->fill(NaN,order+1);weight_norm=1.)
    @test_throws ArgumentError integrate_moments(bad)
end

@testset "High-degree inverse-phase precision" begin
    # Degree 64 in the automatic near-critical regime (abs(beta)/b <= 1).
    # Large detuning is already covered at degree 16 below;
    # 1024-bit expint at large imaginary argument is too costly for this check.
    for beta in (-0.3,1e-6,0.3)
        weights=inverse_phase_weights(beta,.31,64)
        precise=inverse_phase_weights(beta,.31,64;precision_bits=1024)
        @test weights ≈ precise atol=2e-14 rtol=0
    end
end

@testset "Inverse-phase moments: exact primitives and precision" begin
    b=.31
    for beta in (-30.,-.3,-1e-6,0.,1e-6,.3,30.)
        weights=Moments.inverse_phase_weights(beta,b,16)
        for q in (2,3,6,12)
            H(x)=q*x^(q-1)+im*beta*x^(q-2)
            coeffs=Moments.chebyshev_coefficients(H,(0.,b),16)
            @test sum(coeffs.*weights) ≈ b^q*cis(-beta/b) atol=2e-13 rtol=2e-13
        end
        @test weights ≈ Moments.inverse_phase_weights(beta,b,16;precision_bits=512) atol=2e-14 rtol=0
    end
    weights=Moments.inverse_phase_weights(0.,b,32)
    for k in 0:32
        @test weights[k+1] ≈ (isodd(k) ? 0. : b/(1-k*k)) atol=1e-14
    end
end
