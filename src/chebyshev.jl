# Shared by Levin collocation and supplied-moment quadrature. Coordinates run
# from right to left; order is the polynomial degree, so there are order+1 roots.
chebyshev_gauss_nodes(order, ::Type{T}=Float64) where {T} =
    [cospi(T(2*j + 1) / T(2*(order+1))) for j in 0:order]

function from_chebyshev_coordinate(coordinate, interval)
    left, right = interval
    return left + (right-left) * (coordinate+1)/2
end

function to_chebyshev_coordinate(x, interval)
    left, right = interval
    return 2 * (x-left) / (right-left) - 1
end

# Clenshaw evaluation of sum(coefficients[k+1]*T_k(coordinate)); the constant
# coefficient is used in full. Initializing with the product preserves promotion
# when, for example, Float64 coefficients are evaluated at a BigFloat coordinate.
function chebyshev_value(coefficients, coordinate)
    next_value = zero(coefficients[1] * coordinate)
    following_value = next_value
    for k in length(coefficients):-1:2
        current = coefficients[k] + 2 * coordinate * next_value - following_value
        following_value, next_value = next_value, current
    end
    return coefficients[1] + coordinate * next_value - following_value
end
