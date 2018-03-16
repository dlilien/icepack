"""Functions for calculating the viscosity of a glacier

This module contains procedures for computing the viscosity of a glacier
and, in particular, the viscous part of the action functional for ice flow.
Several flow models all have essentially the same viscous part.
"""

from numpy import exp, zeros
from firedrake import grad, dx, sqrt, Identity, inner, sym, tr as trace
from icepack.constants import year, ideal_gas as R, glen_flow_law as n

transition_temperature = 263.15     # K
A0_cold = 3.985e-13 * year * 1.0e18 # mPa**-3 yr**-1
A0_warm = 1.916e3 * year * 1.0e18
Q_cold = 60                         # kJ / mol
Q_warm = 139


def rate_factor(T):
    """Compute the rate factor in Glen's flow law for a given temperature

    The strain rate :math:`\dot\\varepsilon` of ice resulting from a stress
    :math:`\\tau` is

    .. math::
       \dot\\varepsilon = A(T)\\tau^3

    where :math:`A(T)` is the temperature-dependent rate factor:

    .. math::
       A(T) = A_0\exp(-Q/RT)

    where :math:`R` is the ideal gas constant, :math:`Q` has units of
    energy per mole, and :math:`A_0` is a prefactor with units of
    pressure :math:`\\text{MPa}^{-3}\\times\\text{yr}^{-1}`.
    """
    cold = T < transition_temperature
    A0 = A0_cold if cold else A0_warm
    Q = Q_cold if cold else Q_warm
    return A0 * exp(-Q / (R * T))


def rate_factor_array(T_array):
    """Compute the rate factor in Glen's flow law for a given temperature

    The strain rate :math:`\dot\\varepsilon` of ice resulting from a stress
    :math:`\\tau` is

    .. math::
       \dot\\varepsilon = A(T)\\tau^3

    where :math:`A(T)` is the temperature-dependent rate factor:

    .. math::
       A(T) = A_0\exp(-Q/RT)

    where :math:`R` is the ideal gas constant, :math:`Q` has units of
    energy per mole, and :math:`A_0` is a prefactor with units of
    pressure :math:`\\text{MPa}^{-3}\\times\\text{yr}^{-1}`.
    """
    cold = T_array < transition_temperature
    A0 = zeros(T_array.shape)
    A0[cold] = A0_cold
    A0[~cold] = A0_warm

    Q = zeros(T_array.shape)
    Q[cold] = Q_cold
    Q[~cold] = Q_warm
    return A0 * exp(-Q / (R * T_array))


def M(eps, A):
    """Calculate the membrane stress for a given strain rate and fluidity"""
    I = Identity(2)
    tr = trace(eps)
    eps_e = sqrt((inner(eps, eps) + tr**2) / 2)
    mu = 0.5 * A**(-1/n) * eps_e**(1/n - 1)
    return 2 * mu * (eps + tr * I)


def eps(u):
    """Calculate the strain rate for a given flow velocity"""
    return sym(grad(u))


def viscosity_depth_averaged(u=None, h=None, A=None):
    """Return the viscous part of the action for depth-averaged models

    The viscous component of the action for depth-averaged ice flow is

    .. math::
        E(u) = \\frac{n}{n+1}\int_\Omega h\cdot
        M(\dot\\varepsilon, A):\dot\\varepsilon\hspace{2pt} dx

    where :math:`M(\dot\\varepsilon, A)` is the membrane stress tensor

    .. math::
        M(\dot\\varepsilon, A) = A^{-1/n}|\dot\\varepsilon|^{1/n - 1}
        (\dot\\varepsilon + \\text{tr}\dot\\varepsilon\cdot I).

    This form assumes that we're using the fluidity parameter instead
    the rheology parameter, the temperature, etc. To use a different
    variable, you can implement your own viscosity functional and pass it
    as an argument when initializing model objects to use your functional
    instead.

    Parameters
    ----------
    u : firedrake.Function
        ice velocity
    h : firedrake.Function
        ice thickness
    A : firedrake.Function
        ice fluidity parameter

    Returns
    -------
    firedrake.Form
    """
    return n/(n + 1) * h * inner(M(eps(u), A), eps(u)) * dx
