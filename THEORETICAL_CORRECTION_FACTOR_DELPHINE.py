import math
def expr(x_0, m, n, h):
    """Computes (2/pi) * atan( e^{-πx0/h} (e^{πn/h} - e^{πm/h}) / (1 + e^{-2πx0/h + π(m+n)/h}) )."""
    pi = math.pi
    numerator = math.exp(-pi * x_0 / h) * (math.exp(pi * n / h) - math.exp(pi * m / h))
    denominator = 1.0 + math.exp(-2 * pi * x_0 / h) * math.exp(pi * (m + n) / h)
    return (2.0 / pi) * math.atan(numerator / denominator)

from typing import Callable, Sequence

# ---- Weight (constant prefactor cancels in the ratio) ----
def w(x: float, h: float) -> float:
    """Weight proportional to C_{P,R}(x): sech(pi*x/h)."""
    return 1.0 / math.cosh(math.pi * x / h)

# ---- A) Functional form of η_x(x) ----
def eta_weighted_function(eta_x: Callable[[float], float], c: float, h: float, N: int = 5000) -> float:
    """
    Computes η = ∫_0^1 C_{P,R}(c*ξ) η_x(c*ξ) dξ / ∫_0^1 C_{P,R}(c*ξ) dξ
    using trapezoidal integration with N panels.
    Only the shape w(c*ξ)=sech(pi*c*ξ/h) is needed (prefactor cancels).
    """
    if N < 2:
        raise ValueError("N must be >= 2")
    dxi = 1.0 / (N - 1)
    num = 0.0
    den = 0.0
    for i in range(N):
        xi = i * dxi               # xi in [0,1]
        x  = c * xi                # physical x along chord
        wi = w(x, h)
        fi = eta_x
        # trapezoid weights
        trap_w = 0.5 if (i == 0 or i == N-1) else 1.0
        num += trap_w * wi * fi
        den += trap_w * wi
    num *= dxi
    den *= dxi
    return num / den

# ---- B) Discrete samples (x/c and η_x) ----
def eta_weighted_samples(x_over_c: Sequence[float], eta_vals: Sequence[float], c: float, h: float) -> float:
    """
    Same formula but with discrete samples. x_over_c must be sorted in [0,1].
    Uses trapezoidal rule over the provided grid.
    """
    if len(x_over_c) != len(eta_vals):
        raise ValueError("x_over_c and eta_vals must have the same length")
    if len(x_over_c) < 2:
        raise ValueError("Need at least two sample points")

    num = 0.0
    den = 0.0
    for i in range(len(x_over_c) - 1):
        xi0, xi1 = x_over_c[i], x_over_c[i+1]
        if not (0.0 <= xi0 <= 1.0 and 0.0 <= xi1 <= 1.0):
            raise ValueError("x_over_c values must be in [0,1]")
        x0, x1 = c * xi0, c * xi1
        w0, w1 = w(x0, h), w(x1, h)
        f0, f1 = eta_vals[i], eta_vals[i+1]
        dxi = xi1 - xi0
        # trapezoid on the mapped interval
        num += 0.5 * dxi * (w0 * f0 + w1 * f1)
        den += 0.5 * dxi * (w0 + w1)
    return num / den



# ------------------ Example usage ------------------
if __name__ == "__main__":

        # --- Set your values here ---
    x_0 = 0
    m   = 1.2355
    n   = 1.17375
    h   = 1.7
    # ----------------------------

    value = expr(x_0, m, n, h)
    print(value)

    # Parameters
    c = .5     # chord length
   

    # A) Functional: define η_x(x). Example: a gentle linear variation along the chord.
    def eta_x_func(x: float) -> float:
        xi = x / c
        return 0.8 + 0.2 * (1 - xi)  # replace with your own η_x(x)

    eta_func = eta_weighted_function(value, c=c, h=h, N=4001)
    print(f"eta (functional) = {eta_func:.10f}")

    # B) Discrete samples: suppose you have arrays of x/c and η_x
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    et = [1.0, 0.95, 0.90, 0.88, 0.85]
    eta_samp = eta_weighted_samples(xs, et, c=c, h=h)
    print(f"eta (samples)    = {eta_samp:.10f}")