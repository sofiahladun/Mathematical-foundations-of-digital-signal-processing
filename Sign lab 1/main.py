import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from math import pi

# ----- Exact function ----
def exact_function(x, n=5):
    x = np.asarray(x)
    return n * np.sin(pi * n * x)

#---- Coefficients computation (numerical integration using high-resolution grid) ----
def compute_coeffs_numeric(f, N, L=pi, num_points=20001):
    xs = np.linspace(0, L, num_points)
    fx = f(xs)
    # a0
    a0 = (1.0 / L) * np.trapezoid(fx, xs)
    ak = np.zeros(N+1)
    bk = np.zeros(N+1)
    for k in range(1, N+1):
        cos_k = np.cos(k * pi * xs / L)
        sin_k = np.sin(k * pi * xs / L)
        ak[k] = (2.0 / L) * np.trapezoid(fx * cos_k, xs)
        bk[k] = (2.0 / L) * np.trapezoid(fx * sin_k, xs)
    return a0, ak, bk

# ---- Fourier approximation to order N ----
def fourier_approx(x, a0, ak, bk, L=pi):
    x = np.asarray(x)
    y = np.full_like(x, a0/2.0, dtype=float)
    N = len(ak) - 1
    for k in range(1, N+1):
        y += ak[k] * np.cos(k * pi * x / L) + bk[k] * np.sin(k * pi * x / L)
    return y

# ---- Plot harmonics and frequency-domain coefficients ----
def plot_harmonics_and_spectrum(f, a0, ak, bk, L=pi, Nplot=10, num_points=1001, show=True):
    xs = np.linspace(0, L, num_points)
    fx = f(xs)
    # Plot original function and partial sums of individual harmonics (k-th harmonic = ak cos(kx)+bk sin(kx))
    plt.figure(figsize=(10, 5))
    plt.title("Original function and sum of harmonics (partial reconstruction)")
    plt.plot(xs, fx, label='f(x) (exact)')
    # build cumulative sum to show how harmonics add up
    cumulative = np.full_like(xs, a0/2.0, dtype=float)
    for k in range(1, Nplot+1):
        harmonic = ak[k] * np.cos(k * pi * xs / L) + bk[k] * np.sin(k * pi * xs / L)
        cumulative = cumulative + harmonic
        plt.plot(xs, harmonic, linewidth=1, alpha=0.7, label=f'harmonic k={k}')
    plt.plot(xs, cumulative, linewidth=2, linestyle='--', label=f'partial sum N={Nplot}')
    plt.xlabel('x')
    plt.ylabel('value')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    if show:
        plt.show()

    # Frequency domain: stem plot of coefficient magnitudes
    ks = np.arange(0, len(ak))
    magnitudes = np.sqrt(ak**2 + bk**2)
    plt.figure(figsize=(8,4))
    plt.stem(ks, magnitudes)
    plt.xlabel('k (harmonic index)')
    plt.ylabel('magnitude sqrt(ak^2 + bk^2)')
    plt.title('Fourier coefficient magnitudes')
    if show:
        plt.show()

# ---- Relative error (L2 norm over [0,L]) ----
def relative_error_L2(f, approx_func, L=pi, num_points=20001):
    xs = np.linspace(0, L, num_points)
    fx = f(xs)
    gx = approx_func(xs)
    num = np.sqrt(np.trapezoid((fx - gx)**2, xs))
    den = np.sqrt(np.trapezoid(fx**2, xs))
    return num / den if den != 0 else np.inf

# ---- Save results to file (JSON) ----
def save_results(filename, N, a0, ak, bk, rel_error):
    data = {
        "N": int(N),
        "a0": float(a0),
        "ak": {str(k): float(ak[k]) for k in range(1, len(ak))},
        "bk": {str(k): float(bk[k]) for k in range(1, len(bk))},
        "relative_error_L2": float(rel_error)
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filename

# ---- Main program to run the approximation ----
def main_example(n=5, N=10, L=pi):
    print(f"Running Fourier approximation for f(x) = {n} * sin(pi * {n} * x) on [0, {L}] with N={N}\n")
    f = lambda x: exact_function(x, n=n)
    # compute coefficients
    a0, ak, bk = compute_coeffs_numeric(f, N, L=L)
    # build approximation function
    approx = lambda x: fourier_approx(x, a0, ak, bk, L=L)
    # compute relative error
    rel_err = relative_error_L2(f, approx, L=L)
    print(f"Computed a0 = {a0:.6g}")
    for k in range(1, N+1):
        print(f"k={k:2d}: ak={ak[k]:+.6g}, bk={bk[k]:+.6g}")
    print(f"\nRelative L2 error of approximation (N={N}): {rel_err:.6e}\n")
    #plots
    plot_harmonics_and_spectrum(f, a0, ak, bk, L=L, Nplot=min(N,10))
    # overlay original and approximation
    xs = np.linspace(0, L, 1001)
    plt.figure(figsize=(10,4))
    plt.plot(xs, f(xs), label='f(x) exact')
    plt.plot(xs, approx(xs), linestyle='--', label=f'Fourier approx N={N}')
    plt.title('Function vs Fourier approximation')
    plt.xlabel('x')
    plt.ylabel('value')
    plt.legend()
    plt.show()

    # save results
    filename = f'fourier_results_N{N}.json'
    saved = save_results(filename, N, a0, ak, bk, rel_err)
    print(f"Results saved to: {saved}")
    # present coefficients as pandas DataFrame for convenience
    df = pd.DataFrame({
        'k': np.arange(1, N+1),
        'ak': ak[1:N+1],
        'bk': bk[1:N+1],
        'magnitude': np.sqrt(ak[1:N+1]**2 + bk[1:N+1]**2)
    })
    try:
        from caas_jupyter_tools import display_dataframe_to_user
        display_dataframe_to_user("Fourier coefficients", df)
    except Exception:
        display = df.head(20)
        print("\nCoefficients (first rows):\n", display.to_string(index=False))
    return {
        'N': N,
        'a0': a0,
        'ak': ak,
        'bk': bk,
        'relative_error_L2': rel_err,
        'filename': filename
    }

# Run main example with n=5, N=10
results = main_example(n=5, N=10)