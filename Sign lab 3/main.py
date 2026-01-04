import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Utility
def show_df(title, df):
    print("\n" + "=" * 70)
    print(title)
    print(df.to_string(index=False))
    print("=" * 70)

# DFT — Naive Implementation
def kth_fourier_term(signal, k):
    N = len(signal)
    n = np.arange(N)

    angle = 2 * pi * k * n / N
    real = np.sum(signal * np.cos(angle))
    imag = -np.sum(signal * np.sin(angle))

    return real + 1j * imag, real, imag


def compute_dft_naive(signal):
    N = len(signal)
    C = np.zeros(N, dtype=complex)
    A = np.zeros(N)
    B = np.zeros(N)

    total_mults = 0
    total_adds = 0

    for k in range(N):
        Ck, Ak, Bk = kth_fourier_term(signal, k)
        C[k] = Ck
        A[k] = Ak
        B[k] = Bk

        total_mults += 2 * N  # cos + sin multiplications
        total_adds += 2 * (N - 1)  # additions

    return C, A, B, total_mults, total_adds

# FFT — Cooley–Tukey
def fft_recursive(signal, counter):
    N = len(signal)
    if N <= 1:
        return signal

    even = fft_recursive(signal[::2], counter)
    odd = fft_recursive(signal[1::2], counter)

    counter["mult"] += N // 2  # twiddle multiplications
    counter["add"] += N  # additions for butterfly

    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([
        even + factor[:N // 2] * odd,
        even - factor[:N // 2] * odd
    ])


def compute_fft(signal):
    N = len(signal)
    # Padding to nearest power of 2
    M = 1 << (N - 1).bit_length()
    padded = np.zeros(M, dtype=complex)
    padded[:N] = signal

    counter = {"mult": 0, "add": 0}

    C_fft = fft_recursive(padded, counter)

    # Trim to original length
    return C_fft[:N], counter["mult"], counter["add"], M


#   MAIN PROGRAM LOOP (Updated for Instructor's Request)

# Список варіантів n для дослідження (малий і більший розмір)
n_values = [5, 15]

for n in n_values:
    print(f"\n\n{'#' * 30}")
    print(f" PROCESSING VARIANT n = {n}")
    print(f"{'#' * 30}")

    #   Generate data
    N1 = 10 + n
    np.random.seed(0)  # Фіксуємо seed для відтворюваності в кожній ітерації
    signal = np.random.randn(N1)

    print(f"Variant n = {n}")
    print(f"N1 = {N1}")

    #   PART I — DFT
    start = time.time()
    C1, A1, B1, mults_dft, adds_dft = compute_dft_naive(signal)
    t_dft = time.time() - start

    amp1 = np.abs(C1)
    ph1 = np.angle(C1)

    df1 = pd.DataFrame({
        "k": np.arange(N1),
        "Re(Ck)": np.round(A1, 6),
        "Im(Ck)": np.round(B1, 6),
        "|Ck|": np.round(amp1, 6),
        "arg(Ck)": np.round(ph1, 6)
    })

    # Виводимо коефіцієнти, якщо таблиця не надто велика (або завжди)
    show_df(f"DFT coefficients (naive) for n={n}", df1)

    print(f"\nDFT Time: {t_dft:.6f} s | Mult: {mults_dft} | Add: {adds_dft}")

    # ----- plots (DFT only) --------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.stem(np.arange(N1), amp1, markerfmt='C0o', linefmt='C0-')
    plt.title(f"Amplitude — DFT (n={n})")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.stem(np.arange(N1), ph1, markerfmt='C0o', linefmt='C0-')
    plt.title(f"Phase — DFT (n={n})")
    plt.grid(True)

    plt.tight_layout()
    print(f"Close the plot window to proceed with FFT for n={n}...")
    plt.show()

    #   PART II — FFT
    start = time.time()
    C_fft, mults_fft, adds_fft, padded_size = compute_fft(signal)
    t_fft = time.time() - start

    amp_fft = np.abs(C_fft)
    ph_fft = np.angle(C_fft)

    df_fft = pd.DataFrame({
        "k": np.arange(N1),
        "Re(Ck)": np.round(C_fft.real, 6),
        "Im(Ck)": np.round(C_fft.imag, 6),
        "|Ck|": np.round(amp_fft, 6),
        "arg(Ck)": np.round(ph_fft, 6),
    })
    show_df(f"FFT coefficients for n={n}", df_fft)

    print(f"\nFFT Time: {t_fft:.6f} s | Mult: {mults_fft} | Add: {adds_fft} | Padded to: {padded_size}")

    # ------- plots (Comparison)--------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.stem(np.arange(N1), amp1, markerfmt='C0o', linefmt='C0-', label='DFT')
    plt.stem(np.arange(N1), amp_fft, markerfmt='C1x', linefmt='C1--', label='FFT')
    plt.title(f"Amplitude Spectrum — DFT vs FFT (n={n})")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.stem(np.arange(N1), ph1, markerfmt='C0o', linefmt='C0-', label='DFT')
    plt.stem(np.arange(N1), ph_fft, markerfmt='C1x', linefmt='C1--', label='FFT')
    plt.title(f"Phase Spectrum — DFT vs FFT (n={n})")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    print(f"Close the comparison plot window to finish n={n}...")
    plt.show()

    #   PART III — Comparison table
    df_compare = pd.DataFrame({
        "Algorithm": ["DFT (naive)", "FFT (Cooley–Tukey)"],
        "Time (s)": [t_dft, t_fft],
        "Mult operations": [mults_dft, mults_fft],
        "Add operations": [adds_dft, adds_fft],
        "Total operations": [mults_dft + adds_dft, mults_fft + adds_fft],
    })
    show_df(f"Comparison: DFT vs FFT (n={n})", df_compare)