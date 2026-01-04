import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Utility function for printing
def show_df(title, df):
    print("\n" + "="*60)
    print(title)
    print(df.to_string(index=False))
    print("="*60)

# PART I — DFT (naive implementation)
def kth_fourier_term(signal, k):
    N = len(signal)
    n = np.arange(N)
    angle = 2 * pi * k * n / N

    real = np.sum(signal * np.cos(angle))
    imag = -np.sum(signal * np.sin(angle))  # minus for exp(-jθ)
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

        total_mults += 2 * N       # x*cos, x*sin
        total_adds += 2 * (N - 1)  # for sum

    return C, A, B, total_mults, total_adds


def amplitude_phase(C):
    return np.abs(C), np.angle(C)

# MAIN — PART I
n = 5
N1 = 10 + n
np.random.seed(0)
signal = np.random.randn(N1)

print(f"Variant n = {n}")
print(f"PART I: N = {N1}")

start = time.time()
C1, A1, B1, mults, adds = compute_dft_naive(signal)
elapsed = time.time() - start

amp1, ph1 = amplitude_phase(C1)

df1 = pd.DataFrame({
    "k": np.arange(N1),
    "Re(Ck)": np.round(A1, 6),
    "Im(Ck)": np.round(B1, 6),
    "|Ck|": np.round(amp1, 6),
    "arg(Ck)": np.round(ph1, 6)
})

show_df("PART I: DFT coefficients", df1)
print(f"\nTime: {elapsed:.6f} s | Multiplications: {mults} | Additions: {adds}")

# ---- PLOTS ----
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.stem(np.arange(N1), amp1)
plt.title("Amplitude Spectrum")
plt.grid(True)

plt.subplot(1,2,2)
plt.stem(np.arange(N1), ph1)
plt.title("Phase Spectrum")
plt.grid(True)
plt.tight_layout()
plt.show()

# PART II — 8-bit sequence & DFT

N2 = 96 + n
binary = format(N2, "08b")

# Force MSB based on parity of n
binary = ("1" if n % 2 == 1 else "0") + binary[1:]

samples = np.array([int(b) for b in binary], dtype=float)

print("\nPART II")
print(f"N2 = {N2}, original binary = {format(N2,'08b')}")
print(f"Forced MSB → {binary}")
print("Samples:", samples.tolist())

def compute_dft(signal):
    N = len(signal)
    C = np.zeros(N, dtype=complex)
    n = np.arange(N)
    for k in range(N):
        angle = 2 * pi * k * n / N
        C[k] = np.sum(signal * (np.cos(angle) - 1j * np.sin(angle)))
    return C

C2 = compute_dft(samples)
amp2 = np.abs(C2)
ph2 = np.angle(C2)

df2 = pd.DataFrame({
    "k": np.arange(8),
    "Re(Ck)": np.round(C2.real, 6),
    "Im(Ck)": np.round(C2.imag, 6),
    "|Ck|": np.round(amp2, 6),
    "arg(Ck)": np.round(ph2, 6),
})

show_df("PART II — DFT of 8-bit samples", df2)


def print_analog_formula(C):
    print("\n--- Analytical expression for s(t) ---")
    terms = []
    # N = len(C), у моєму випадку це 8
    for k in range(len(C)):
        coeff = C[k]
        # Форматуємо: (Re + Im*j)
        # Використовуємо .3f для компактності
        re_s = f"{coeff.real:.3f}"
        im_s = f"{coeff.imag:+.3f}j"

        # Формуємо рядок доданку: (A + jB) * e^(j*2*pi*k*t)
        #ділення на 8 (1/N) винесемо за дужки загальної суми
        terms.append(f"({re_s}{im_s})e^(j2π·{k}t)")

    # Об'єднуємо все в одну строку
    full_expr = "s(t) = 1/8 * [" + " + ".join(terms) + "]"
    print(full_expr)

# ---- Reconstruction of continuous signal ----
t = np.linspace(0, 1, 400, endpoint=False)
s_t = np.zeros_like(t, dtype=complex)

for k in range(8):
    s_t += C2[k] * np.exp(1j * 2 * pi * k * t)

s_t /= 8

print("\nАНАЛІТИЧНИЙ ВИРАЗ s(t): ")
print_analog_formula(C2)

plt.figure(figsize=(8,3))
plt.plot(t, s_t.real, label="Re{s(t)}")
plt.plot(t, s_t.imag, "--", label="Im{s(t)}")
plt.title("Reconstructed analog signal s(t)")
plt.grid(True)
plt.legend()
plt.show()

# PART III — IDFT

def compute_idft(C):
    N = len(C)
    m = np.arange(N)
    signal = np.zeros(N, dtype=complex)
    for n in range(N):
        angle = 2 * pi * n * m / N
        signal[n] = np.sum(C * (np.cos(angle) + 1j * np.sin(angle))) / N
    return signal

rec = compute_idft(C2)

print("\nPART III — reconstructed samples via IDFT:")
for i, x in enumerate(rec):
    print(f"s[{i}] = {x.real:.6f} {'+' if x.imag>=0 else '-'} {abs(x.imag):.6f}j")

# Analytical expressions for n=0,1
def analytic_sample(m):
    expr = []
    val = 0j
    for k in range(8):
        term = C2[k] * np.exp(1j * 2 * pi * k * m / 8) / 8
        expr.append(f"({C2[k].real:.3f}{C2[k].imag:+.3f}j)*e^(j2π{k}{m}/8)/8")
        val += term
    return " + ".join(expr), val

expr0, v0 = analytic_sample(0)
expr1, v1 = analytic_sample(1)

print("\nAnalytical expression for s(0):")
print(expr0)
print(f"s(0) = {v0}")

print("\nAnalytical expression for s(1):")
print(expr1)
print(f"s(1) = {v1}")

# Comparison table
df3 = pd.DataFrame({
    "n": np.arange(8),
    "original": samples.astype(int),
    "rec_real": np.round(rec.real, 6),
    "rec_imag": np.round(rec.imag, 6)
})
show_df("Original samples vs IDFT reconstruction", df3)