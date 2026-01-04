# lab4_fourier_n5_improved.py
import numpy as np
import matplotlib.pyplot as plt
import os

n = 5
N = 100
f = lambda t: t**(2 * n)       # t^10

T_list = [4, 8, 16, 32, 64, 128]
K_max = 200
num_grid = 20001
out_dir = "lab4_results_n5_improved"
os.makedirs(out_dir, exist_ok=True)

t = np.linspace(-N, N, num_grid)
dt = t[1] - t[0]
ft = f(t)

def compute_F(omegas):
    F_vals = np.zeros(len(omegas), dtype=np.complex128)
    for i, w in enumerate(omegas):
        integrand = ft * np.exp(-1j * w * t)
        F_vals[i] = np.trapezoid(integrand, t)
    return F_vals


for T in T_list:
    ks = np.arange(0, K_max + 1)
    omegas = 2 * np.pi * ks / T

    # Обчислення
    Fk = compute_F(omegas)

    # Нормалізація
    F0_abs = np.abs(Fk[0]) if np.abs(Fk[0]) > 0 else 1.0
    Fk_norm = Fk / F0_abs

    AbsFk = np.abs(Fk_norm)

    # --- ГРАФІК 1: Нормований спектр (Лінійна шкала) ---
    plt.figure(figsize=(8, 4.5))
    plt.plot(ks, AbsFk, marker='.', linewidth=0.6)

    if T in [4, 8, 16]:
        plt.xlim(0, 50)  # Зменшуємо інтервал для деталізації
        plt.title(f'Нормований спектр |F(w_k)| (ZOOM 0-50) для T={T}, n={n}, N={N}')
    else:
        plt.xlim(0, K_max)  # Для інших залишаємо повний діапазон
        plt.title(f'Нормований спектр |F(w_k)| для T={T}, n={n}, N={N}')

    plt.xlabel('k')
    plt.ylabel(r'|F($\omega_k$)| / |F(0)|')
    plt.grid(True)
    plt.tight_layout()

    fname_lin = os.path.join(out_dir, f"AbsF_norm_T{T}_n{n}.png")
    plt.savefig(fname_lin, dpi=200)
    plt.close()

    # --- ГРАФІК 2: Логарифмічна шкала ---
    eps = 1e-30
    AbsFk_db = 20 * np.log10(AbsFk + eps)

    plt.figure(figsize=(8, 4.5))
    plt.plot(ks, AbsFk_db, marker='.', linewidth=0.6)


    if T in [4, 8, 16]:
        plt.xlim(0, 50)
    else:
        plt.xlim(0, K_max)

    plt.xlabel('k')
    plt.ylabel(r'20*log10(|F($\omega_k$)|/|F(0)|)')
    plt.title(f'Лог шкала спектра для T={T}, n={n}, N={N}')
    plt.grid(True)
    plt.tight_layout()

    fname_db = os.path.join(out_dir, f"AbsF_dB_T{T}_n{n}.png")
    plt.savefig(fname_db, dpi=200)
    plt.close()

    print(f"T={T}: збережено {fname_lin} та {fname_db}")

print("Готово. Перевірте теку:", out_dir)