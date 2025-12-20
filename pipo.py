import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def active(t, windows):
    return any(a <= t < b for a, b in windows)

def write_all(t, windows):
    return active(t, windows)

def write_i(t, i, per_bit_windows):
    
    w = per_bit_windows.get(i, [])
    return active(t, w)



def hill(x, Kd, n):
    x = 0.0 if x < 0 else float(x)
    return (x / Kd) ** n

def ff_core(y4, d, clk, params):
    a, not_a, q, not_q = y4
    alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n = params

    D   = hill(d, Kd, n)
    CLK = hill(clk, Kd, n)
    A   = hill(a, Kd, n)
    NA  = hill(not_a, Kd, n)
    Q   = hill(q, Kd, n)
    NQ  = hill(not_q, Kd, n)

    da     = alpha2 * (D / (1 + D + CLK + D*CLK)) + alpha2 * (1 / (1 + NA)) - delta1 * a
    dnot_a = alpha1 * (1 / (1 + D + CLK + D*CLK)) + alpha2 * (1 / (1 + A))  - delta1 * not_a
    dq     = alpha3 * ((A*CLK) / (1 + A + CLK + A*CLK)) + alpha4 * (1 / (1 + NQ)) - delta2 * q
    dnot_q = alpha3 * ((NA*CLK) / (1 + NA + CLK + NA*CLK)) + alpha4 * (1 / (1 + Q))  - delta2 * not_q

    return np.array([da, dnot_a, dq, dnot_q], dtype=float)



def clock_fn(t):
    A = 50.0
    period = 50.0
    return A * (np.sin(2 * np.pi * t / period) + 1.0)

def din_fn(t):
    D0 = 50.0 if t < 200 else 0.0
    D1 = 0.0 if (300 <= t < 400) else 50.0
    D2 = 0.0 if (300 <= t < 400) else 50.0
    D3 = 0.0 if (300 <= t < 400) else 50.0
    return [D0, D1, D2, D3]



N_BITS = 4

WRITE_ALL_WINDOWS = [(0, 100), (300, 400)]
WRITE_BIT_WINDOWS = {
    0: [(200, 300)],   # bit0 enabled 200..300
    # 1: [(...), (...)],
    # 2: [...],
    # 3: [...],
}


def pipo(t, y, params):
    clk = clock_fn(t)
    D = din_fn(t)

    wa = write_all(t, WRITE_ALL_WINDOWS)
    dy = np.zeros_like(y)

    for i in range(N_BITS):
        base = 4 * i
        y4 = y[base:base + 4]
        q_i = y4[2]

        en_i = wa or write_i(t, i, WRITE_BIT_WINDOWS)
        d_i = D[i] if en_i else q_i  # mux hold
        dy[base:base + 4] = ff_core(y4, d_i, clk, params)

    return dy



t0, t1 = 0.0, 400.0
T_eval = np.linspace(t0, t1, 2000)

params_ff = (90.0, 20.0, 90.0, 20.0, 1.23, 0.30, 7.46, 4.0)
y0 = np.array([0, 50, 0, 50] * N_BITS, dtype=float)

sol = solve_ivp(
    fun=lambda t, y: pipo(t, y, params_ff),
    t_span=(t0, t1),
    y0=y0,
    t_eval=T_eval,
    max_step=0.5,
    rtol=1e-6,
    atol=1e-9
)

t = sol.t
Y = sol.y.T

Q = np.column_stack([Y[:, 4*i + 2] for i in range(N_BITS)])
NQ = np.column_stack([Y[:, 4*i + 3] for i in range(N_BITS)])

CLK = np.array([clock_fn(tt) for tt in t])
D = np.array([din_fn(tt) for tt in t])

WRITE_ALL = np.array([50.0 if write_all(tt, WRITE_ALL_WINDOWS) else 0.0 for tt in t])
WRITE_BITS = np.column_stack([
    np.array([50.0 if write_i(tt, i, WRITE_BIT_WINDOWS) else 0.0 for tt in t])
    for i in range(N_BITS)
])

bits = (Q > NQ).astype(float)



plt.figure(figsize=(12, 10))

plt.subplot(6, 1, 1)
plt.plot(t, CLK, "--", label="CLK", alpha=0.4)
plt.plot(t, WRITE_ALL, label="WRITE_ALL")
for i in range(N_BITS):
    plt.plot(t, WRITE_BITS[:, i], label=f"WRITE_{i}")
plt.legend(loc="right")
plt.title("Controls")

for i in range(N_BITS):
    plt.subplot(6, 1, 2 + i)
    plt.plot(t, Q[:, i], label=f"Q{i}")
    plt.plot(t, D[:, i], "--", label=f"D{i}")
    plt.legend(loc="right")

plt.subplot(6, 1, 6)
for i in range(N_BITS):
    plt.plot(t, bits[:, i], label=f"bit{i}")
plt.ylim(-0.1, 1.1)
plt.title("Digital readout (Q > not-Q)")
plt.legend(loc="right")

plt.tight_layout()
plt.show()
