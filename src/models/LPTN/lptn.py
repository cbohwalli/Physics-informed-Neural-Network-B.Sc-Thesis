import numpy as np

# --- Thermal Network Parameters (Constants) ---
C_Y, C_T, C_W, C_M = 5590.0, 2910.0, 2620.0, 10800.0
R_YW, R_YT, R_TW, R_TM, R_WM = 0.289, 0.013, 0.019, 0.599, 1.149
R_YC, R_MA = 0.017, 2.451

K1_COEFFS = (0.719, -0.059, 0.069, -0.619)
K2_COEFFS = (0.742, -0.213, -0.023, -0.474)

def split_residual(P_res, I, n):
    """Splits residual losses into yoke, tooth, and rotor portions."""
    k1 = np.clip(K1_COEFFS[0] + K1_COEFFS[1]*I + K1_COEFFS[2]*n + K1_COEFFS[3]*I*n, 0, 1)
    k2 = np.clip(K2_COEFFS[0] + K2_COEFFS[1]*I + K2_COEFFS[2]*n + K2_COEFFS[3]*I*n, 0, 1)
    
    Pv_s = k1 * P_res
    Pv_r = (1 - k1) * P_res
    Py = k2 * Pv_s
    Pt = (1 - k2) * Pv_s
    return Py, Pt, Pv_r

def lptn_simulate(Py, Pt, Pw, Ta, Tc, T0, steps, dt=0.5):
    """Euler-integration solver for the 4-node LPTN."""
    Ty, Tt, Tw, Tm = [np.zeros(steps) for _ in range(4)]
    Ty[0], Tt[0], Tw[0], Tm[0] = T0

    for k in range(steps - 1):
        dTy = (Py[k] + (Tt[k]-Ty[k])/R_YT + (Tw[k]-Ty[k])/R_YW + (Tc[k]-Ty[k])/R_YC) / C_Y
        dTt = (Pt[k] + (Ty[k]-Tt[k])/R_YT + (Tw[k]-Tt[k])/R_TW + (Tm[k]-Tt[k])/R_TM) / C_T
        dTw = (Pw[k] + (Ty[k]-Tw[k])/R_YW + (Tt[k]-Tw[k])/R_TW + (Tm[k]-Tw[k])/R_WM) / C_W
        dTm = ((Tt[k]-Tm[k])/R_TM + (Tw[k]-Tm[k])/R_WM + (Ta[k]-Tm[k])/R_MA) / C_M

        Ty[k+1] = Ty[k] + dt * dTy
        Tt[k+1] = Tt[k] + dt * dTt
        Tw[k+1] = Tw[k] + dt * dTw
        Tm[k+1] = Tm[k] + dt * dTm

    return Ty, Tt, Tw, Tm