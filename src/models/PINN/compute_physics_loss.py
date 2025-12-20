import torch

# Thermal constants (SI units)
C_Y, C_T, C_W, C_M = 5590.0, 2910.0, 2620.0, 10800.0
R_YW, R_YT, R_TW, R_TM, R_WM = 0.289, 0.013, 0.019, 0.599, 1.149
R_YC, R_MA = 0.017, 2.451

# Physics coefficients for residual splitting
K1_COEFFS = (0.719, -0.059, 0.069, -0.619)
K2_COEFFS = (0.742, -0.213, -0.023, -0.474)

def torch_split_residual(P_res, I, n):
    """Splits residual losses using PyTorch tensors for backprop."""
    k1 = K1_COEFFS[0] + K1_COEFFS[1]*I + K1_COEFFS[2]*n + K1_COEFFS[3]*I*n
    k2 = K2_COEFFS[0] + K2_COEFFS[1]*I + K2_COEFFS[2]*n + K2_COEFFS[3]*I*n

    k1 = torch.clamp(k1, 0.0, 1.0)
    k2 = torch.clamp(k2, 0.0, 1.0)

    Pv_s = k1 * P_res
    Pv_r = (1.0 - k1) * P_res
    Py = k2 * Pv_s
    Pt = (1.0 - k2) * Pv_s

    return Py, Pt, Pv_r

def compute_physics_loss(T_pred, P_seq, dt=0.5):
    """
    Calculates the ODE residual loss.
    T_pred: (Batch, 4, Seq_len)
    P_seq:  (Batch, Seq_len, 6) -> [res, copper, I, n, ambient, coolant]
    """
    # Extract Temperatures
    Ty, Tt, Tw, Tm = T_pred[:,0,:], T_pred[:,1,:], T_pred[:,2,:], T_pred[:,3,:]
    
    # Extract Physics Inputs
    P_res, P_cu, I, n, Ta, Tc = [P_seq[:,:,i] for i in range(6)]
    
    Py, Pt, _ = torch_split_residual(P_res, I, n)

    # Time derivative (Central/Forward difference)
    T_next, T_curr = T_pred[:, :, 1:], T_pred[:, :, :-1]
    dT_pred = (T_next - T_curr) / dt

    # Align inputs with the time-step of the derivative
    Ty_k, Tt_k, Tw_k, Tm_k = Ty[:, :-1], Tt[:, :-1], Tw[:, :-1], Tm[:, :-1]
    Py_k, Pt_k, Pw_k, Ta_k, Tc_k = Py[:, :-1], Pt[:, :-1], P_cu[:, :-1], Ta[:, :-1], Tc[:, :-1]

    # LPTN ODE Right-Hand Side
    rhs_Ty = (Py_k + (Tt_k - Ty_k)/R_YT + (Tw_k - Ty_k)/R_YW + (Tc_k - Ty_k)/R_YC)/C_Y
    rhs_Tt = (Pt_k + (Ty_k - Tt_k)/R_YT + (Tw_k - Tt_k)/R_TW + (Tm_k - Tt_k)/R_TM)/C_T
    rhs_Tw = (Pw_k + (Ty_k - Tw_k)/R_YW + (Tt_k - Tw_k)/R_TW + (Tm_k - Tw_k)/R_WM)/C_W
    rhs_Tm = ((Tt_k - Tm_k)/R_TM + (Tw_k - Tm_k)/R_WM + (Ta_k - Tm_k)/R_MA)/C_M

    rhs = torch.stack([rhs_Ty, rhs_Tt, rhs_Tw, rhs_Tm], dim=1)
    
    # Residual = (dT/dt) - Physics_Equation
    res = dT_pred - rhs
    return torch.mean(res**2)