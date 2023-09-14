import numpy as np
import pandas as pd
from scipy.integrate import quad, solve_ivp

# Function used for integration

def integrate(func,a,b):
    f = lambda t,y: func(t)
    sol = solve_ivp(f,[a,b],[0],atol=1e-12,rtol=1e-12)
    output = sol.y[0,-1]
    return output

# Fixed numerical inputs

g_pi_gam_omega = 1.83 # Updated
g_eta_gam_omega = 0.35

g_pi_gam_rho = 0.57
g_eta_gam_rho = 1.22

g_pi_gam_phi = 0.138 
g_eta_gam_phi = 0.704

f_omega = 0.201 # GeV
f_phi = 0.228 # GeV
f_rho = 0.220 # GeV

beta_PNN = 1.87 # 1/GeV
alpha_prime_P_0 = 0.25 # 1/GeV^2

alpha_EM = 1/137.035999139

mp = 0.9382720813 # GeV
me = 0.511e-3 # GeV
m_omega = 0.78266 # GeV Updated
m_rho = 0.77526  # GeV
m_phi = 1.019461 # GeV
m_pi = 0.1349770 # GeV
m_eta = 0.547862 # GeV

Gamma_omega = 8.68e-3 # GeV Updated
Gamma_phi = 4.249e-3 # GeV
Gamma_rho = 0.1474 # GeV

units = 389.3793656 # Conversion from 1/GeV^2 to mu_barn

# Phenomenological model parameters:
#  params[0] = g_pi_NN
#  params[1] = g_eta_NN
#  params[2] = Lambda_pi_NN
#  params[3] = Lambda_eta_NN
#  params[4] = Lambda_pi_gam_omega
#  params[5] = Lambda_eta_gam_omega
#  params[6] = Lambda_pi_gam_phi
#  params[7] = Lambda_eta_gam_phi
#  params[8] = b_Pom_omega_omega
#  params[9] = b_Pom_phi_phi
#  params[10] = B_omega (form factor slope)
#  params[11] = B_phi (form factor slope)
#  params[12] = alpha_Pom_prime (Pomeron trajectory slope)
#  params[13] = a_Pom_omega_omega
#  params[14] = a_Pom_phi_phi 


# Differential cross sections
# Note: plus = pomeron exchange
#       minus = pseudoscalar exchange
# rho cross sections are incomplete and do not describe data at low s 
# (need sigma-meson exchange or other terms)

def dsig_dt_omega_plus(W,t,params):
    
    s = W**2
    
    a_P_omega_omega = params[13]
    b_P_omega_omega = params[8]
    B_omega = params[10]
    
    prefactor = alpha_EM * f_omega**2 / m_omega**2 / 8
    form_factor = np.exp(0.5*B_omega*t)
    
    alpha_P_0 = params[12]
    s0 = 1/alpha_prime_P_0
    alpha_P = alpha_P_0 + t/s0
    exponent = 2*alpha_P - 2 
    
    term1 = b_P_omega_omega**2 + 2 * a_P_omega_omega * b_P_omega_omega * m_omega**2
    term2 = (m_omega**2 - t)**2 * a_P_omega_omega**2
    factor_PVV = term1 + term2  
    
    dsig_dt = prefactor * beta_PNN**2 * factor_PVV * form_factor**2 * (s/s0)**exponent
    
    return dsig_dt * units

def dsig_dt_rho_plus(W,t,params):
    
    s = W**2
    
    a_P_rho_rho = params[13] 
    b_P_rho_rho = params[8]
    B_rho = params[10]
    
    prefactor = alpha_EM * f_rho**2 / m_rho**2 / 8 * 9
    form_factor = np.exp(0.5*B_rho*t)
    
    alpha_P_0 = params[12]
    s0 = 1/alpha_prime_P_0
    alpha_P = alpha_P_0 + t/s0
    exponent = 2*alpha_P - 2 
    
    term1 = b_P_rho_rho**2 + 2 * a_P_rho_rho * b_P_rho_rho * m_rho**2
    term2 = (m_rho**2 - t)**2 * a_P_rho_rho**2
    factor_PVV = term1 + term2  
    
    dsig_dt = prefactor * beta_PNN**2 * factor_PVV * form_factor**2 * (s/s0)**exponent
    
    return dsig_dt * units

def dsig_dt_phi_plus(W,t,params):
    
    s = W**2
    
    a_P_phi_phi = params[14]
    b_P_phi_phi = params[9]
    B_phi = params[11]
    
    prefactor = alpha_EM * f_phi**2 / m_phi**2 / 4
    form_factor = np.exp(0.5*B_phi*t)
    
    alpha_P_0 = params[12]
    s0 = 1/alpha_prime_P_0
    alpha_P = alpha_P_0 + t/s0
    exponent = 2*alpha_P - 2 
    
    term1 = b_P_phi_phi**2 + 2 * a_P_phi_phi * b_P_phi_phi * m_phi**2
    term2 = (m_phi**2 - t)**2 * a_P_phi_phi**2 
    factor_PVV = term1 + term2 
    
    dsig_dt = prefactor * beta_PNN**2 * factor_PVV * form_factor**2 * (s/s0)**exponent
    
    return dsig_dt * units

def dsig_dt_omega_minus(W,t,params):
    
    s = W**2
    
    g_pi_NN = params[0]
    g_eta_NN = params[1]
    L_pi_NN = params[2]
    L_eta_NN = params[3]
    L_pi_gam_omega = params[4]
    L_eta_gam_omega = params[5]
    
    def F(Lambda,m): 
        return (Lambda**2 - m**2) / (Lambda**2 - t)
    
    A_omega_pi = g_pi_NN * g_pi_gam_omega * F(L_pi_NN,m_pi) * F(L_pi_gam_omega,m_pi) / (t - m_pi**2)
    A_omega_eta = g_eta_NN * g_eta_gam_omega * F(L_eta_NN,m_eta) * F(L_eta_gam_omega,m_eta) / (t - m_eta**2)
    A_tot = A_omega_pi + A_omega_eta
    
    prefactor = - alpha_EM * t * (t - m_omega**2)**2 / (16 * m_omega**2 * (s - mp**2)**2 )
    
    dsig_dt = prefactor * np.abs(A_tot)**2
    
    return dsig_dt * units

def dsig_dt_phi_minus(W,t,params):
    
    s = W**2
    
    g_pi_NN = params[0]
    g_eta_NN = params[1]
    L_pi_NN = params[2]
    L_eta_NN = params[3]
    L_pi_gam_phi = params[6]
    L_eta_gam_phi = params[7]
    
    def F(Lambda,m): 
        return (Lambda**2 - m**2) / (Lambda**2 - t)
    
    A_phi_pi = g_pi_NN * g_pi_gam_phi * F(L_pi_NN,m_pi) * F(L_pi_gam_phi,m_pi) / (t - m_pi**2)
    A_phi_eta = g_eta_NN * g_eta_gam_phi * F(L_eta_NN,m_eta) * F(L_eta_gam_phi,m_eta) / (t - m_eta**2)
    A_tot = A_phi_pi + A_phi_eta
    
    prefactor = - alpha_EM * t * (t - m_phi**2)**2 / (16 * m_phi**2 * (s - mp**2)**2 )
    
    dsig_dt = prefactor * np.abs(A_tot)**2
    
    return dsig_dt * units

def dsig_dt_rho_minus(W,t,params):
    
    s = W**2
    
    g_pi_NN = params[0]
    g_eta_NN = params[1]
    L_pi_NN = params[2]
    L_eta_NN = params[3]
    L_pi_gam_omega = params[4] # Assume same for rho
    L_eta_gam_omega = params[5] # Assume same for rho
    
    def F(Lambda,m): 
        return (Lambda**2 - m**2) / (Lambda**2 - t)
    
    A_rho_pi = g_pi_NN * g_pi_gam_rho * F(L_pi_NN,m_pi) * F(L_pi_gam_omega,m_pi) / (t - m_pi**2)
    A_rho_eta = g_eta_NN * g_eta_gam_rho * F(L_eta_NN,m_eta) * F(L_eta_gam_omega,m_eta) / (t - m_eta**2)
    A_tot = A_rho_pi + A_rho_eta
    
    prefactor = - alpha_EM * t * (t - m_rho**2)**2 / (16 * m_rho**2 * (s - mp**2)**2 )
    
    dsig_dt = prefactor * np.abs(A_tot)**2
    
    return dsig_dt * units

def dsig_dt_rho(W,t,params):
    return dsig_dt_rho_plus(W,t,params) + dsig_dt_rho_minus(W,t,params)

def dsig_dt_omega(W,t,params):
    return dsig_dt_omega_plus(W,t,params) + dsig_dt_omega_minus(W,t,params)

def dsig_dt_phi(W,t,params):
    return dsig_dt_phi_plus(W,t,params) + dsig_dt_phi_minus(W,t,params)

# General function for computing dsig/dcosth
def dsig_dcosth(W,costh,params,mV,dsig_dt_func):
    
    s = W**2
    
    # Initial photon kinematics
    Eq1 = 0.5*(s - mp**2)/W
    q1 = Eq1
    
    # Final vector kinematics
    Eq2 = 0.5*(s - mp**2 + mV**2)/W
    q2 = np.sqrt(Eq2**2 - mV**2)
    
    # Compute t
    t = mV**2 - 2*Eq1*Eq2 + 2*q1*q2*costh
    
    # Jacobian
    J = 2*q1*q2
    
    # Diff cross sec
    try:
        dsig_dt = dsig_dt_func(W,t,params)
    except:
        dsig_dt = dsig_dt_func(W,t,params,mV)
        
    return dsig_dt * J
    
def dsig_dcosth_omega(W,costh,params):
    return dsig_dcosth(W,costh,params,m_omega,dsig_dt_omega)

def dsig_dcosth_phi(W,costh,params):
    return dsig_dcosth(W,costh,params,m_phi,dsig_dt_phi)

def dsig_dcosth_omega_plus(W,costh,params):
    return dsig_dcosth(W,costh,params,m_omega,dsig_dt_omega_plus)

def dsig_dcosth_phi_plus(W,costh,params):
    return dsig_dcosth(W,costh,params,m_phi,dsig_dt_phi_plus)

def dsig_dcosth_omega_minus(W,costh,params):
    return dsig_dcosth(W,costh,params,m_omega,dsig_dt_omega_minus)

def dsig_dcosth_phi_minus(W,costh,params):
    return dsig_dcosth(W,costh,params,m_phi,dsig_dt_phi_minus)


# General quantile function
def quantile(func,chain,q,*args):
    
    # Only one quantile
    if np.ndim(q) == 0:
        
        # Check dim of func for single parameter point
        try:
            size = len(func(*args,chain[0]))
        
            # Initialize array
            arr = np.zeros((len(chain),size))
            for i,p in enumerate(chain):
                arr[i] = func(*args,chain[i])
                
        except:
            arr = np.zeros(len(chain))
            for i,p in enumerate(chain):
                arr[i] = func(*args,chain[i])
        
        return np.quantile(arr,q,axis=0)
    
    # More than one quantile
    else:
        return [quantile(func,chain,qi,*args) for qi in q]
    
def dsig_dt_omega_quantile(W,t,chain,q):
    return quantile(dsig_dt_omega,chain,q,W,t)

def dsig_dt_phi_quantile(W,t,chain,q):
    return quantile(dsig_dt_phi,chain,q,W,t)

def dsig_dcosth_omega_quantile(W,costh,chain,q):
    return quantile(dsig_dcosth_omega,chain,q,W,costh)

def dsig_dcosth_phi_quantile(W,costh,chain,q):
    return quantile(dsig_dcosth_phi,chain,q,W,costh)
              
def sig(func,W,params,mV):
    
    s = W**2
    
    # Initial photon kinematics
    Eq1 = 0.5*(s - mp**2)/W
    q1 = Eq1
    
    # Final vector kinematics
    Eq2 = 0.5*(s - mp**2 + mV**2)/W
    q2 = np.sqrt(Eq2**2 - mV**2)
    
    # Compute t
    tmin = mV**2 - 2*Eq1*Eq2 - 2*q1*q2
    tmax = mV**2 - 2*Eq1*Eq2 + 2*q1*q2
    
    if np.ndim(W) == 0:
        try:
            f = lambda x: func(W,x,params)
            # return quad(f,tmin,tmax,epsrel=1e-14,epsabs=1e-14)[0]
            return integrate(f,tmin,tmax)
        except:
            f = lambda x: func(W,x,params,mV)
            # return quad(f,tmin,tmax,epsrel=1e-14,epsabs=1e-14)[0]
            return integrate(f,tmin,tmax)
    else:
        return np.array([sig(func,Wi,params,mV) for Wi in W])
    
def sig_omega(W,params):
    return sig(dsig_dt_omega,W,params,m_omega)

def sig_omega_plus(W,params):
    return sig(dsig_dt_omega_plus,W,params,m_omega)

def sig_omega_minus(W,params):
    return sig(dsig_dt_omega_minus,W,params,m_omega)

def sig_rho(W,params):
    return sig(dsig_dt_rho,W,params,m_rho)

def sig_phi(W,params):
    return sig(dsig_dt_phi,W,params,m_phi)

def sig_phi_plus(W,params):
    return sig(dsig_dt_phi_plus,W,params,m_phi)

def sig_phi_minus(W,params):
    return sig(dsig_dt_phi_minus,W,params,m_phi)

def sig_omega_quantile(W,chain,q):
    return quantile(sig_omega,chain,q,W)

def sig_phi_quantile(W,chain,q):
    return quantile(sig_phi,chain,q,W)

def sig_rho_quantile(W,chain,q):
    return quantile(sig_rho,chain,q,W)

def dsig_dt_B_plus(W,t,params,mB):
    
    s = W**2
    
    alpha_P_0 = params[12]
    s0 = 1/alpha_prime_P_0
    alpha_P = alpha_P_0 + t/s0
    exponent = 2*alpha_P - 2 
    
    a_P_omega_omega = params[13]
    b_P_omega_omega = params[8]
    b_omega = params[10]
    
    a_P_phi_phi = params[14]
    b_P_phi_phi = params[9]
    b_phi = params[11]
    
    F_omega = 1/(1 - mB**2/m_omega**2 - 1j*Gamma_omega/m_omega)
    F_phi = 1/(1 - mB**2/m_phi**2 - 1j*Gamma_phi/m_phi)
    
    F_p_omega = np.exp(0.5*b_omega*t)
    F_p_phi = np.exp(0.5*b_phi*t)
    
    c_omega = f_omega**2 * F_omega * F_p_omega / m_omega**2
    c_phi = f_phi**2 * F_phi * F_p_phi / m_phi**2
    
    a_P_gamma_B = c_omega * a_P_omega_omega - c_phi * a_P_phi_phi
    b_P_gamma_B = c_omega * b_P_omega_omega - c_phi * b_P_phi_phi
    
    prefactor = np.pi * alpha_EM/9 * beta_PNN**2 * (s/s0)**exponent
    
    terms = np.abs(a_P_gamma_B)**2 * (mB**2 - t)**2 + np.abs(b_P_gamma_B)**2 
    terms += 2 * np.real(a_P_gamma_B*np.conjugate(b_P_gamma_B)) * mB**2
    
    return prefactor * terms * units
    
def dsig_dt_B_minus(W,t,params,mB):
    
    s = W**2
    
    g_pi_NN = params[0]
    g_eta_NN = params[1]
    L_pi_NN = params[2]
    L_eta_NN = params[3]
    L_pi_gam_omega = params[4]
    L_eta_gam_omega = params[5]
    L_pi_gam_phi = params[6]
    L_eta_gam_phi = params[7]
    
    def F(Lambda,m): 
        return (Lambda**2 - m**2) / (Lambda**2 - t)
    
    F_omega = 1/(1 - mB**2/m_omega**2 - 1j*Gamma_omega/m_omega)
    F_phi = 1/(1 - mB**2/m_phi**2 - 1j*Gamma_phi/m_phi)
    
    A_omega_pi = g_pi_NN * g_pi_gam_omega * F(L_pi_NN,m_pi) * F(L_pi_gam_omega,m_pi) / (t - m_pi**2) 
    A_omega_eta = g_eta_NN * g_eta_gam_omega * F(L_eta_NN,m_eta) * F(L_eta_gam_omega,m_eta) / (t - m_eta**2)
    A_omega_tot = np.sqrt(2) * (A_omega_pi + A_omega_eta) * f_omega * F_omega / m_omega**2
    
    A_phi_pi = g_pi_NN * g_pi_gam_phi * F(L_pi_NN,m_pi) * F(L_pi_gam_phi,m_pi) / (t - m_pi**2)
    A_phi_eta = g_eta_NN * g_eta_gam_phi * F(L_eta_NN,m_eta) * F(L_eta_gam_phi,m_eta) / (t - m_eta**2)
    A_phi_tot = (A_phi_pi + A_phi_eta) * f_phi * F_phi / m_phi**2
    
    prefactor = np.pi * alpha_EM * np.abs(t) * (t - mB**2)**2 / (36 * s**2)
    
    dsig_dt = prefactor * np.abs(A_phi_tot + A_omega_tot)**2
    
    return dsig_dt * units

def dsig_dt_B(W,t,params,mB):
    
    return dsig_dt_B_plus(W,t,params,mB) + dsig_dt_B_minus(W,t,params,mB)

def sig_B(W,mB,params):
    return sig(dsig_dt_B,W,params,mB)

def sig_B_plus(W,mB,params):
    return sig(dsig_dt_B_plus,W,params,mB)

def sig_B_minus(W,mB,params):
    return sigt(dsig_dt_B_minus,W,params,mB)

def sig_B_quantile(W,mB,chain,q):
    return quantile(sig_B,chain,q,W,mB)

def dsig_dcosth_B(W,costh,params,mB):
    return dsig_dcosth(W,costh,params,mB,dsig_dt_B)

def dsig_dcosth_B_plus(W,costh,params,mB):
    return dsig_dcosth(W,costh,params,mB,dsig_dt_B_plus)

def dsig_dcosth_B_minus(W,costh,params,mB):
    return dsig_dcosth(W,costh,params,mB,dsig_dt_B_minus)

def dsigT_dt_B_plus(W,t,Q2,params,mB):
    
    if W < mB + mp:
        return 0.0

    s = W**2
    
    alpha_P_0 = params[12]
    s0 = 1/alpha_prime_P_0
    alpha_P = alpha_P_0 + t/s0
    exponent = 2*alpha_P - 2 
    
    a_P_omega_omega = params[13]
    b_P_omega_omega = params[8]
    B_omega = params[10]
    
    a_P_phi_phi = params[14]
    b_P_phi_phi = params[9]
    B_phi = params[11]
    
    F_omega_1 = 1/(1 - mB**2/m_omega**2 - 1j*Gamma_omega/m_omega)
    F_omega_2 = 1/(1 + Q2/m_omega**2 - 1j*Gamma_omega/m_omega)
    
    F_phi_1 = 1/(1 - mB**2/m_phi**2 - 1j*Gamma_phi/m_phi)
    F_phi_2 = 1/(1 + Q2/m_phi**2 - 1j*Gamma_phi/m_phi)
    
    F_p_omega = np.exp(0.5*B_omega*t)
    F_p_phi = np.exp(0.5*B_phi*t)
    
    C_omega = f_omega**2/m_omega**2 * F_omega_1 * F_omega_2 * F_p_omega
    C_phi = f_phi**2/m_phi**2 * F_phi_1 * F_phi_2 * F_p_phi
    
    a_P_gamma_B = C_omega * a_P_omega_omega - C_phi * a_P_phi_phi
    b_P_gamma_B = C_omega * b_P_omega_omega - C_phi * b_P_phi_phi
    
    prefactor = np.pi * alpha_EM/9 * beta_PNN**2 * (s/s0)**exponent * (1 + 0.5 * Q2/s)**2
    
    terms = np.abs(a_P_gamma_B)**2 * ((mB**2 - t)**2 - 2 * Q2 * mB**2 + Q2**2)
    terms += np.abs(b_P_gamma_B)**2 
    terms += 2 * (mB**2 - Q2) * np.real(a_P_gamma_B*np.conjugate(b_P_gamma_B))
    
    return prefactor * terms * units

def dsigL_dt_B_plus(W,t,Q2,params,mB):
    
    if W < mB + mp:
        return 0.0

    s = W**2
    
    alpha_P_0 = params[12]
    s0 = 1/alpha_prime_P_0
    alpha_P = alpha_P_0 + t/s0
    exponent = 2*alpha_P - 2 
    
    a_P_omega_omega = params[13]
    B_omega = params[10]
    
    a_P_phi_phi = params[14]
    B_phi = params[11]
    
    F_omega_1 = 1/(1 - mB**2/m_omega**2 - 1j*Gamma_omega/m_omega)
    F_omega_2 = 1/(1 + Q2/m_omega**2 - 1j*Gamma_omega/m_omega)
    
    F_phi_1 = 1/(1 - mB**2/m_phi**2 - 1j*Gamma_phi/m_phi)
    F_phi_2 = 1/(1 + Q2/m_phi**2 - 1j*Gamma_phi/m_phi)
    
    F_p_omega = np.exp(0.5*B_omega*t)
    F_p_phi = np.exp(0.5*B_phi*t)
    
    C_omega = f_omega**2/m_omega**2 * F_omega_1 * F_omega_2 * F_p_omega
    C_phi = f_phi**2/m_phi**2 * F_phi_1 * F_phi_2 * F_p_phi
    
    a_P_gamma_B = C_omega * a_P_omega_omega - C_phi * a_P_phi_phi
    
    prefactor = 4*np.pi * alpha_EM/9 * beta_PNN**2 * (s/s0)**exponent * (1 + 0.5 * Q2/s)**2
    
    terms = np.abs(a_P_gamma_B)**2 * (mB**2 - t) * Q2 
    
    return prefactor * terms * units
    
def dsigT_dt_B_minus(W,t,Q2,params,mB):

    if W < mB + mp:
        return 0.0

    s = W**2
    
    g_pi_NN = params[0]
    g_eta_NN = params[1]
    L_pi_NN = params[2]
    L_eta_NN = params[3]
    L_pi_gam_omega = params[4]
    L_eta_gam_omega = params[5]
    L_pi_gam_phi = params[6]
    L_eta_gam_phi = params[7]
    
    def F(Lambda,m): 
        return (Lambda**2 - m**2) / (Lambda**2 - t)
    
    F_omega_1 = 1/(1 - mB**2/m_omega**2 - 1j*Gamma_omega/m_omega)
    F_omega_2 = 1/(1 + Q2/m_omega**2 - 1j*Gamma_omega/m_omega)
    
    F_phi_1 = 1/(1 - mB**2/m_phi**2 - 1j*Gamma_phi/m_phi)
    F_phi_2 = 1/(1 + Q2/m_phi**2 - 1j*Gamma_phi/m_phi)
    
    A_omega_pi = g_pi_NN * g_pi_gam_omega * F(L_pi_NN,m_pi) * F(L_pi_gam_omega,m_pi) / (t - m_pi**2) 
    A_omega_eta = g_eta_NN * g_eta_gam_omega * F(L_eta_NN,m_eta) * F(L_eta_gam_omega,m_eta) / (t - m_eta**2)
    A_omega_tot = np.sqrt(2) * (A_omega_pi + A_omega_eta) * f_omega * F_omega_1 * F_omega_2 / m_omega**2
    
    A_phi_pi = g_pi_NN * g_pi_gam_phi * F(L_pi_NN,m_pi) * F(L_pi_gam_phi,m_pi) / (t - m_pi**2)
    A_phi_eta = g_eta_NN * g_eta_gam_phi * F(L_eta_NN,m_eta) * F(L_eta_gam_phi,m_eta) / (t - m_eta**2)
    A_phi_tot = (A_phi_pi + A_phi_eta) * f_phi * F_phi_1 * F_phi_2 / m_phi**2
    
    prefactor = np.pi * alpha_EM * np.abs(t) * ((t - mB**2)**2 + Q2**2 + 2 * Q2 * mB**2) / (36 * s * (s + Q2))
    
    dsig_dt = prefactor * np.abs(A_phi_tot + A_omega_tot)**2
    
    return dsig_dt * units

def dsigL_dt_B_minus(W,t,Q2,params,mB):
    
    if W < mB + mp:
        return 0.0
    
    s = W**2
    
    g_pi_NN = params[0]
    g_eta_NN = params[1]
    L_pi_NN = params[2]
    L_eta_NN = params[3]
    L_pi_gam_omega = params[4]
    L_eta_gam_omega = params[5]
    L_pi_gam_phi = params[6]
    L_eta_gam_phi = params[7]
    
    def F(Lambda,m): 
        return (Lambda**2 - m**2) / (Lambda**2 - t)
    
    F_omega_1 = 1/(1 - mB**2/m_omega**2 - 1j*Gamma_omega/m_omega)
    F_omega_2 = 1/(1 + Q2/m_omega**2 - 1j*Gamma_omega/m_omega)
    
    F_phi_1 = 1/(1 - mB**2/m_phi**2 - 1j*Gamma_phi/m_phi)
    F_phi_2 = 1/(1 + Q2/m_phi**2 - 1j*Gamma_phi/m_phi)
    
    A_omega_pi = g_pi_NN * g_pi_gam_omega * F(L_pi_NN,m_pi) * F(L_pi_gam_omega,m_pi) / (t - m_pi**2) 
    A_omega_eta = g_eta_NN * g_eta_gam_omega * F(L_eta_NN,m_eta) * F(L_eta_gam_omega,m_eta) / (t - m_eta**2)
    A_omega_tot = np.sqrt(2) * (A_omega_pi + A_omega_eta) * f_omega * F_omega_1 * F_omega_2 / m_omega**2
    
    A_phi_pi = g_pi_NN * g_pi_gam_phi * F(L_pi_NN,m_pi) * F(L_pi_gam_phi,m_pi) / (t - m_pi**2)
    A_phi_eta = g_eta_NN * g_eta_gam_phi * F(L_eta_NN,m_eta) * F(L_eta_gam_phi,m_eta) / (t - m_eta**2)
    A_phi_tot = (A_phi_pi + A_phi_eta) * f_phi * F_phi_1 * F_phi_2 / m_phi**2
    
    prefactor = np.pi * alpha_EM * np.abs(t) * t**2 * Q2 / (9 * s * (s + Q2))
    
    dsig_dt = prefactor * np.abs(A_phi_tot + A_omega_tot)**2
    
    return dsig_dt * units
    
def dsigT_dt_B(W,t,Q2,params,mB):
    return dsigT_dt_B_plus(W,t,Q2,params,mB) + dsigT_dt_B_minus(W,t,Q2,params,mB)

def dsigL_dt_B(W,t,Q2,params,mB):
    return dsigL_dt_B_plus(W,t,Q2,params,mB) + dsigL_dt_B_minus(W,t,Q2,params,mB)
    
def sig_electro(func,W,Q2,params,mB):
       
    if (np.ndim(W) == 0) and (np.ndim(Q2) == 0):
    
        if (W < mB + mp):
            return 0

        # Initial photon kinematics
        Eq1 = 0.5*(W**2 - mp**2 - Q2)/W
        q1 = np.sqrt(Eq1**2 + Q2)

        # Final B kinematics
        Eq2 = 0.5*(W**2 - mp**2 + mB**2)/W
        q2 = np.sqrt(Eq2**2 - mB**2)

        # Compute t
        tmin = mB**2 - Q2 - 2*Eq1*Eq2 - 2*q1*q2
        tmax = mB**2 - Q2 - 2*Eq1*Eq2 + 2*q1*q2
    
        f = lambda t: func(W,t,Q2,params,mB)
        
        # Calculate integral over t
        return integrate(f,tmin,tmax)

    # Other cases to handle array inputs
    elif (np.ndim(Q2) == 0):
        return np.array([sig_electro(func,Wi,Q2,params,mB) for Wi in W])

    elif (np.ndim(W) == 0):
        return np.array([sig_electro(func,W,Q2i,params,mB) for Q2i in Q2])

    else:
        return np.array([sig_electro(func,Wi,Q2i,params,mB) for Wi,Q2i in zip(W,Q2)])

def sigT_B_plus(W,Q2,mB,params):
    return sig_electro(dsigT_dt_B_plus,W,Q2,params,mB)

def sigL_B_plus(W,Q2,mB,params):
    return sig_electro(dsigL_dt_B_plus,W,Q2,params,mB)

def sigT_B_minus(W,Q2,mB,params):
    return sig_electro(dsigT_dt_B_minus,W,Q2,params,mB)

def sigL_B_minus(W,Q2,mB,params):
    return sig_electro(dsigL_dt_B_minus,W,Q2,params,mB)

def sigT_B(W,Q2,mB,params):
    return sig_electro(dsigT_dt_B,W,Q2,params,mB)

def sigL_B(W,Q2,mB,params):
    return sig_electro(dsigL_dt_B,W,Q2,params,mB)

def Gamma_T(Q2,y):
    prefactor = alpha_EM / (2*np.pi * Q2 * y)
    return prefactor * (1 + (1-y)**2 - 2 * me**2 * y**2 /Q2)

def Gamma_L(Q2,y):
    prefactor = alpha_EM / (np.pi * Q2 * y)
    return prefactor * (1-y)

def sigT_B_quantile(W,Q2,mB,chain,q):
    return quantile(sigT_B,chain,q,W,Q2,mB)

def sigL_B_quantile(W,Q2,mB,chain,q):
    return quantile(sigL_B,chain,q,W,Q2,mB)