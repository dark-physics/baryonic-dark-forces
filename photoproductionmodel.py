import cross_sections as cs

# Default parameters: taken from Ben Othman et al
# Except guesses for rho parameters
default_params = np.zeros(20, dtype='float')
default_params[0] = 14.0 # g_pi_NN
default_params[1] = 4.10 # g_eta_NN
default_params[2] = 0.74 # Lambda_pi_NN
default_params[3] = 0.70 # Lambda_eta_NN
default_params[4] = 0.80 # Lambda_pi_gam_omega
default_params[5] = 0.80 # Lambda_eta_gam_omega
default_params[6] = 0.50 # Lambda_pi_gam_phi
default_params[7] = 0.60 # Lambda_eta_gam_phi
default_params[8] = 7.40 # b_Pom_omega_omega
default_params[9] = 2.00 # b_Pom_phi_phi
default_params[10] = 7.40 # B_omega (form factor slope)
default_params[11] = 3.10 # B_phi (form factor slope)
default_params[12] = 1.091 # alpha_Pom (Pomeron trajectory)
default_params[13] = 0.2 # a_Pom_omega_omega
default_params[14] = 0.9 # a_Pom_phi_phi 
default_params[15] = 0.80 # Lambda_pi_gam_rho
default_params[16] = 0.80 # Lambda_eta_gam_rho
default_params[17] = 7.40 # b_Pom_rho_rho
default_params[18] = 0.2 # a_Pom_rho_rho
default_params[19] = 7.4 # B_rho (form factor slope)

class model:

    def __init__(self,params=default_params):
        
        # Phenomenological model parameters
        self.params = params

    # Real photoproduction cross sections
    # photon + proton -> vector meson + proton
    # for vector mesons = omega, rho^0, phi
    
    def dsig_dt_omega(self,W,t):
        return cs.dsig_dt_omega(W,t,self.params)

    def dsig_dt_rho(self,W,t):
        return cs.dsig_dt_rho(W,t,self.params)

    def dsig_dt_phi(self,W,t):
        return cs.dsig_dt_phi(W,t,self.params)

    def dsig_dcosth_omega(self,W,costh):
        return cs.dsig_dcosth_omega(W,costh,self.params)

    def dsig_dcosth_rho(self,W,costh):
        return cs.dsig_dcosth_rho(W,costh,self.params)

    def dsig_dcosth_phi(self,W,costh):
        return cs.dsig_dcosth_phi(W,costh,self.params)

    def sig_omega(self,W):
        return cs.sig_omega(W,self.params)

    def sig_rho(self,W):
        return cs.sig_rho(W,self.params)

    def sig_phi(self,W):
        return cs.sig_phi(W,self.params)

    # Virtual photoproduction cross sections via 
    # photon^* + proton -> vector meson + proton
    # for vector mesons = omega, rho^0, phi
    
    def sigv_omega(self,s_tot,W,Q2):
        return cs.sig_virtual_omega(s_tot,W,Q2,self.params)

    def sigv_rho(self,s_tot,W,Q2):
        return cs.sig_virtual_rho(s_tot,W,Q2,self.params)

    def sigv_phi(self,s_tot,W,Q2):
        return cs.sig_virtual_phi(s_tot,W,Q2,self.params)



    
        

        

        