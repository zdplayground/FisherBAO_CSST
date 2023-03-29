#!/usr/bin/env python
# coding: utf-8
import numpy as np


def comoving_dis_fun(z, Omega_m):
    return 1./(100. * ((1.0+z)**3.0*Omega_m + 1.0 - Omega_m)**0.5)

def pkmu_linear_model(theta, splPlin, splPsm, k_o, mu_o, norm_gf):
    alpha_1, alpha_2, f, b1, b2 = theta[:]
    dim_mu = len(mu_o)
    dim_k = len(k_o)
    Pkmu_model = np.zeros((dim_k, dim_mu))
    for j in range(dim_mu):
        for i in range(dim_k):
            Kaiser_term = (b1 + f * mu_o[j]**2.0)*(b2 + f * mu_o[j]**2.0)
            Pkmu_model[i,j] = norm_gf**2.0 * Kaiser_term * splPlin(k_o[i])
            
    return Pkmu_model


def pkmu_specz_model(theta, splPlin, splPsm, k_o, mu_o, k_t, mu_t, norm_gf, BAOonly): 
    f, Sigma_xy, Sigma_z, Sigma_fog, Sigma_specz, bg_specz, Pspecz_sys = theta[:]
    dim_mu = len(mu_o)
    dim_k = len(k_o)
    Pkmu_model = np.zeros((dim_k, dim_mu))
    for j in range(dim_mu):
        for i in range(dim_k):
            Kaiser_term = (bg_specz + f * mu_o[j]**2.0)**2.0
            
            BAO_damping = np.exp(k_o[i]**2.0 *(mu_o[j]**2.0 *(Sigma_xy + Sigma_z)*(Sigma_xy - Sigma_z) - Sigma_xy**2.0)/2.0)
            if BAOonly == True:
                # Linear model added with BAO damping 
                # Differed from Lado's code, we only consider alphas in Kaiser term
                Pkmu_model[i,j] = norm_gf**2.0 * Kaiser_term * ((splPlin(k_t[i,j])-splPsm(k_t[i,j]))* BAO_damping + splPsm(k_o[i]))
            else:       
                # model considers redshift error and Finger-of-God damping
                # Lorentz form for the Finger-of-God effect
                fog_damping = 1.0/(1.0 + (k_o[i] * mu_o[j] * Sigma_fog)**2.0/2.0)**2.0
                # spec-z error term
                serr_damping = np.exp(-(k_o[i] * mu_o[j] * Sigma_specz)**2.0)
  
                Pkmu_model[i,j] = norm_gf**2.0 * Kaiser_term * ((splPlin(k_t[i,j])-splPsm(k_t[i,j]))* BAO_damping + splPsm(k_o[i])) * fog_damping * serr_damping + Pspecz_sys
        
    return Pkmu_model

def pkmu_photoz_model(theta, splPlin, splPsm, k_o, mu_o, k_t, mu_t, norm_gf, BAOonly): 
    f, Sigma_xy, Sigma_z, Sigma_fog, Sigma_pz, bg_photoz = theta[:]
    dim_mu = len(mu_o)
    dim_k = len(k_o)
    Pkmu_model = np.zeros((dim_k, dim_mu))
    for j in range(dim_mu):
        for i in range(dim_k):
            Kaiser_term = (bg_photoz + f * mu_o[j]**2.0)**2.0
            
            BAO_damping = np.exp(k_o[i]**2.0 *(mu_o[j]**2.0 *(Sigma_xy + Sigma_z)*(Sigma_xy - Sigma_z) - Sigma_xy**2.0)/2.0)
            
            if BAOonly == True:
                Pkmu_model[i,j] = norm_gf**2.0 * Kaiser_term * ((splPlin(k_t[i,j])-splPsm(k_t[i,j]))* BAO_damping + splPsm(k_o[i]))
            else:             
                fog_damping = 1.0/(1.0 + (k_o[i] * mu_o[j] * Sigma_fog)**2.0/2.0)**2.0 
                # photo-z error term
                perr_damping = np.exp(-(k_o[i] * mu_o[j] * Sigma_pz)**2.0)
            
                Pkmu_model[i,j] = norm_gf**2.0 * Kaiser_term * ((splPlin(k_t[i,j])-splPsm(k_t[i,j]))* BAO_damping + splPsm(k_o[i])) * fog_damping * perr_damping
        
    return Pkmu_model

# assume density field in spec-z is after reconstruction, while the density field in photo-z is before reconstruction
def pkmu_cross_model(theta, splPlin, splPsm, k_o, mu_o, k_t, mu_t, norm_gf, BAOonly): 
    ##f_growthrate, Sigma_xy_recon, Sigma_perp_photoz, Sigma_z_recon, Sigma_para_photoz, Sigma_fog_specz, Sigma_specz_error, Sigma_photoz_error, bg_specz, bg_photoz = theta[:]
    # [f_growthrate, Sigma_xy_recon, Sigma_perp_photoz, Sigma_z_recon, Sigma_para_photoz, Sigma_fog_specz, Sigma_fog_photoz, Sigma_specz_error, Sigma_photoz_error, bg_specz, bg_photoz]
    # for now assume the cross Pk uses the same FoG term as the spec-z one
    f, Sigma_xy_recon, Sigma_xy, Sigma_z_recon, Sigma_z, Sigma_fog_specz, Sigma_fog_photoz, Sigma_specz, Sigma_pz, bg_specz, bg_photoz = theta[:]
    dim_mu = len(mu_o)
    dim_k = len(k_o)
    Pkmu_model = np.zeros((dim_k, dim_mu))
    for j in range(dim_mu):
        for i in range(dim_k):
            Kaiser_term = (bg_specz + f * mu_o[j]**2.0)*(bg_photoz + f * mu_o[j]**2.0)
            
            #BAO_damping = np.exp(k_o[i]**2.0 *(mu_o[j]**2.0 *(Sigma_xy+Sigma_z)*(Sigma_xy-Sigma_z)-Sigma_xy**2.0)/2.0)
            BAO_damping = np.exp(-k_o[i]**2.0 *(mu_o[j]**2.0 *(Sigma_z**2.0 + Sigma_z_recon**2.0) + (1-mu_o[j]**2.0) * (Sigma_xy**2.0 + Sigma_xy_recon**2.0))/4.0)
            if BAOonly == True:
                Pkmu_model[i,j] = norm_gf**2.0 * Kaiser_term * ((splPlin(k_t[i,j])-splPsm(k_t[i,j]))* BAO_damping + splPsm(k_o[i])) 
            else:                
                ##fog_damping = 1.0/(1.0 + (k_o[i] * mu_o[j] * Sigma_fog)**2.0/2.0)**2.0 
                fog_damping = 1.0/((1.0 + (k_o[i] * mu_o[j] * Sigma_fog_specz)**2.0/2.0)*(1.0 + (k_o[i] * mu_o[j] * Sigma_fog_photoz)**2.0/2.0))
                # note that there is a factor 2 in the denominator
                serr_damping = np.exp(-(k_o[i] * mu_o[j] * Sigma_specz)**2.0/2.0)
                perr_damping = np.exp(-(k_o[i] * mu_o[j] * Sigma_pz)**2.0/2.0)

                Pkmu_model[i,j] = norm_gf**2.0 * Kaiser_term * ((splPlin(k_t[i,j])-splPsm(k_t[i,j]))* BAO_damping + splPsm(k_o[i]))* fog_damping * serr_damping * perr_damping
        
    return Pkmu_model


# k_o, mu_o are the observed coordinates
def Pkmu2d(theta, k_o, mu_o, splPlin, splPsm, norm_gf, BAOonly):
    alpha_1, alpha_2, f_growthrate, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, Sigma_fog_specz, Sigma_fog_photoz, Sigma_specz_error, Sigma_photoz_error, bg_specz, bg_photoz, Pspecz_sys = theta[:]
    
    # set alpha_1: alpha_perp; alpha_2: alpha_para
    # F = alpha_para/alpha_perp
    F = alpha_2/alpha_1  
    
    k_factor = (1.0 + mu_o**2.0 * (1.0/F**2.0 - 1.0))**0.5 
    k_t = np.outer(k_o, k_factor/alpha_1)
    mu_t = mu_o /( F * k_factor)
    # 1./alp_v is for the change of volume, ignore the factor (r_s^fid/r_s)^3
    alp_v = alpha_1**2.0 * alpha_2

    # Adding the Pk amplitude factor 1./alp_v does affect the forecast of alphas a lot! But it is not physically contributed from BAO information.  
                                  
    theta_specz = np.array([f_growthrate, Sigma_perp_specz, Sigma_para_specz, Sigma_fog_specz, Sigma_specz_error, bg_specz, Pspecz_sys])
    # for specz, consider BAO reconstruction
    Pkmu_specz = pkmu_specz_model(theta_specz, splPlin, splPsm, k_o, mu_o, k_t, mu_t, norm_gf, BAOonly)
    
    theta_photoz = np.array([f_growthrate, Sigma_perp_photoz, Sigma_para_photoz, Sigma_fog_photoz, Sigma_photoz_error, bg_photoz])
    Pkmu_photoz = pkmu_photoz_model(theta_photoz, splPlin, splPsm, k_o, mu_o, k_t, mu_t, norm_gf, BAOonly)
    # not sure whether I need to introduce new parameters from the cross power spectrum, e.g. Sigma_perp_cross. Based on the Gongbo's multitracer paper,
    # they introduce parameters for LRG and ELG, respectively. For now, I adopt their setting. And I set the FoG term is the same as that of spec-z.
    theta_cross = np.array([f_growthrate, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, Sigma_fog_specz, Sigma_fog_photoz, Sigma_specz_error, Sigma_photoz_error, bg_specz, bg_photoz])
    Pkmu_cross = pkmu_cross_model(theta_cross, splPlin, splPsm, k_o, mu_o, k_t, mu_t, norm_gf, BAOonly)
    
    return Pkmu_specz, Pkmu_photoz, Pkmu_cross


# calculate the effective volume
def cal_Veff(Vsurvey, pkmu, num_den):
    return (num_den*pkmu/(1.0+num_den*pkmu))**2.0 * Vsurvey


def read_pkmu(ifile):
    data = np.load(ifile)
    parameters = data['parameters']
    pkmu_specz = data['pkmu_specz']
    pkmu_photoz = data['pkmu_photoz']
    pkmu_cross = data['pkmu_cross']
    return parameters, pkmu_specz, pkmu_photoz, pkmu_cross

# ## calculate the derivative $\frac{dP(k, \mu)}{d\lambda_i}$.
def dp_dlambda(input_file, param_id, alpha_minus=0.99, alpha_plus=1.01):
    parameters_m, specz_m, photoz_m, cross_m = read_pkmu(input_file%(param_id, alpha_minus))
    parameters_p, specz_p, photoz_p, cross_p = read_pkmu(input_file%(param_id, alpha_plus))
    #print(parameters_m, parameters_p)
    delta_param = parameters_p[param_id] - parameters_m[param_id]
    # for some parameter setted to be 0
    if delta_param == 0.0:
        dspecz_dp = np.zeros(specz_p.shape)
        dphotoz_dp = np.zeros(photoz_p.shape)
        dcross_dp = np.zeros(cross_p.shape)
    else:
        dspecz_dp = (specz_p - specz_m)/delta_param
        dphotoz_dp = (photoz_p - photoz_m)/delta_param
        dcross_dp = (cross_p - cross_m)/delta_param
    return dspecz_dp, dphotoz_dp, dcross_dp
        
def select_params_id(all_param_names, skip_params_list):
    param_id_list = []
    param_name_list = []
    for i, param_name in enumerate(all_param_names):
        if param_name not in skip_params_list:
            param_id_list.append(i)
            param_name_list.append(param_name)
    
    return param_id_list, param_name_list

