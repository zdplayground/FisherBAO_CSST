#!/usr/bin/env python
# coding: utf-8 

# \begin{align}
# F_{ij} = \int_{-1}^{1}\int_{k_{\text{min}}}^{k_{\text{max}}} \frac{2\pi k^2 dk d\mu}{2(2\pi)^3} V_{\text{survey}}\left[\frac{\partial\,P^T}{\partial \lambda_i} C^{-1}\frac{\partial\, P}{\partial \lambda_j}\right],
# \end{align}
# where
# \begin{align}
# P=
# \begin{bmatrix} 
# \hat{P}_A(k, \mu) \\
# P_{AB}(k,\mu)
# \end{bmatrix},
# \end{align}
# $A$ and $B$ denote the spectro-z and photo-z, respectively. For the auto power spectrum, we consider shot noise, i.e.,
# \begin{align}
# \hat{P} = P + \frac{1}{\bar{n}}.
# \end{align}
# The covariance matrix $C$ is given by
# \begin{align}
# C=
# \begin{bmatrix} 
# \hat{P}_A^2 & \hat{P}_A P_{AB} \\
# \hat{P}_{A} P_{AB} & \frac{1}{2}(P_{AB}^2+\hat{P}_A \hat{P}_B) \\
# \end{bmatrix}.
# \end{align}

import os, sys
import numpy as np
from scipy import linalg, integrate
from pkmu_module import read_pkmu, dp_dlambda, select_params_id
from argparse import ArgumentParser


## Calculate the cross covariance between different power spectra.
## !!! Pa and Pb should include shot noise !!!
def cal_cross_cov(Pa, Pab, Pb):
    cov = np.zeros((2,2))
    cov[0,0] = Pa * Pa
    cov[0,1] = Pa * Pab
    cov[1,0] = Pa * Pab
    cov[1,1] = 0.5*(Pab* Pab + Pa*Pb)
    return cov


## dp_dm, dp_dn should be vectors, p contains different power spectra.
## !!! Pa and Pb should include shot noise !!!
def cal_Fmn(kobs, mu_obs, V_survey, Pa, Pab, Pb, dPa_dm, dPab_dm, dPa_dn, dPab_dn):
    Fmn_k = np.zeros(len(kobs))
    
    for i in range(len(kobs)):
        temp = np.zeros(len(mu_obs))
        for j in range(len(mu_obs)):
            cross_cov = cal_cross_cov(Pa[i,j], Pab[i,j], Pb[i,j])
            inv_cov = linalg.inv(cross_cov)
            
            dP_dm_array = np.array([dPa_dm[i,j], dPab_dm[i,j]])
            
            dP_dn_array = np.array([dPa_dn[i,j], dPab_dn[i,j]])
            
            temp[j] = V_survey * np.dot(np.dot(dP_dm_array, inv_cov), dP_dn_array) * (kobs[i]/(2*np.pi))**2.0 
        Fmn_k[i] = integrate.trapz(temp, mu_obs)
    
    Fmn = integrate.trapz(Fmn_k, kobs) 
                        
    return Fmn


def main():
    parser = ArgumentParser(description="Fisher forecast of BAO from the combination of spec-z and the cross-correlation between spec-z and photo-z survey.")
    parser.add_argument("--survey_area", type=float, help="The sky area (deg^2)", required=True)
    parser.add_argument("--kmax", type=float, help="The maximum k of P(k,mu).", required=True)
    parser.add_argument("--zmin", type=float, help="z minimum.", default=0.0)
    parser.add_argument("--zmax", type=float, help="z maximum.", default=1.6)
    parser.add_argument("--nzbins", type=int, help="The number of z bins.", required=True)
    parser.add_argument("--sigma_specz", type=float, help="spec-z error sigma_0.", required=True)
    parser.add_argument("--sigma_photoz", type=float, help="photo-z error, sigma_z.", required=True)
    parser.add_argument("--Sigma_fog", type=float, help="Finger-of-God damping term.", default=0.0, required=True)
    parser.add_argument("--const_low", type=float, help="The lower constant multiplied on parameters.", required=True)
    parser.add_argument("--const_up", type=float, help="The upper constant multiplied on parameters.", required=True)
    parser.add_argument("--input_dir", help="input directory.", required=True)
    parser.add_argument("--output_dir", help="output directory.", required=True)
    
    args = parser.parse_args()
    
    survey_area = args.survey_area
    kmax = args.kmax
    #kmax = 0.5
    zmin = args.zmin
    zmax = args.zmax
    nzbins = args.nzbins
    ##sigma_specz = 0.000
    sigma_specz = args.sigma_specz
    sigma_photoz = args.sigma_photoz

    Sigma_fog = args.Sigma_fog
    #Sigma_fog = 3.0

    const_low = args.const_low
    const_up = args.const_up

    zbins = np.linspace(zmin, zmax, nzbins+1)
    zlow_bins = zbins[0:-1]
    zup_bins = zbins[1:]
    
    path = args.input_dir
    print(path)
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ##theta_names = ['alpha_perp', 'alpha_para', 'f_growthrate', 'Sigma_perp_specz', 'Sigma_perp_photoz', 'Sigma_para_specz', 'Sigma_para_photoz', 'Sigma_fog_specz', 'Sigma_fog_photoz', 'Sigma_specz_error', 'Sigma_photoz_error', 'bg_specz', 'bg_photoz', 'Pspecz_sys']
    if Sigma_fog < 1.e-7:
        skip_params_list = ['Sigma_fog_specz', 'Sigma_fog_photoz']
    else:
        skip_params_list = []

    Fisher_matrix_diffz = []
    output_alphas_mar = []  # marginalize other parameters
    output_alphas_unmar = [] # unmarginalize (or fix) other parameters
    output_all_params = []
    for z_low, z_up in zip(zlow_bins, zup_bins):
        filename_nonlinear = "pkmu_nonlinear_csst_z{0:.2f}_{1:.2f}_sigmaspecz{2:.3f}_sigmaphotoz{3:.3f}_kmax{4:.2f}_param%dX%.3f.npz".format(z_low, z_up, sigma_specz, sigma_photoz, kmax)

        default_res = np.load(path+"pkmu_nonlinear_csst_z{0:.2f}_{1:.2f}_sigmaspecz{2:.3f}_sigmaphotoz{3:.3f}_kmax{4:.2f}_default_params.npz".format(z_low, z_up, sigma_specz, sigma_photoz, kmax))

        kcenter = default_res['kcenter']
        print("k dim:", len(kcenter))
        kmin = default_res['kmin']
        mu = default_res['mu']
        parameters = default_res['parameters']
        all_param_names = default_res['param_names']
        Pkmu_specz = default_res['pkmu_specz']        # includes Pspecz_sys
        Pkmu_photoz = default_res['pkmu_photoz']
        Pkmu_cross = default_res['pkmu_cross']
        V_survey = default_res['V_survey']

        sn_photoz = default_res['sn_photoz']
        sn_specz = default_res['sn_specz']

        Pkmu_specz_obs = Pkmu_specz + sn_specz
        Pkmu_photoz_obs = Pkmu_photoz + sn_photoz

        print("kmin:", kmin)

        k_width = kcenter[1] - kcenter[0]
        mu_width = mu[1] - mu[0]
        print(" {}<z{}:".format(z_low, z_up))
        print("sn_photoz, sn_specz:", sn_photoz, sn_specz)

        num_all_params = len(parameters)
        input_file = path + filename_nonlinear 
        # note that the photo-z error parameter is not relevent to spec-z forecast, which is 0 on the relevent row and column
        # s: spectroscopy, c: cross, p: photo-z
        param_id_list, param_name_list = select_params_id(all_param_names, skip_params_list) 
        
        num_params = len(param_id_list)    # number of selected parameters
        temp_matrix  = np.zeros((num_params, num_params))

        for i in range(num_params):
            dspecz_dpi, dphotoz_dpi, dcross_dpi = dp_dlambda(input_file, param_id_list[i], alpha_minus=const_low, alpha_plus=const_up)

            for j in range(i, num_params):
                dspecz_dpj, dphotoz_dpj, dcross_dpj = dp_dlambda(input_file, param_id_list[j], alpha_minus=const_low, alpha_plus=const_up)

                temp_matrix[i, j] = cal_Fmn(kcenter, mu, V_survey, Pkmu_specz_obs, Pkmu_cross, Pkmu_photoz_obs, dspecz_dpi, dcross_dpi, dspecz_dpj, dcross_dpj)

        Fisher_matrix = temp_matrix + temp_matrix.T
        np.fill_diagonal(Fisher_matrix, np.diag(temp_matrix))
        print("Fisher_matrix:", Fisher_matrix)
        
        Fisher_matrix_diffz.append(Fisher_matrix)
        
        Cov_params = linalg.inv(Fisher_matrix)
        print("The error of parameters:", np.diag(Cov_params)**0.5)
        sigma_alpha_perp, sigma_alpha_para = Cov_params[0,0]**0.5, Cov_params[1,1]**0.5
        
        cross_coeff = Cov_params[0, 1]/(Cov_params[0,0]*Cov_params[1,1])**0.5
        output_alphas_mar.append(np.array([z_low, z_up, sigma_alpha_perp, sigma_alpha_para, cross_coeff]))

        F_alphas_unmar = Fisher_matrix[0:2, 0:2]
        cov_alphas_unmar = linalg.inv(F_alphas_unmar)
        cross_coeff_unmar = cov_alphas_unmar[0, 1]/ (cov_alphas_unmar[0, 0] * cov_alphas_unmar[1, 1])**0.5
        output_alphas_unmar.append([z_low, z_up] + list(np.diag(cov_alphas_unmar)**0.5) + [cross_coeff_unmar])

        output_all_params.append([z_low, z_up] + list(np.diag(Cov_params)**0.5))

    print("alpha with marginalization:", output_alphas_mar)
    print("alpha without marginalization:", output_alphas_unmar)

    alphas_mar = np.array(output_alphas_mar)
    alphas_unmar = np.array(output_alphas_unmar)

    params_free = np.zeros(num_all_params, dtype=int)
    params_free[param_id_list] = 1       
    params_str = ''.join(str(x) for x in params_free)
    
    ofile = output_dir + "Fisher_matrix_diffz_specz_add_cross_tracer_zerror_specz{0:.3f}_photoz{1:.3f}_kmax{2:.2f}_params{3}.npz".format(sigma_specz, sigma_photoz, kmax, params_str)
    np.savez(ofile, Fisher_matrix_diffz=Fisher_matrix_diffz, zmid=(zlow_bins+zup_bins)/2.0)
    
    ofile = output_dir + "sigma_alpha_specz_add_cross_tracer_zerror_specz{0:.3f}_photoz{1:.3f}_kmax{2:.2f}_params{3}.npz".format(sigma_specz, sigma_photoz, kmax, params_str)
    np.savez(ofile, alphas_mar = alphas_mar, alphas_unmar = alphas_unmar)
    

if __name__ == '__main__':
    main()




