#!/usr/bin/env python
# coding: utf-8
# Add Pspecz_sys as an additional parameter. --10-11-2022
#
import os, sys
import logging
from pathlib import Path
import numpy as np
from scipy import linalg, integrate, interpolate
from argparse import ArgumentParser


## since the mu range in our $P(k,\mu)$ is in [0, 1], we do not need to consider 1/2 prefactor due to the same mode from k and -k.  
## calculated from 
def cal_Nmodes_kmu(k_low, k_up, mu_low, mu_up, V_survey):
    return (k_up**3.0 - k_low**3.0)*(mu_up-mu_low)/(12*np.pi**2.0) * V_survey


## dlnP_dm and inv_cov_lnP are matrices
## note that inv_cov_lnP is inverse covariance matrix of [dlnPs_dm, dlnPsp_dm, dlnPp_dm]
def cal_Fmn(kobs, mu_obs, V_survey, dlnPa_dm, dlnPab_dm, dlnPb_dm, inv_cov_lnP, dlnPa_dn, dlnPab_dn, dlnPb_dn):
    Fmn_k = np.zeros(len(kobs))
    
    for i in range(len(kobs)):
        temp = np.zeros(len(mu_obs))
        for j in range(len(mu_obs)):
            
            inv_cov = inv_cov_lnP[i, j]
            
            dlnP_dm_array = np.array([dlnPa_dm[i,j], dlnPab_dm[i,j], dlnPb_dm[i,j]])
            
            dlnP_dn_array = np.array([dlnPa_dn[i,j], dlnPab_dn[i,j], dlnPb_dn[i,j]])
            
            #temp[j] = V_survey * np.dot(np.dot(dP_dm_array, inv_cov), dP_dn_array) * (kobs[i]/(2*np.pi))**2.0 
            temp[j] = np.dot(np.dot(dlnP_dm_array, inv_cov), dlnP_dn_array)

        Fmn_k[i] = integrate.trapz(temp, mu_obs) * V_survey * (kobs[i]/(2*np.pi))**2.0
    
    Fmn = integrate.trapz(Fmn_k, kobs) 
                        
    return Fmn

def interpolate_dlnP_dparam(dlnPs_dpi_m, dlnPp_dpi_m, dlnPps_dpi_m, k_p, k_o):
    len_kp, len_mu = np.shape(dlnPs_dpi_m)[:]
    res = []
    for matrix in [dlnPs_dpi_m, dlnPp_dpi_m, dlnPps_dpi_m]: 
        new_matrix = np.zeros((len(k_o), len_mu))
        for i in range(len_mu):
            spl_dlnP_dpi = interpolate.InterpolatedUnivariateSpline(k_p, matrix[:, i]) # interpolate dlnP_dpi as a function of k
            new_matrix[:, i] = spl_dlnP_dpi(k_o)
            
        res.append(new_matrix)
    return res[:]
    


def main():
    parser = ArgumentParser(description="Fisher forecast of BAO from the combination of spec-z and the cross-correlation between spec-z and photo-z survey.")
#     parser.add_argument("--survey_area", type=float, help="The sky area (deg^2)", required=True)
    parser.add_argument("--kmax", type=float, help="The maximum k of P(k,mu).", required=True)
    parser.add_argument("--zmin", type=float, help="z minimum.", default=0.0)
    parser.add_argument("--zmax", type=float, help="z maximum.", default=1.6)
    parser.add_argument("--nzbins", type=int, help="The number of z bins.", required=True)
#     parser.add_argument("--sigma_specz", type=float, help="spec-z error sigma_0.", required=True)
#     parser.add_argument("--sigma_photoz", type=float, help="photo-z error, sigma_z.", required=True)
    parser.add_argument("--Sigma_fog", type=float, help="Finger-of-God damping term.", default=0.0, required=True)
#     parser.add_argument("--const_low", type=float, help="The lower constant multiplied on parameters.", required=True)
#     parser.add_argument("--const_up", type=float, help="The upper constant multiplied on parameters.", required=True)
    parser.add_argument("--input_dir", help="input directory.", required=True)
    parser.add_argument("--output_dir", help="output directory.", required=True)
    
    logging.basicConfig(level=logging.INFO)
    
    args = parser.parse_args()

#     survey_area = args.survey_area
    kmax = args.kmax
    #kmax = 0.5
    zmin = args.zmin
    zmax = args.zmax
    nzbins = args.nzbins
    ##sigma_specz = 0.000
#     sigma_specz = args.sigma_specz
#     sigma_photoz = args.sigma_photoz

    Sigma_fog = args.Sigma_fog

#     const_low = args.const_low
#     const_up = args.const_up
    
    logging.basicConfig(format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    zbins = np.linspace(zmin, zmax, nzbins+1)
    zlow_bins = zbins[0:-1]
    zup_bins = zbins[1:]
    zmid = (zlow_bins+zup_bins)/2.0
    
    input_dir = args.input_dir
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ##theta_names = ['alpha_perp', 'alpha_para', 'f_growthrate', 'Sigma_perp_specz', 'Sigma_perp_photoz', 'Sigma_para_specz', 'Sigma_para_photoz', 'Sigma_fog_specz', 'Sigma_fog_photoz', 'Sigma_specz_error', 'Sigma_photoz_error', 'bg_specz', 'bg_photoz']
    if Sigma_fog < 1.e-7:
        skip_params_list = ['Sigma_fog_specz', 'Sigma_fog_photoz']  # if Sigma_FoG=0
    else:
        skip_params_list = []    # we assumed all the parameters used in the real fitting
        ##skip_params_list = ['Sigma_perp_specz', 'Sigma_para_specz', 'Sigma_perp_photoz', 'Sigma_para_photoz'] 
        
    Fisher_matrix_diffz = []
    output_alphas_mar = []       # marginalize other parameters
    output_alphas_unmar = []     # unmarginalize (or fix) other parameters
    output_all_params = []
    
    theta_names = ['alpha_perp', 'alpha_para', 'f_growthrate', 'Sigma_perp_specz', 'Sigma_perp_photoz', 'Sigma_para_specz', 'Sigma_para_photoz', 'Sigma_fog_specz', 'Sigma_fog_photoz', 'Sigma_specz_error', 'Sigma_photoz_error', 'bg_specz', 'bg_photoz', 'Pspecz_sys']
    N_params = len(theta_names)
    for z_low, z_up in zip(zlow_bins, zup_bins):
        logging.info(f"{z_low:.2f}<z<{z_up:.2f}")
        filename = f"inv_cov_dlnP_specz_cross_photoz_{z_low:.2f}z{z_up:.2f}.npz"
        ifile = Path(input_dir, filename)
        data = np.load(ifile)
        k_o = data['k']     # the integration points of the Fisher matrix
        mu_o = data['mu']
        V_survey = data['V_survey']
        inv_cov_lnP = data['inv_cov_lnP']   # shape: (len_k, len_mu, 3, 3)
        
        filename = f"dlnPspecz_dparam_{z_low:.2f}z{z_up:.2f}.npz"
        ifile = Path(input_dir, filename)
        data = np.load(ifile)
        k_p = data['k']    # for interpolation
        mu_p = data['mu']  # equal to mu_o
        
        dlnPspecz_dparams = np.zeros((N_params, len(k_p), len(mu_p)))
        specz_params_id = [0, 1, 2, 3, 5, 7, 9, 11, 13]
        # shape: (len_k, len_mu)
        dlnPspecz_dalperp = data['dlnPspecz_dalperp']
        dlnPspecz_dalpara = data['dlnPspecz_dalpara']
        dlnPspecz_df = data['dlnPspecz_df']
        dlnPspecz_dSigmaperp = data['dlnPspecz_dSigmaperp']
        dlnPspecz_dSigmapara = data['dlnPspecz_dSigmapara']
        dlnPspecz_dSigmafog = data['dlnPspecz_dSigmafog']
        dlnPspecz_dSigmazerr = data['dlnPspecz_dSigmazerr']
        dlnPspecz_dbg = data['dlnPspecz_dbg'] 
        dlnPspecz_dPsys = data['dlnPspecz_dPsys']
        dlnPspecz_dparams[specz_params_id] = dlnPspecz_dalperp, dlnPspecz_dalpara, dlnPspecz_df, dlnPspecz_dSigmaperp, dlnPspecz_dSigmapara, dlnPspecz_dSigmafog, dlnPspecz_dSigmazerr, dlnPspecz_dbg, dlnPspecz_dPsys
        
        filename = f"dlnPphotoz_dparam_{z_low:.2f}z{z_up:.2f}.npz"
        ifile = Path(input_dir, filename)
        data = np.load(ifile)
        
        dlnPphotoz_dparams = np.zeros((N_params, len(k_p), len(mu_p)))
        photoz_params_id = [0, 1, 2, 4, 6, 8, 10, 12]
        # shape: (len_k, len_mu)
        dlnPphotoz_dalperp = data['dlnPphotoz_dalperp']
        dlnPphotoz_dalpara = data['dlnPphotoz_dalpara']
        dlnPphotoz_df = data['dlnPphotoz_df']
        dlnPphotoz_dSigmaperp = data['dlnPphotoz_dSigmaperp']
        dlnPphotoz_dSigmapara = data['dlnPphotoz_dSigmapara']
        dlnPphotoz_dSigmafog = data['dlnPphotoz_dSigmafog']
        dlnPphotoz_dSigmazerr = data['dlnPphotoz_dSigmazerr']
        dlnPphotoz_dbg = data['dlnPphotoz_dbg'] 
        dlnPphotoz_dparams[photoz_params_id] = dlnPphotoz_dalperp, dlnPphotoz_dalpara, dlnPphotoz_df, dlnPphotoz_dSigmaperp, dlnPphotoz_dSigmapara, dlnPphotoz_dSigmafog, dlnPphotoz_dSigmazerr, dlnPphotoz_dbg
        
        filename = f"dlnPcross_dparam_{z_low:.2f}z{z_up:.2f}.npz"
        ifile = Path(input_dir, filename)
        data = np.load(ifile)
        
        dlnPcross_dparams = np.zeros((N_params, len(k_p), len(mu_p)))
        
        dlnPcross_dalperp = data['dlnPcross_dalperp']
        dlnPcross_dalpara = data['dlnPcross_dalpara']
        dlnPcross_df = data['dlnPcross_df']
        dlnPcross_dSigmaperp_specz = data['dlnPcross_dSigmaperp_specz']
        dlnPcross_dSigmaperp_photoz = data['dlnPcross_dSigmaperp_photoz']
        dlnPcross_dSigmapara_specz = data['dlnPcross_dSigmapara_specz']
        dlnPcross_dSigmapara_photoz = data['dlnPcross_dSigmapara_photoz']
        dlnPcross_dSigmafog_specz = data['dlnPcross_dSigmafog_specz']
        dlnPcross_dSigmafog_photoz = data['dlnPcross_dSigmafog_photoz']
        dlnPcross_dSigmazerr_specz = data['dlnPcross_dSigmazerr_specz']
        dlnPcross_dSigmazerr_photoz = data['dlnPcross_dSigmazerr_photoz']
        dlnPcross_dbg_specz = data['dlnPcross_dbg_specz']
        dlnPcross_dbg_photoz = data['dlnPcross_dbg_photoz']
        ## Pcross does not include Psys
        dlnPcross_dparams[0:-1] = dlnPcross_dalperp, dlnPcross_dalpara, dlnPcross_df, dlnPcross_dSigmaperp_specz, dlnPcross_dSigmaperp_photoz, dlnPcross_dSigmapara_specz, dlnPcross_dSigmapara_photoz, dlnPcross_dSigmafog_specz, dlnPcross_dSigmafog_photoz, dlnPcross_dSigmazerr_specz, dlnPcross_dSigmazerr_photoz, dlnPcross_dbg_specz, dlnPcross_dbg_photoz
        
        temp_matrix = np.zeros((N_params, N_params))
        for i in range(N_params):
            dlnPspecz_dpi, dlnPphotoz_dpi, dlnPcross_dpi = interpolate_dlnP_dparam(dlnPspecz_dparams[i], dlnPphotoz_dparams[i], dlnPcross_dparams[i], k_p, k_o)

            for j in range(i, N_params):
                dlnPspecz_dpj, dlnPphotoz_dpj, dlnPcross_dpj = interpolate_dlnP_dparam(dlnPspecz_dparams[j], dlnPphotoz_dparams[j], dlnPcross_dparams[j], k_p, k_o)

                temp_matrix[i, j] = cal_Fmn(k_o, mu_o, V_survey, dlnPspecz_dpi, dlnPcross_dpi, dlnPphotoz_dpi, inv_cov_lnP, dlnPspecz_dpj, dlnPcross_dpj, dlnPphotoz_dpj)
                

        Fisher_matrix = temp_matrix + temp_matrix.T
        np.fill_diagonal(Fisher_matrix, np.diag(temp_matrix))

        Fisher_matrix_diffz.append(Fisher_matrix)
        
        Cov_params = linalg.inv(Fisher_matrix)
        logger.info(f"Fisher matrix inverse successful: {np.allclose(np.identity(N_params), np.dot(Cov_params, Fisher_matrix))}")

        logger.info(f"The error of parameters: {np.diag(Cov_params)**0.5}")
        sigma_alpha_perp, sigma_alpha_para = Cov_params[0,0]**0.5, Cov_params[1,1]**0.5
        
        cross_coeff = Cov_params[0, 1]/(Cov_params[0,0]*Cov_params[1,1])**0.5
        output_alphas_mar.append(np.array([z_low, z_up, sigma_alpha_perp, sigma_alpha_para, cross_coeff]))

        F_alphas_unmar = Fisher_matrix[0:2, 0:2]
        cov_alphas_unmar = linalg.inv(F_alphas_unmar)
        cross_coeff_unmar = cov_alphas_unmar[0, 1]/ (cov_alphas_unmar[0, 0] * cov_alphas_unmar[1, 1])**0.5
        output_alphas_unmar.append([z_low, z_up] + list(np.diag(cov_alphas_unmar)**0.5) + [cross_coeff_unmar])

        output_all_params.append([z_low, z_up] + list(np.diag(Cov_params)**0.5))

    logger.info(f"alpha with marginalization: {output_alphas_mar}")
    logger.info(f"alpha without marginalization: {output_alphas_unmar}")

    alphas_mar = np.array(output_alphas_mar)
    alphas_unmar = np.array(output_alphas_unmar)

#     params_free = np.zeros(num_all_params, dtype=int)
#     params_free[param_id_list] = 1
#     params_str = ''.join(str(x) for x in params_free)
    
#    ofile = output_dir + "Fisher_matrix_diffz_specz_photoz_add_cross_tracer_zerror_specz{0:.3f}_photoz{1:.3f}_kmax{2:.2f}_params{3}.npz".format(sigma_specz, sigma_photoz, kmax, params_str)
    ofile = Path(output_dir, f"Fisher_matrix_diffz_specz_photoz_add_cross_kmax{kmax:.2f}.npz")
    np.savez(ofile, Fisher_matrix_diffz=Fisher_matrix_diffz, zmid=zmid, theta_names=theta_names)
    
    ofile = Path(output_dir, f"sigma_alpha_specz_photoz_add_cross_kmax{kmax:.2f}.npz")
    np.savez(ofile, alphas_mar = alphas_mar, alphas_unmar = alphas_unmar)
    
if __name__ == '__main__':
    main()





