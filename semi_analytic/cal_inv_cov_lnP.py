#!/usr/bin/env python
# coding: utf-8
# Calculate the inverse covariance corresponding to lnP, i.e. dlnP/dqi Cov^{-1} dlnP/dqj. --10-02-2022
# To test the effect from Pspecz_sys, we add Pspecz_sys on Pspecz_true. --10-06-2022
#
import os, sys
import logging
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from functools import reduce
from scipy import interpolate, integrate, linalg
sys.path.append("/home/zjding/csst_bao/fisher_pkmu/")
from mcmc_funs import growth_factor
import matplotlib.pyplot as plt
# sys.path.append("/home/zjding/csst_bao/fisher_pkmu/pkmu_model/")
# from pkmu_prerec import match_params
from argparse import ArgumentParser


# define D_H as 1/H(z)
def D_H(z, Omega_m):
    return 1./(100. * ((1.0+z)**3.0*Omega_m + 1.0 - Omega_m)**0.5)

# k, mu denote the observed coordinates
def F_rsd(k, mu, b_g, f, Sigma_fog):
    return (b_g + f * mu**2) * F_fog(k, mu, Sigma_fog)
    
def F_fog(k, mu, Sigma_fog):
    return 1.0/(1.0 + (k*mu*Sigma_fog)**2.0/2.0)

def F_zerr(k, mu, Sigma_zerr):
    return np.exp(-(k * mu * Sigma_zerr)**2.0/2.0)

def BAO_damping(k, mu, Sigma_perp, Sigma_para):
    return np.exp(-((k * mu * Sigma_para)**2 + k**2 * (1-mu**2.0)*Sigma_perp**2.0)/4.0)  # note that the denominator is 4.0

def Pkmu_linear_model(theta, ln_ko, mu_o, norm_gf, spl_lnPlin):
    alpha_1, alpha_2, f, bg = theta[:]
    dim_mu = len(mu_o)
    dim_k = len(ln_ko)
    Pkmu_model = np.zeros((dim_k, dim_mu))
    for j in range(dim_mu):
        for i in range(dim_k):
            Kaiser_term = (bg + f * mu_o[j]**2.0)**2.0
            Pkmu_model[i,j] = norm_gf**2.0 * Kaiser_term * np.exp(spl_lnPlin(ln_ko[i]))
            
    return Pkmu_model

def Pkmu_auto_model(theta, k, ln_k, mu, norm_gf, BAOonly, spl_lnPlin, spl_lnPsm): 
    f, Sigma_perp, Sigma_para, Sigma_fog, Sigma_zerr, bg = theta[:]
    dim_mu = len(mu)
    dim_k = len(k)
    Pkmu_model = np.zeros((dim_k, dim_mu))
    for j in range(dim_mu):
        for i in range(dim_k):
            Kaiser_term = (bg + f * mu[j]**2.0)**2.0
            Psm = np.exp(spl_lnPsm(ln_k[i]))
            Pbao_linear = np.exp(spl_lnPlin(ln_k[i])) - Psm
            bao_damping_factor = BAO_damping(k[i], mu[j], Sigma_perp, Sigma_para)**2.0
            
            if BAOonly == True:
                # Linear model added with BAO damping 
                # Differed from Lado's code, we only consider alphas in Kaiser term
                Pkmu_model[i,j] = norm_gf**2.0 * Kaiser_term * (Pbao_linear * bao_damping_factor + Psm)
            else:       
                # model considers redshift error and Finger-of-God damping
                # Lorentz form for the Finger-of-God effect
                fog_damping = F_fog(k[i], mu[j], Sigma_fog)**2.0
                # spec-z error term
                serr_damping = F_zerr(k[i], mu[j], Sigma_zerr)**2.0
  
                Pkmu_model[i,j] = norm_gf**2.0 * Kaiser_term * (Pbao_linear * bao_damping_factor + Psm) * fog_damping * serr_damping
        
    return Pkmu_model

# We can consider the density field in spec-z post-reconstruction, and the density field in photo-z pre-reconstruction
def Pkmu_cross_model(theta, k, ln_k, mu, norm_gf, BAOonly, spl_lnPlin, spl_lnPsm): 
    # [f_growthrate, Sigma_xy_recon, Sigma_perp_photoz, Sigma_z_recon, Sigma_para_photoz, Sigma_fog_specz, Sigma_fog_photoz, Sigma_specz_error, Sigma_photoz_error, bg_specz, bg_photoz]
    # for now assume the cross Pk uses the same FoG term as the spec-z one
    f, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, Sigma_fog_specz, Sigma_fog_photoz, Sigma_specz_error, Sigma_photoz_error, bg_specz, bg_photoz = theta[:]
    dim_mu = len(mu)
    dim_k = len(k)
    Pkmu_model = np.zeros((dim_k, dim_mu))
    for j in range(dim_mu):
        for i in range(dim_k):
            Kaiser_term = (bg_specz + f * mu[j]**2.0)*(bg_photoz + f * mu[j]**2.0)
            Psm = np.exp(spl_lnPsm(ln_k[i]))
            Pbao_linear = np.exp(spl_lnPlin(ln_k[i])) - Psm
            bao_damping_factor = BAO_damping(k[i], mu[j], Sigma_perp_specz, Sigma_para_specz) * BAO_damping(k[i], mu[j], Sigma_perp_photoz, Sigma_para_photoz)
            
            if BAOonly == True:
                Pkmu_model[i,j] = norm_gf**2.0 * Kaiser_term * (Pbao_linear * bao_damping_factor + Psm) 
            else:                
                ##fog_damping = 1.0/(1.0 + (k_o[i] * mu_o[j] * Sigma_fog)**2.0/2.0)**2.0 
                fog_damping = F_fog(k[i], mu[j], Sigma_fog_specz)*F_fog(k[i], mu[j], Sigma_fog_photoz)
                # note that there is a factor 2 in the denominator
                zerr_damping = F_zerr(k[i], mu[j], Sigma_specz_error) * F_zerr(k[i], mu[j], Sigma_photoz_error)

                Pkmu_model[i,j] = norm_gf**2.0 * Kaiser_term * (Pbao_linear * bao_damping_factor + Psm)* fog_damping * zerr_damping
        
    return Pkmu_model

def inv_cov_model(Ps, Psp, Pp, sn_specz, sn_photoz):
    Ps_h = Ps + sn_specz
    Pp_h = Pp + sn_photoz

    A = np.zeros((3, 3))
    A[0, 0] = (Pp_h * Ps)**2
    A[0, 1] = -2 * Pp_h * Ps * Psp**2
    A[0, 2] = Ps * Pp * Psp**2
    A[1, 0] = A[0, 1]
    A[1, 1] = 2*(Ps_h * Pp_h + Psp*Psp)*Psp**2
    A[1, 2] = -2*Ps_h * Psp**2 * Pp
    A[2, 0] = A[0, 2]
    A[2, 1] = A[1, 2]
    A[2, 2] = (Ps_h * Pp)**2
    A = A/(Ps_h * Pp_h - Psp**2)**2

    return A
    

def main():
    parser = ArgumentParser(description="Calculate the predicted (pre-reconstruction) P(k,mu) for the spec-z, photo-z and cross ones.")
    parser.add_argument("--survey_area", type=float, help="The sky area (deg^2)", required=True)
    parser.add_argument("--kmax", type=float, help="The maximum k of P(k,mu).", required=True)
    parser.add_argument("--k_width", type=float, help="The k step width", required=True)
    parser.add_argument("--zmin", type=float, help="z minimum.", default=0.0)
    parser.add_argument("--zmax", type=float, help="z maximum.", default=1.6)
    parser.add_argument("--nzbins", type=int, help="The number of z bins.", required=True)
    parser.add_argument("--ngalspecz_file", type=str, help="input file contains Ngal in each spec-z bin.", required=True)
    parser.add_argument("--ngalphotoz_file", type=str, help="input file contains Ngal in each photo-z bin.", required=True)
    parser.add_argument("--sigma_specz", type=float, help="spec-z error sigma_0.", required=True)
    parser.add_argument("--sigma_photoz", type=float, help="photo-z error, sigma_z.", required=True)
    parser.add_argument("--Sigma_fog", type=float, help="Finger-of-God damping term.", default=0.0, required=True)
#     parser.add_argument("--const_low", type=float, help="The lower constant multiplied on parameters.", required=True)
#     parser.add_argument("--const_up", type=float, help="The upper constant multiplied on parameters.", required=True)
    parser.add_argument("--with_recon", type=str, help="With/without the density field reconstruction.", default="True", required=True)
    parser.add_argument("--f0eff", type=float, help="The fraction of good redshift measurement at z=0.", default=0.5, required=True)
    parser.add_argument("--Pspecz_sys", type=float, help="Systemtic noise of spec-z data from grism redshift measurement", required=True)
    parser.add_argument("--input_dir", help="input directory for some parameters", required=True)
    parser.add_argument("--output_dir", help="output directory.", required=True)
    parser.add_argument("--photoz_params_bf", help="The boost factor for the photo-z parameters.", type=float, default=1.0)
    parser.add_argument("--photoz_Sigma_perp_bf", help="The boost factor for the photo-z parameter Sigma_perp.", type=float, default=1.0)
    
    args = parser.parse_args()
    survey_area = args.survey_area
    kmax = args.kmax
    k_width = args.k_width    # k bin width
    zmin = args.zmin
    zmax = args.zmax
    nzbins = args.nzbins
#     const_low = args.const_low
#     const_up = args.const_up
    f0eff = args.f0eff
    Pspecz_sys = args.Pspecz_sys
    input_dir = args.input_dir
    photoz_params_bf = args.photoz_params_bf
    odir = args.output_dir
    if not os.path.exists(odir):
        os.makedirs(odir)
    
    logging.basicConfig(format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    #parameter_file = Path(input_dir, "input_params.yaml")
    parameter_file = Path(input_dir, "input_params_Zvonimir.yaml")
    yaml = YAML() 
    with open(parameter_file, 'r') as fr:
        input_params = yaml.load(fr)

    # input (theoretical) linear power spectrum
    kwig, Plin_wig = np.loadtxt(input_params['Pwig_linear'], dtype='f8', comments='#', unpack=True) 
    spl_lnPlin = interpolate.InterpolatedUnivariateSpline(np.log(kwig), np.log(Plin_wig))

    ksm, Psm = np.loadtxt(input_params['Pnow_linear'], dtype='f8', comments='#', unpack=True)
    spl_lnPsm = interpolate.InterpolatedUnivariateSpline(np.log(ksm), np.log(Psm))

    # ## estimate the survey volume 
    speed_c = 299792.458    # speed of light, km/s
    Omega_m = 0.3075
    skyarea_total = 4*np.pi * (180./np.pi)**2.0
    fsky = survey_area/skyarea_total
    G_0 = growth_factor(0.0, Omega_m)       # G_0 at z=0, normalization factor 

    # ## estimate the power spectrum damping parameter $\Sigma_{specz}$, $\Sigma_{pz}$ from the spec-z and photo-z uncertainty
    sigma_specz = args.sigma_specz
    sigma_photoz = args.sigma_photoz
    fsky_cosmos = 2.0/skyarea_total
    fsky_zcosmos = 1.7/skyarea_total
    
#     # set up the observed k array
#     k_o = np.logspace(-3.9, -0.09, 1000)  # kmin=1.28e-4, kmax=0.81
#     len_k = len(k_o)
#     ln_ko = np.log(k_o)
    
    # set up the (fiducially) observed mu array, using Nmubins=100 or 200 has no influence 
    len_mu = 101
    mu_o = np.linspace(0.0, 1.0, len_mu)
    
    ## load the spec-z and photo-z n(z) distribution
    ifile = args.ngalspecz_file
    zlow_bins, zup_bins, _, Nz_speczbins = np.loadtxt(ifile, unpack=True)
    
    ifile = args.ngalphotoz_file
    zlow_bins, zup_bins, _, Nz_photozbins = np.loadtxt(ifile, unpack=True)
    
    # load galaxy bias and growth rate
    ifile = Path(input_dir, "parameter_preparation/output/ELG_bias_growthrate_zmin{0:.1f}_zmax{1:.1f}_{2}zbins.out".format(zmin, zmax, nzbins))
    zmid_array, _, f_array = np.loadtxt(ifile, unpack=True)
    # match with the galaxy bias in Miao et al. 2022
    bias_array = 1.0 + 0.84*zmid_array
    
    # Reconstruction factor. How well will the reconstruction work depending on nP(k=0.16 h/Mpc, mu=0.6)
    # E.g. 0.0 means a perfect reconstruction, 0.5 means 50% of the nonlinear degradation will be reconstructed, 
    # 1.0 means reconstructino does not work.
    # The assumptions, methodology, and the numerical values for r_factor are from Font-Ribera et al.
    nP_array = np.array([0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 6.0, 10.0])
    r_factor = np.array([1.0, 0.9, 0.8, 0.70, 0.6, 0.55, 0.52, 0.5])
    RF = interpolate.interp1d(nP_array, r_factor)
    
    # set finger-of-god parameter as redshift independent
    Sigma_fog = args.Sigma_fog
    shotnoise_specz_list = []
    shotnoise_photoz_list = []

    for z_low, z_up, Ngal_specz, Ngal_photoz, f_growthrate, bias in zip(zlow_bins, zup_bins, Nz_speczbins, Nz_photozbins, f_array, bias_array):
        z_mid = (z_low + z_up)/2.0
        
        norm_gf = growth_factor(z_mid, Omega_m)/G_0
        logger.info(f"z_mid, Gz/G0: {z_mid}, {norm_gf}")
                           
        # we consider both the input sigma_specz and sigma_photoz as sigma_0
        Sigma_specz_error = speed_c * sigma_specz * (1.0 + z_mid) * D_H(z_mid, Omega_m)
        Sigma_photoz_error = speed_c * sigma_photoz * (1.0 + z_mid) * D_H(z_mid, Omega_m)
        print("Sigma_specz_error, Sigma_photoz_error:", Sigma_specz_error, Sigma_photoz_error)
        
        ## unit: Mpc/h
        r_low = speed_c * integrate.quad(D_H, 0., z_low, args=(Omega_m))[0]
        r_up  = speed_c * integrate.quad(D_H, 0., z_up,  args=(Omega_m))[0]
        print(r_low, r_up)

        V_total = 4./3. * np.pi *(r_up**3.0 - r_low**3.0)
        V_survey = V_total * fsky
        print("V_survey:", V_survey)
        
        n_specz = Ngal_specz/(fsky_zcosmos * V_total) * f0eff /(1.0 + z_mid)
        n_photoz = Ngal_photoz/(fsky_cosmos * V_total)  # unit: (Mpc/h)^-3 

        kmin = 2*np.pi/V_survey**(1./3)   # fundermental k 
        print("kmin:", kmin)
        
        ## assume kmin is limitted by survey volume
        #k_o = np.arange(kmin, kmax, 0.005)
        nkbins = int((kmax-kmin)/k_width)
        k_o = np.linspace(kmin, kmax, nkbins+1)
        len_k = len(k_o)
        ln_ko = np.log(k_o)
        print("# of k_o points:", len(k_o))
        
        sn_photoz = 1./n_photoz
        sn_specz = 1./n_specz
        print("sn_photoz:", sn_photoz)
        print("sn_specz:", sn_specz)
        shotnoise_specz_list.append(sn_specz)
        shotnoise_photoz_list.append(sn_photoz) 
        
        # calculate Plin
        theta = np.array([1.0, 1.0, f_growthrate, bias])
        theta_names=['alpha_perp', 'alpha_para', 'f_growthrate', 'gal_bias']
        pkmu_linear = Pkmu_linear_model(theta, ln_ko, mu_o, norm_gf, spl_lnPlin)
        
        ## load Sigma_nl distribution
        ifile = Path(input_dir, "parameter_preparation/output/Sigma_nl_z%.2f.out"%z_mid)
        data = np.loadtxt(ifile)
        Sigma_para, Sigma_perp = data[0, :]  # for pre-reconstruction
        logger.info(f"Sigma_para, Sigma_perp before recon: {Sigma_para}, {Sigma_perp}")
        
        if args.with_recon == "True":
            # consider the effect of BAO reconstruction on Sigma_xy and Sigma_z
            k_sel = 0.14
            mu_sel = 0.6

            k_id = np.argmin(np.abs(k_o-k_sel))
            mu_id = np.argmin(np.abs(mu_o-mu_sel)) 

            nP_specz = n_specz * pkmu_linear[k_id, mu_id]/0.1734
            if nP_specz < nP_array[0]:
                r_val = r_factor[0]
            elif nP_specz > nP_array[-1]:
                r_val = r_factor[-1]
            else:
                r_val = RF(nP_specz) 
        else:
            r_val = 1.0   # for pre-recon
            
        logger.info(f"r_val from BAO reconstruction: {r_val}")
        # consider the change of Sigma_perp and Sigma_para from BAO reconstruction
        Sigma_perp_specz = Sigma_perp * r_val      
        Sigma_para_specz = Sigma_para * r_val
        
        Sigma_perp_photoz = Sigma_perp*photoz_params_bf * args.photoz_Sigma_perp_bf
        Sigma_para_photoz = Sigma_para*photoz_params_bf 
        Sigma_fog_specz, Sigma_fog_photoz = Sigma_fog, Sigma_fog*photoz_params_bf 
        bg_specz, bg_photoz = bias, bias*photoz_params_bf
        
        BAOonly = False

        theta = np.array([f_growthrate, Sigma_perp_specz, Sigma_para_specz, Sigma_fog_specz, Sigma_specz_error, bg_specz])
        Pspecz_true = Pkmu_auto_model(theta, k_o, ln_ko, mu_o, norm_gf, BAOonly, spl_lnPlin, spl_lnPsm) + Pspecz_sys

        theta = np.array([f_growthrate, Sigma_perp_photoz, Sigma_para_photoz, Sigma_fog_photoz, Sigma_photoz_error, bg_photoz])
        Pphotoz_true = Pkmu_auto_model(theta, k_o, ln_ko, mu_o, norm_gf, BAOonly, spl_lnPlin, spl_lnPsm)

        # In the real analysis, we distinguish some (nuisance) parameters for multi-tracers. 
        theta = np.array([f_growthrate, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, Sigma_fog_specz, Sigma_fog_photoz, Sigma_specz_error, Sigma_photoz_error, bg_specz, bg_photoz])
        ##theta_names = ['f_growthrate', 'Sigma_perp_specz', 'Sigma_perp_photoz', 'Sigma_para_specz', 'Sigma_para_photoz', 'Sigma_fog_specz', 'Sigma_fog_photoz', 'Sigma_specz_error', 'Sigma_photoz_error', 'bg_specz', 'bg_photoz']
        logger.info(f"theta for Pcross: {theta}")
        Pcross = Pkmu_cross_model(theta, k_o, ln_ko, mu_o, norm_gf, BAOonly, spl_lnPlin, spl_lnPsm)
        
        inv_cov_all = []
        for i in range(len_k): 
            inv_cov_diffmu = []
            for j in range(len_mu):

                inv_cov = inv_cov_model(Pspecz_true[i, j], Pcross[i, j], Pphotoz_true[i, j], sn_specz, sn_photoz)
                inv_cov_diffmu.append(inv_cov)
                
            inv_cov_all.append(inv_cov_diffmu)
            
        
        filename = f"inv_cov_dlnP_specz_cross_photoz_{z_low:.2f}z{z_up:.2f}.npz"
        ofile = Path(odir, filename)
        np.savez(ofile, inv_cov_lnP=inv_cov_all, k=k_o, mu=mu_o, V_survey=V_survey)  
        
        
        
        
        

if __name__ == '__main__':
    main()




