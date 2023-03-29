#!/usr/bin/env python
# coding: utf-8
# Include the factor Pg/(Pg+Psys) for the logarithmic derivative of Pspecz to account for the addition of Psys. Add Psys as a free parameter. --10-11-2022
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

# k_t and k_o denotes the true and fiducial (observed) coordinate, respectively
def cal_Pauto_BAOdamped(ln_kt, ln_ko, k_o, mu, Sigma_perp, Sigma_para, spl_lnPlin, spl_lnPsm):
    res = (np.exp(spl_lnPlin(ln_kt)) - np.exp(spl_lnPsm(ln_kt)))*BAO_damping(k_o, mu, Sigma_perp, Sigma_para)**2.0 + np.exp(spl_lnPsm(ln_ko))
    return res

def cal_Pcross_BAOdamped(ln_kt, ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm):
    term1 = np.exp(spl_lnPlin(ln_kt)) - np.exp(spl_lnPsm(ln_kt))
    term2 = BAO_damping(k_o, mu, Sigma_perp_specz, Sigma_para_specz) * BAO_damping(k_o, mu, Sigma_perp_photoz, Sigma_para_photoz)
    term3 = np.exp(spl_lnPsm(ln_ko))
    return term1 * term2 + term3

def dlnPauto_dSigmafog(k, mu, Sigma_fog):
    res = 4 * (F_fog(k, mu, Sigma_fog)-1.0)/Sigma_fog
    return res

def dlnPcross_dSigmafog(k, mu, Sigma_fog):
    res = 2 * (F_fog(k, mu, Sigma_fog)-1.0)/Sigma_fog
    return res

def dlnPauto_dSigmazerr(k, mu, Sigma_zerr):
    return -2*(k*mu)**2.0 * Sigma_zerr

def dlnPcross_dSigmazerr(k, mu, Sigma_zerr):
    return -(k*mu)**2.0 * Sigma_zerr    


def dlnPauto_dSigmaperp(ln_ko, k_o, mu, Sigma_perp, Sigma_para, spl_lnPlin, spl_lnPsm):
    Pbao_nl_sm = cal_Pauto_BAOdamped(ln_ko, ln_ko, k_o, mu, Sigma_perp, Sigma_para, spl_lnPlin, spl_lnPsm)
    Psm = np.exp(spl_lnPsm(ln_ko))
    Pbao_nl = Pbao_nl_sm - Psm
    term1 = Pbao_nl/Pbao_nl_sm
    term2 = -k_o**2.0*(1-mu**2.0) * Sigma_perp
    return term1 * term2

def cal_dlnPcross_dSigmaperp_specz(ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm):
    Pbao_nl_sm = cal_Pcross_BAOdamped(ln_ko, ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm)
    Psm = np.exp(spl_lnPsm(ln_ko))
    Pbao_nl = Pbao_nl_sm - Psm
    term1 = Pbao_nl/Pbao_nl_sm
    term2 = -0.5*k_o**2.0 * (1-mu**2.0) * Sigma_perp_specz
    return term1 * term2

def cal_dlnPcross_dSigmaperp_photoz(ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm):
    Pbao_nl_sm = cal_Pcross_BAOdamped(ln_ko, ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm)
    Psm = np.exp(spl_lnPsm(ln_ko))
    Pbao_nl = Pbao_nl_sm - Psm
    term1 = Pbao_nl/Pbao_nl_sm
    term2 = -0.5*k_o**2.0 * (1-mu**2.0) * Sigma_perp_photoz
    return term1 * term2

def dlnPauto_dSigmapara(ln_ko, k_o, mu, Sigma_perp, Sigma_para, spl_lnPlin, spl_lnPsm):
    Pbao_nl_sm = cal_Pauto_BAOdamped(ln_ko, ln_ko, k_o, mu, Sigma_perp, Sigma_para, spl_lnPlin, spl_lnPsm)
    Psm = np.exp(spl_lnPsm(ln_ko))
    Pbao_nl = Pbao_nl_sm - Psm
    term1 = Pbao_nl/Pbao_nl_sm
    term2 = -(k_o*mu)**2.0 * Sigma_para
    return term1 * term2

def cal_dlnPcross_dSigmapara_specz(ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm):
    Pbao_nl_sm = cal_Pcross_BAOdamped(ln_ko, ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm)
    Psm = np.exp(spl_lnPsm(ln_ko))
    Pbao_nl = Pbao_nl_sm - Psm
    term1 = Pbao_nl/Pbao_nl_sm
    term2 = -0.5*(k_o*mu)**2.0 * Sigma_para_specz
    return term1 * term2

def cal_dlnPcross_dSigmapara_photoz(ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm):
    Pbao_nl_sm = cal_Pcross_BAOdamped(ln_ko, ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm)
    Psm = np.exp(spl_lnPsm(ln_ko))
    Pbao_nl = Pbao_nl_sm - Psm
    term1 = Pbao_nl/Pbao_nl_sm
    term2 = -0.5*(k_o*mu)**2.0 * Sigma_para_photoz
    return term1 * term2


def dlnPauto_df(mu, f, bg):
    return 2.0*mu**2.0/(bg+f*mu**2.0)

def cal_dlnPcross_df(mu, f, bg_specz, bg_photoz):
    fmu2 = f*mu**2.0
    return mu**2.0/(bg_specz + fmu2) + mu**2.0/(bg_photoz + fmu2)

def dlnPauto_dbg(mu, f, bg):
    return 2.0/(bg + f*mu**2.0)

def dlnPcross_dbg(mu, f, bg):
    return 1.0/(bg + f*mu**2.0)


def pkmu_linear_model(theta, ln_ko, mu_o, norm_gf, spl_lnPlin):
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


def main():
    parser = ArgumentParser(description="Calculate the predicted (pre-reconstruction) P(k,mu) for the spec-z, photo-z and cross ones.")
    parser.add_argument("--survey_area", type=float, help="The sky area (deg^2)", required=True)
#     parser.add_argument("--kmax", type=float, help="The maximum k of P(k,mu).", required=True)
#     parser.add_argument("--k_width", type=float, help="The k step width", required=True)
    parser.add_argument("--zmin", type=float, help="z minimum.", default=0.0)
    parser.add_argument("--zmax", type=float, help="z maximum.", default=1.6)
    parser.add_argument("--nzbins", type=int, help="The number of z bins.", required=True)
    parser.add_argument("--ngalspecz_file", type=str, help="input file contains Ngal in each spec-z bin.", required=True)
    parser.add_argument("--ngalphotoz_file", type=str, help="input file contains Ngal in each photo-z bin.", required=True)
    parser.add_argument("--sigma_specz", type=float, help="spec-z error sigma_0.", required=True)
    parser.add_argument("--sigma_photoz", type=float, help="photo-z error, sigma_z.", required=True)
    parser.add_argument("--Sigma_fog", type=float, help="Finger-of-God damping term.", default=0.0, required=True)
    parser.add_argument("--const_low", type=float, help="The lower constant multiplied on parameters.", required=True)
    parser.add_argument("--const_up", type=float, help="The upper constant multiplied on parameters.", required=True)
    parser.add_argument("--with_recon", type=str, help="With/without the density field reconstruction.", default="True", required=True)
    parser.add_argument("--f0eff", type=float, help="The fraction of good redshift measurement at z=0.", default=0.5, required=True)
    parser.add_argument("--Pspecz_sys", type=float, help="Systemtic noise of spec-z data from grism redshift measurement", required=True)
    parser.add_argument("--input_dir", help="input directory for some parameters", required=True)
    parser.add_argument("--output_dir", help="output directory.", required=True)
    parser.add_argument("--photoz_params_bf", help="The boost factor for the photo-z parameters.", type=float, default=1.0)
    parser.add_argument("--photoz_Sigma_perp_bf", help="The boost factor for the photo-z parameter Sigma_perp.", type=float, default=1.0)
    
    args = parser.parse_args()
    survey_area = args.survey_area
#     kmax = args.kmax
#     k_width = args.k_width    # k bin width
    zmin = args.zmin
    zmax = args.zmax
    nzbins = args.nzbins
    const_low = args.const_low
    const_up = args.const_up
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
    
    # set up the observed k array
    k_o = np.logspace(-3.9, -0.09, 1000)  # kmin=1.28e-4, kmax=0.81
    len_k = len(k_o)
    ln_ko = np.log(k_o)
    ln_kp = ln_ko * const_up
    ln_km = ln_ko * const_low
    
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
        logger.info(f"{z_low:.2f} < z < {z_up:.2f}")
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
        
        n_specz = Ngal_specz/(fsky_zcosmos * V_total) * f0eff /(1.0 + z_mid)
        
        # calculate Plin
        
        theta = np.array([1.0, 1.0, f_growthrate, bias])
        theta_names=['alpha_perp', 'alpha_para', 'f_growthrate', 'gal_bias']
        pkmu_linear = pkmu_linear_model(theta, ln_ko, mu_o, norm_gf, spl_lnPlin)
        
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
        Pspecz_pure = Pkmu_auto_model(theta, k_o, ln_ko, mu_o, norm_gf, BAOonly, spl_lnPlin, spl_lnPsm)
        
        
        # In the real analysis, we distinguish some (nuisance) parameters for multi-tracers. 
        theta = np.array([1.0, 1.0, f_growthrate, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, Sigma_fog_specz, Sigma_fog_photoz, Sigma_specz_error, Sigma_photoz_error, bg_specz, bg_photoz, Pspecz_sys])
        logger.info(f"theta for Pcross: {theta}")

        theta_names = ['alpha_perp', 'alpha_para', 'f_growthrate', 'Sigma_perp_specz', 'Sigma_perp_photoz', 'Sigma_para_specz', 'Sigma_para_photoz', 'Sigma_fog_specz', 'Sigma_fog_photoz', 'Sigma_specz_error', 'Sigma_photoz_error', 'bg_specz', 'bg_photoz', 'Pspecz_sys']
        
        dlnPspecz_dalperp = np.zeros((len_k, len_mu))
        dlnPspecz_dalpara = np.zeros((len_k, len_mu))
        dlnPspecz_dSigmafog = np.zeros((len_k, len_mu))
        dlnPspecz_dSigmazerr = np.zeros((len_k, len_mu))
        dlnPspecz_dSigmaperp = np.zeros((len_k, len_mu))
        dlnPspecz_dSigmapara = np.zeros((len_k, len_mu))
        dlnPspecz_df = np.zeros((len_k, len_mu))
        dlnPspecz_dbg = np.zeros((len_k, len_mu))
        dlnPspecz_dPsys = np.zeros((len_k, len_mu))
        
        dlnPphotoz_dalperp = np.zeros((len_k, len_mu))
        dlnPphotoz_dalpara = np.zeros((len_k, len_mu))
        dlnPphotoz_dSigmafog = np.zeros((len_k, len_mu))
        dlnPphotoz_dSigmazerr = np.zeros((len_k, len_mu))
        dlnPphotoz_dSigmaperp = np.zeros((len_k, len_mu))
        dlnPphotoz_dSigmapara = np.zeros((len_k, len_mu)) 
        dlnPphotoz_df = np.zeros((len_k, len_mu))
        dlnPphotoz_dbg = np.zeros((len_k, len_mu))
        
        dlnPcross_dalperp = np.zeros((len_k, len_mu))
        dlnPcross_dalpara = np.zeros((len_k, len_mu))
        dlnPcross_dSigmafog_specz = np.zeros((len_k, len_mu))
        dlnPcross_dSigmafog_photoz = np.zeros((len_k, len_mu))
        dlnPcross_dSigmazerr_specz = np.zeros((len_k, len_mu))
        dlnPcross_dSigmazerr_photoz = np.zeros((len_k, len_mu))
        dlnPcross_dSigmaperp_specz = np.zeros((len_k, len_mu))
        dlnPcross_dSigmaperp_photoz = np.zeros((len_k, len_mu))
        dlnPcross_dSigmapara_specz = np.zeros((len_k, len_mu)) 
        dlnPcross_dSigmapara_photoz = np.zeros((len_k, len_mu)) 
        dlnPcross_df = np.zeros((len_k, len_mu))
        dlnPcross_dbg_specz = np.zeros((len_k, len_mu))
        dlnPcross_dbg_photoz = np.zeros((len_k, len_mu))
         
        for i, mu in enumerate(mu_o):

            Psys_factor = Pspecz_pure[:, i]/(Pspecz_pure[:, i] + Pspecz_sys)  # only applies to to spec-z 
            
            lnPspecz_BAOdamped_p = np.log(cal_Pauto_BAOdamped(ln_kp, ln_ko, k_o, mu, Sigma_perp_specz, Sigma_para_specz, spl_lnPlin, spl_lnPsm)) 
            lnPspecz_BAOdamped_m = np.log(cal_Pauto_BAOdamped(ln_km, ln_ko, k_o, mu, Sigma_perp_specz, Sigma_para_specz, spl_lnPlin, spl_lnPsm))
            dlnPspecz_BAOdamped_dlnk = (lnPspecz_BAOdamped_p - lnPspecz_BAOdamped_m)/(ln_kp - ln_km)
            dlnPspecz_dalperp[:, i] = dlnPspecz_BAOdamped_dlnk * (mu**2.0-1.0) * Psys_factor
            dlnPspecz_dalpara[:, i] = dlnPspecz_BAOdamped_dlnk * (-mu**2.0) * Psys_factor
            
            dlnPspecz_dSigmafog[:, i] = dlnPauto_dSigmafog(k_o, mu, Sigma_fog_specz) * Psys_factor
            dlnPspecz_dSigmazerr[:, i] = dlnPauto_dSigmazerr(k_o, mu, Sigma_specz_error) * Psys_factor
            dlnPspecz_dSigmaperp[:, i] = dlnPauto_dSigmaperp(ln_ko, k_o, mu, Sigma_perp_specz, Sigma_para_specz, spl_lnPlin, spl_lnPsm) * Psys_factor
            dlnPspecz_dSigmapara[:, i] = dlnPauto_dSigmapara(ln_ko, k_o, mu, Sigma_perp_specz, Sigma_para_specz, spl_lnPlin, spl_lnPsm) * Psys_factor
            dlnPspecz_df[:, i] = dlnPauto_df(mu, f_growthrate, bg_specz) * Psys_factor
            dlnPspecz_dbg[:, i] = dlnPauto_dbg(mu, f_growthrate, bg_specz) * Psys_factor
            dlnPspecz_dPsys[:, i] = 1.0/(Pspecz_pure[:, i] + Pspecz_sys) 
            
            lnPphotoz_BAOdamped_p = np.log(cal_Pauto_BAOdamped(ln_kp, ln_ko, k_o, mu, Sigma_perp_photoz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm)) 
            lnPphotoz_BAOdamped_m = np.log(cal_Pauto_BAOdamped(ln_km, ln_ko, k_o, mu, Sigma_perp_photoz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm))
            dlnPphotoz_BAOdamped_dlnk = (lnPphotoz_BAOdamped_p - lnPphotoz_BAOdamped_m)/(ln_kp - ln_km)
            dlnPphotoz_dalperp[:, i] = dlnPphotoz_BAOdamped_dlnk * (mu**2.0-1.0)
            dlnPphotoz_dalpara[:, i] = dlnPphotoz_BAOdamped_dlnk * (-mu**2.0)
            
            dlnPphotoz_dSigmafog[:, i] = dlnPauto_dSigmafog(k_o, mu, Sigma_fog_photoz)
            dlnPphotoz_dSigmazerr[:, i] = dlnPauto_dSigmazerr(k_o, mu, Sigma_photoz_error)
            dlnPphotoz_dSigmaperp[:, i] = dlnPauto_dSigmaperp(ln_ko, k_o, mu, Sigma_perp_photoz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm)
            dlnPphotoz_dSigmapara[:, i] = dlnPauto_dSigmapara(ln_ko, k_o, mu, Sigma_perp_photoz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm)
            dlnPphotoz_df[:, i] = dlnPauto_df(mu, f_growthrate, bg_photoz)
            dlnPphotoz_dbg[:, i] = dlnPauto_dbg(mu, f_growthrate, bg_photoz)
            
            lnPcross_BAOdamped_p = np.log(cal_Pcross_BAOdamped(ln_kp, ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm)) 
            lnPcross_BAOdamped_m = np.log(cal_Pcross_BAOdamped(ln_km, ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm))
            dlnPcross_BAOdamped_dlnk = (lnPcross_BAOdamped_p - lnPcross_BAOdamped_m)/(ln_kp - ln_km)
            dlnPcross_dalperp[:, i] = dlnPcross_BAOdamped_dlnk * (mu**2.0-1.0)
            dlnPcross_dalpara[:, i] = dlnPcross_BAOdamped_dlnk * (-mu**2.0)
            
            dlnPcross_dSigmafog_specz[:, i] = dlnPcross_dSigmafog(k_o, mu, Sigma_fog_specz)
            dlnPcross_dSigmafog_photoz[:, i] = dlnPcross_dSigmafog(k_o, mu, Sigma_fog_photoz)
            dlnPcross_dSigmazerr_specz[:, i] = dlnPcross_dSigmazerr(k_o, mu, Sigma_specz_error)
            dlnPcross_dSigmazerr_photoz[:, i] = dlnPcross_dSigmazerr(k_o, mu, Sigma_photoz_error)
            dlnPcross_dSigmaperp_specz[:, i] = cal_dlnPcross_dSigmaperp_specz(ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm)
            dlnPcross_dSigmaperp_photoz[:, i] = cal_dlnPcross_dSigmaperp_photoz(ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm)
            dlnPcross_dSigmapara_specz[:, i] = cal_dlnPcross_dSigmapara_specz(ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm)
            dlnPcross_dSigmapara_photoz[:, i] = cal_dlnPcross_dSigmapara_photoz(ln_ko, k_o, mu, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, spl_lnPlin, spl_lnPsm)
            
            
            dlnPcross_df[:, i] = cal_dlnPcross_df(mu, f_growthrate, bg_specz, bg_photoz)
            dlnPcross_dbg_specz[:, i] = dlnPcross_dbg(mu, f_growthrate, bg_specz)
            dlnPcross_dbg_photoz[:, i] = dlnPcross_dbg(mu, f_growthrate, bg_photoz)
            
        filename = f"dlnPspecz_dparam_{z_low:.2f}z{z_up:.2f}.npz"
        ofile = Path(odir, filename)
        np.savez(ofile, k=k_o, mu=mu_o, theta=theta, theta_names=theta_names, dlnPspecz_dalperp=dlnPspecz_dalperp, dlnPspecz_dalpara=dlnPspecz_dalpara,\
                 dlnPspecz_dSigmafog=dlnPspecz_dSigmafog, dlnPspecz_dSigmazerr=dlnPspecz_dSigmazerr, dlnPspecz_dSigmaperp=dlnPspecz_dSigmaperp,\
                 dlnPspecz_dSigmapara=dlnPspecz_dSigmapara, dlnPspecz_df=dlnPspecz_df, dlnPspecz_dbg=dlnPspecz_dbg, dlnPspecz_dPsys=dlnPspecz_dPsys)
        
        filename = f"dlnPphotoz_dparam_{z_low:.2f}z{z_up:.2f}.npz"
        ofile = Path(odir, filename)
        np.savez(ofile, k=k_o, mu=mu_o, theta=theta, theta_names=theta_names, dlnPphotoz_dalperp=dlnPphotoz_dalperp, dlnPphotoz_dalpara=dlnPphotoz_dalpara,\
                 dlnPphotoz_dSigmafog=dlnPphotoz_dSigmafog, dlnPphotoz_dSigmazerr=dlnPphotoz_dSigmazerr, dlnPphotoz_dSigmaperp=dlnPphotoz_dSigmaperp,\
                 dlnPphotoz_dSigmapara=dlnPphotoz_dSigmapara, dlnPphotoz_df=dlnPphotoz_df, dlnPphotoz_dbg=dlnPphotoz_dbg)
        
        filename = f"dlnPcross_dparam_{z_low:.2f}z{z_up:.2f}.npz"
        ofile = Path(odir, filename)
        np.savez(ofile, k=k_o, mu=mu_o, theta=theta, theta_names=theta_names, dlnPcross_dalperp=dlnPcross_dalperp, dlnPcross_dalpara=dlnPcross_dalpara,\
                 dlnPcross_dSigmafog_specz=dlnPcross_dSigmafog_specz, dlnPcross_dSigmafog_photoz=dlnPcross_dSigmafog_photoz,\
                 dlnPcross_dSigmazerr_specz=dlnPcross_dSigmazerr_specz, dlnPcross_dSigmazerr_photoz=dlnPcross_dSigmazerr_photoz,\
                 dlnPcross_dSigmaperp_specz=dlnPcross_dSigmaperp_specz, dlnPcross_dSigmaperp_photoz=dlnPcross_dSigmaperp_photoz,\
                 dlnPcross_dSigmapara_specz=dlnPcross_dSigmapara_specz, dlnPcross_dSigmapara_photoz=dlnPcross_dSigmapara_photoz,\
                 dlnPcross_df=dlnPcross_df, dlnPcross_dbg_specz=dlnPcross_dbg_specz, dlnPcross_dbg_photoz=dlnPcross_dbg_photoz)
        
        
        

if __name__ == '__main__':
    main()




