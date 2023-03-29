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
from argparse import ArgumentParser
from pkmu_module import comoving_dis_fun, pkmu_linear_model, pkmu_specz_model, pkmu_photoz_model, pkmu_cross_model, Pkmu2d


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
    parser.add_argument("--const_low", type=float, help="The lower constant multiplied on parameters.", required=True)
    parser.add_argument("--const_up", type=float, help="The upper constant multiplied on parameters.", required=True)
    parser.add_argument("--with_recon", type=str, help="With/without the density field reconstruction.", default="True", required=True)
    parser.add_argument("--f0eff", type=float, help="The fraction of good redshift measurement at z=0.", default=0.5, required=True)
    parser.add_argument("--Pspecz_sys", type=float, help="Systemtic noise of spec-z data from grism redshift measurement", required=True)
    parser.add_argument("--input_dir", help="input directory for some parameters", required=True)
    parser.add_argument("--output_dir", help="output directory.", required=True)
    parser.add_argument("--photoz_params_bf", help="The boost factor for the photo-z parameters.", type=float, default=1.0)
    parser.add_argument("--photoz_Sigma_perp_bf", help="The boost factor for the photo-z Sigma_perp.", type=float, default=1.0)
    parser.add_argument("--photoz_Sigma_para_bf", help="The boost factor for the photo-z Sigma_para.", type=float, default=1.0)
    parser.add_argument("--photoz_Sigma_fog_bf", help="The boost factor for the photo-z Sigma_fog.", type=float, default=1.0)
    parser.add_argument("--photoz_bg_bf", help="The boost factor for the photo-z bg.", type=float, default=1.0)
    parser.add_argument("--specz_Sigma_perp_bf", help="The boost factor for the spec-z Sigma_perp.", type=float, default=1.0)
    parser.add_argument("--specz_Sigma_para_bf", help="The boost factor for the spec-z Sigma_para.", type=float, default=1.0)
    parser.add_argument("--specz_Sigma_fog_bf", help="The boost factor for the spec-z Sigma_fog.", type=float, default=1.0)
    parser.add_argument("--specz_bg_bf", help="The boost factor for the spec-z bg.", type=float, default=1.0)


    args = parser.parse_args()
    survey_area = args.survey_area
    kmax = args.kmax
    k_width = args.k_width    # k bin width
    zmin = args.zmin
    zmax = args.zmax
    nzbins = args.nzbins
    const_low = args.const_low
    const_up = args.const_up
    f0eff = args.f0eff
    Pspecz_sys = args.Pspecz_sys
    input_dir = args.input_dir
    photoz_Sigma_perp_bf = args.photoz_Sigma_perp_bf
    photoz_Sigma_para_bf = args.photoz_Sigma_para_bf
    photoz_Sigma_fog_bf = args.photoz_Sigma_fog_bf
    photoz_bg_bf = args.photoz_bg_bf

    specz_Sigma_perp_bf = args.specz_Sigma_perp_bf
    specz_Sigma_para_bf = args.specz_Sigma_para_bf
    specz_Sigma_fog_bf = args.specz_Sigma_fog_bf
    specz_bg_bf = args.specz_bg_bf
    
    odir = args.output_dir
    if not os.path.exists(odir):
        os.makedirs(odir)
    
    logging.basicConfig(format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    parameter_file = Path(input_dir, "input_params.yaml")
    yaml = YAML() 
    with open(parameter_file, 'r') as fr:
        input_params = yaml.load(fr)

    # input (theoretical) linear power spectrum
    kwig, Plin_wig = np.loadtxt(input_params['Pwig_linear'], dtype='f8', comments='#', unpack=True) 
    splPlin = interpolate.InterpolatedUnivariateSpline(kwig, Plin_wig)

    ksm, Psm = np.loadtxt(input_params['Pnow_linear'], dtype='f8', comments='#', unpack=True)
    splPsm = interpolate.InterpolatedUnivariateSpline(ksm, Psm)

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

    # set up the (fiducially) observed mu array, using Nmubins=100 or 200 has no influence 
    mu_o = np.linspace(0.0, 1.0, 101)
    ##mu_o = np.linspace(0.0, 1.0, 201)
    
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

        ## unit: Mpc/h
        r_low = speed_c * integrate.quad(comoving_dis_fun, 0., z_low, args=(Omega_m))[0]
        r_up  = speed_c * integrate.quad(comoving_dis_fun, 0., z_up,  args=(Omega_m))[0]
        print(r_low, r_up)

        V_total = 4./3. * np.pi *(r_up**3.0 - r_low**3.0)
        V_survey = V_total * fsky
        print("V_survey:", V_survey)
        # we consider both the input sigma_specz and sigma_photoz as sigma_0
        Sigma_specz_error = speed_c * sigma_specz * (1.0 + z_mid) * comoving_dis_fun(z_mid, Omega_m)
        Sigma_photoz_error = speed_c * sigma_photoz * (1.0 + z_mid) * comoving_dis_fun(z_mid, Omega_m)
        print("Sigma_specz_error, Sigma_photoz_error:", Sigma_specz_error, Sigma_photoz_error)

        # ## estimate the shot noise of photo-z and spec-z data
        n_specz = Ngal_specz/(fsky_zcosmos * V_total) * f0eff /(1.0 + z_mid)
        
        n_photoz = Ngal_photoz/(fsky_cosmos * V_total)  # unit: (Mpc/h)^-3 

        kmin = 2*np.pi/V_survey**(1./3)   # fundermental k 
        print("kmin:", kmin)
    
        ## assume kmin is limitted by survey volume
        #k_o = np.arange(kmin, kmax, 0.005)
        nkbins = int((kmax-kmin)/k_width)
        k_o = np.linspace(kmin, kmax, nkbins+1)
        print("k_o:", k_o.shape)
        
        sn_photoz = 1./n_photoz
        sn_specz = 1./n_specz
        print("sn_photoz:", sn_photoz)
        print("sn_specz:", sn_specz)
        shotnoise_specz_list.append(sn_specz)
        shotnoise_photoz_list.append(sn_photoz) 
        
        # calculate Plin
        
        theta = np.array([1.0, 1.0, f_growthrate, bias, bias])
        theta_names=['alpha_perp', 'alpha_para', 'f_growthrate', 'bias_g1', 'bias_g2']
        pkmu_linear = pkmu_linear_model(theta, splPlin, splPsm, k_o, mu_o, norm_gf)
        ofile = Path(odir, "pkmu_linear_csst_z{0:.2f}_{1:.2f}_sigmaspecz{2:.3f}_sigmaphotoz{3:.3f}_kmax{4:.2f}_default_params.npz".format(z_low, z_up, sigma_specz, sigma_photoz, kmax))
        np.savez(ofile, kcenter=k_o, kmin=kmin, mu=mu_o, parameters=theta, param_names=theta_names, pkmu_linear=pkmu_linear, V_survey=V_survey, Pspecz_sys=Pspecz_sys, sn_specz=sn_specz, sn_photoz=sn_photoz, bias_specz=bias)
        
        ## load Sigma_nl distribution
        ifile = Path(input_dir, "parameter_preparation/output/Sigma_nl_linearorder_z%.2f.out"%z_mid)
        data = np.loadtxt(ifile)
        Sigma_para, Sigma_perp = data[0, :]  # for pre-reconstruction
        print("Sigma_para, Sigma_perp before recon:", Sigma_para, Sigma_perp)
        
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
            r_val = 1.0
            
        logger.info(f"r_val from BAO reconstruction: {r_val}")

        # distinguish parameters from spec-z and photo-z a bit to avoid the degenerency between the parameters of spec-z and photo-z
        Sigma_perp_specz, Sigma_perp_photoz = Sigma_perp * r_val * specz_Sigma_perp_bf, Sigma_perp * photoz_Sigma_perp_bf
        Sigma_para_specz, Sigma_para_photoz = Sigma_para * r_val * specz_Sigma_para_bf, Sigma_para * photoz_Sigma_para_bf
        Sigma_fog_specz, Sigma_fog_photoz = Sigma_fog/(1+z_mid) * specz_Sigma_fog_bf, Sigma_fog/(1+z_mid) * photoz_Sigma_fog_bf
        bg_specz, bg_photoz = bias * specz_bg_bf, bias * photoz_bg_bf
        
        # In the real analysis, we distinguish some (nuisance) parameters for multi-tracers. 
        theta = np.array([1.0, 1.0, f_growthrate, Sigma_perp_specz, Sigma_perp_photoz, Sigma_para_specz, Sigma_para_photoz, Sigma_fog_specz, Sigma_fog_photoz, Sigma_specz_error, Sigma_photoz_error, bg_specz, bg_photoz, Pspecz_sys])
        logger.info(f"thea: {theta}")

        theta_names = ['alpha_perp', 'alpha_para', 'f_growthrate', 'Sigma_perp_specz', 'Sigma_perp_photoz', 'Sigma_para_specz', 'Sigma_para_photoz', 'Sigma_fog_specz', 'Sigma_fog_photoz', 'Sigma_specz_error', 'Sigma_photoz_error', 'bg_specz', 'bg_photoz', "Pspecz_sys"]
                    
        
        BAOonly = True
        pkmu_specz, pkmu_photoz, pkmu_cross = Pkmu2d(theta, k_o, mu_o, splPlin, splPsm, norm_gf, BAOonly)
        print("pkmu_specz shape:", pkmu_specz.shape)
        
        ofile = Path(odir, "pkmu_BAOonly_csst_z{0:.2f}_{1:.2f}_sigmaspecz{2:.3f}_sigmaphotoz{3:.3f}_kmax{4:.2f}_default_params.npz".format(z_low, z_up, sigma_specz, sigma_photoz, kmax))
        np.savez(ofile, kcenter=k_o, kmin=kmin, mu=mu_o, parameters=theta, param_names=theta_names, pkmu_specz=pkmu_specz, pkmu_photoz=pkmu_photoz, pkmu_cross=pkmu_cross, V_survey=V_survey, Pspecz_sys=Pspecz_sys, sn_specz=sn_specz, sn_photoz=sn_photoz, bias_specz=bias)
        
        BAOonly = False
        pkmu_specz, pkmu_photoz, pkmu_cross = Pkmu2d(theta, k_o, mu_o, splPlin, splPsm, norm_gf, BAOonly)
        ofile = Path(odir, "pkmu_nonlinear_csst_z{0:.2f}_{1:.2f}_sigmaspecz{2:.3f}_sigmaphotoz{3:.3f}_kmax{4:.2f}_default_params.npz".format(z_low, z_up, sigma_specz, sigma_photoz, kmax))
        np.savez(ofile, kcenter=k_o, kmin=kmin, mu=mu_o, parameters=theta, param_names=theta_names, pkmu_specz=pkmu_specz, pkmu_photoz=pkmu_photoz, pkmu_cross=pkmu_cross, V_survey=V_survey, Pspecz_sys=Pspecz_sys, sn_specz=sn_specz, sn_photoz=sn_photoz, bias_specz=bias)
        
    
        num_all_params = len(theta)
        for i in range(num_all_params):
            for const in [const_low, const_up]:
                temp = theta[i]
                # slightly increase or decrease the parameter value by 1%
                param_new = const * theta[i]
                theta[i] = param_new

                BAOonly = True
                pkmu_specz, pkmu_photoz, pkmu_cross = Pkmu2d(theta, k_o, mu_o, splPlin, splPsm, norm_gf, BAOonly)
                ofile = Path(odir, "pkmu_BAOonly_csst_z{0:.2f}_{1:.2f}_sigmaspecz{2:.3f}_sigmaphotoz{3:.3f}_kmax{4:.2f}_param{5}X{6:.3f}.npz".format(z_low, z_up, sigma_specz, sigma_photoz, kmax, i, const))
                np.savez(ofile, kcenter=k_o, kmin=kmin, mu=mu_o, parameters=theta, param_names=theta_names, pkmu_specz=pkmu_specz, pkmu_photoz=pkmu_photoz, pkmu_cross=pkmu_cross, Pspecz_sys=Pspecz_sys, V_survey=V_survey, sn_photoz=sn_photoz, sn_specz=sn_specz)
                
                BAOonly = False
                pkmu_specz, pkmu_photoz, pkmu_cross = Pkmu2d(theta, k_o, mu_o, splPlin, splPsm, norm_gf, BAOonly)
                ofile = Path(odir, "pkmu_nonlinear_csst_z{0:.2f}_{1:.2f}_sigmaspecz{2:.3f}_sigmaphotoz{3:.3f}_kmax{4:.2f}_param{5}X{6:.3f}.npz".format(z_low, z_up, sigma_specz, sigma_photoz, kmax, i, const))
                np.savez(ofile, kcenter=k_o, kmin=kmin, mu=mu_o, parameters=theta, param_names=theta_names, pkmu_specz=pkmu_specz, pkmu_photoz=pkmu_photoz, pkmu_cross=pkmu_cross, Pspecz_sys=Pspecz_sys, V_survey=V_survey, sn_photoz=sn_photoz, sn_specz=sn_specz)

                # change the parameter vlaue back to the default
                theta[i] = temp


if __name__ == '__main__':
    main()




