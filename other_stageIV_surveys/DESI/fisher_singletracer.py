## Copy it from /home/zjding/csst_bao/fisher_pkmu/BAO_part/Fisher_sigma_alphas/numerical_method/secondgen, modify it for Euclid prediction. --08-18-2023
#
import os, sys
import numpy as np
import logging
from scipy import linalg, integrate
sys.path.append("/home/zjding/csst_bao/fisher_pkmu/BAO_part/Fisher_sigma_alphas/numerical_method/default/")
from pkmu_module import read_pkmu, dp_dlambda, select_params_id
from argparse import ArgumentParser


## pseudo Veff
def cal_Veff(Vsurvey, Pkmu, shotnoise):
    #return (P1/(P2+shotnoise))**2.0 * Vsurvey 
    # P1 cancels out the P in the demoninator of dlnP/dlambda
    return (1.0/(Pkmu + shotnoise))**2.0 * Vsurvey 

## since the mu range in our $P(k,\mu)$ is in [0, 1], we do not need to consider 1/2 prefactor due to the same mode from k and -k.  
# def cal_Nmodes_kmu(k_low, k_up, mu_low, mu_up, Veff):
#     return (k_up**3.0 - k_low**3.0)*(mu_up-mu_low)/(12*np.pi**2.0) * Veff

## we use a trapezoidal rule to calculate Fij
def cal_Fmn(kobs, mu_obs, Veff, dp_dm, dp_dn):
    Fmn_k = np.zeros(len(kobs))
    
    for i in range(len(kobs)):
        temp = dp_dm[i, :] * dp_dn[i, :] * Veff[i, :] * (kobs[i]/(2*np.pi))**2.0 
        Fmn_k[i] = integrate.trapz(temp, mu_obs)
    Fmn = integrate.trapz(Fmn_k, kobs)

    return Fmn
        
            
def main():
    parser = ArgumentParser(description="Fisher forecast of BAO from the spec-z survey.")
    parser.add_argument("--survey_area", type=float, help="The sky area (deg^2)", required=True)
    parser.add_argument("--kmax", type=float, help="The maximum k of P(k,mu).", required=True)
    parser.add_argument("--zmin", type=float, help="z minimum.", default=0.0)
    parser.add_argument("--zmax", type=float, help="z maximum.", default=1.6)
    parser.add_argument("--nzbins", type=int, help="The number of z bins.", required=True)
    parser.add_argument("--input_nzfile", help="Input file including nz, bias, f, ect", required=True)
    parser.add_argument("--sigma_specz", type=float, help="spec-z error sigma_0.", required=True)
    parser.add_argument("--sigma_photoz", type=float, help="photo-z error, sigma_z.", required=True)
    parser.add_argument("--Sigma_fog", type=float, help="Finger-of-God damping term for both spec-z and photo-z data.", default=0.0, required=True)
    parser.add_argument("--const_low", type=float, help="The lower constant multiplied on parameters.", required=True)
    parser.add_argument("--const_up", type=float, help="The upper constant multiplied on parameters.", required=True)
    parser.add_argument("--tracer", type=str, help="The tracer type, specz or photoz.", required=True)
    parser.add_argument("--input_dir", help="input directory.", required=True)
    parser.add_argument("--output_dir", help="output directory.", required=True)
    
    args = parser.parse_args()
    
    survey_area = args.survey_area
    kmax = args.kmax
    #kmax = 0.5
    zmin = args.zmin
    zmax = args.zmax
    nzbins = args.nzbins
    input_nzfile = args.input_nzfile
    ##sigma_specz = 0.000
    sigma_specz = args.sigma_specz
    sigma_photoz = args.sigma_photoz

    Sigma_fog = args.Sigma_fog

    const_low = args.const_low
    const_up = args.const_up
    tracer = args.tracer

    zbins = np.linspace(zmin, zmax, nzbins+1)
    # zlow_bins = zbins[0:-1]
    # zup_bins = zbins[1:]
    zlow_bins, zup_bins = np.loadtxt(input_nzfile, usecols=[0, 1], unpack=True)

    path = args.input_dir
    print(path)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ##theta_names = ['alpha_perp', 'alpha_para', 'f_growthrate', 'Sigma_perp_specz', 'Sigma_perp_photoz', 'Sigma_para_specz', 'Sigma_para_photoz', 'Sigma_fog_specz', 'Sigma_fog_photoz', 'Sigma_specz_error', 'Sigma_photoz_error', 'bg_specz', 'bg_photoz', 'Pspecz_sys']
    if Sigma_fog < 1.e-7:
        if tracer == "specz":
            skip_params_list = ['Sigma_perp_photoz', 'Sigma_para_photoz', 'Sigma_fog_specz', 'Sigma_fog_photoz', 'Sigma_photoz_error', 'bg_photoz']
        elif tracer == "photoz":
            skip_params_list = ['Sigma_perp_specz', 'Sigma_para_specz', 'Sigma_fog_specz', 'Sigma_fog_photoz', 'Sigma_specz_error', 'bg_specz', 'Pspecz_sys']
    else:
        if tracer == "specz":
            skip_params_list = ['Sigma_perp_photoz', 'Sigma_para_photoz', 'Sigma_fog_photoz', 'Sigma_photoz_error', 'bg_photoz']
        elif tracer == "photoz":
            skip_params_list = ['Sigma_perp_specz', 'Sigma_para_specz', 'Sigma_fog_specz', 'Sigma_specz_error', 'bg_specz', 'Pspecz_sys']
        
    
    Fisher_matrix_diffz = []
    output_alphas_mar = []       # marginalize other parameters
    output_alphas_unmar = []     # unmarginalize (or fix) other parameters
    for z_low, z_up in zip(zlow_bins, zup_bins):

        filename_nonlinear = "pkmu_nonlinear_csst_z{0:.2f}_{1:.2f}_sigmaspecz{2:.3f}_sigmaphotoz{3:.3f}_kmax{4:.2f}_param%dX%.3f.npz".format(z_low, z_up, sigma_specz, sigma_photoz, kmax)

        default_res = np.load(path+"pkmu_nonlinear_csst_z{0:.2f}_{1:.2f}_sigmaspecz{2:.3f}_sigmaphotoz{3:.3f}_kmax{4:.2f}_default_params.npz".format(z_low, z_up, sigma_specz, sigma_photoz, kmax))

        kcenter = default_res['kcenter']
        kmin = default_res['kmin']
        mu = default_res['mu']
        parameters = default_res['parameters']
        print("Parameters:", parameters)
        all_param_names = default_res['param_names']
        ##all_param_names = ['alpha_perp', 'alpha_para', 'f_growthrate', 'Sigma_perp_specz', 'Sigma_perp_photoz', 'Sigma_para_specz', 'Sigma_para_photoz', 'Sigma_fog_specz', 'Sigma_fog_photoz', 'Sigma_specz_error', 'Sigma_photoz_error', 'bg_specz', 'bg_photoz', 'Pspecz_sys']
        print("Parameter names:", all_param_names)
        pkmu_specz = default_res['pkmu_specz']        # includes Pspecz_sys
        pkmu_photoz = default_res['pkmu_photoz']
        pkmu_cross = default_res['pkmu_cross']
        Vsurvey = default_res['V_survey']

        sn_photoz = default_res['sn_photoz']
        sn_specz = default_res['sn_specz']
        Pspecz_sys = default_res['Pspecz_sys'] 

        print("kmin:", kmin)
        print(" {}<z{}:".format(z_low, z_up))
        print("sn_photoz, sn_specz:", sn_photoz, sn_specz)
        print("Pspecz_sys:", Pspecz_sys)

        # for single tracer case, not all parameters are related
        num_all_params = len(parameters)
         
        ##input_file = path + filename_BAOonly 
        input_file = path + filename_nonlinear
        
        # ## In the covariance we use the power spectrum with the BAO damping and redshift error
        if tracer == "specz":
            Veff = cal_Veff(Vsurvey, pkmu_specz, sn_specz)
        elif tracer == "photoz":
            Veff = cal_Veff(Vsurvey, pkmu_photoz, sn_photoz)
        
        param_id_list, param_name_list = select_params_id(all_param_names, skip_params_list) 
        
        num_params = len(param_id_list)    # number of selected parameters
        temp_matrix  = np.zeros((num_params, num_params))
        
        for i in range(num_params):

            dspecz_dpi, dphotoz_dpi, dcross_dpi = dp_dlambda(input_file, param_id_list[i], alpha_minus=const_low, alpha_plus=const_up)

            for j in range(i, num_params):

                dspecz_dpj, dphotoz_dpj, dcross_dpj = dp_dlambda(input_file, param_id_list[j], alpha_minus=const_low, alpha_plus=const_up)        
                if tracer == "specz":
                    temp_matrix[i, j] = cal_Fmn(kcenter, mu, Veff, dspecz_dpi, dspecz_dpj)
                elif tracer == "photoz":
                    temp_matrix[i, j] = cal_Fmn(kcenter, mu, Veff, dphotoz_dpi, dphotoz_dpj)

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

    # Indeed the statisitial error of alphas is influenced little by marginalization or not. 
    print("alpha with marginalization:", output_alphas_mar)
    print("alpha without marginalization:", output_alphas_unmar)

    alphas_mar = np.array(output_alphas_mar)
    alphas_unmar = np.array(output_alphas_unmar)

    params_free = np.zeros(num_all_params, dtype=int)
    params_free[param_id_list] = 1       
    params_str = ''.join(str(x) for x in params_free)
    
    ofile = output_dir + f"Fisher_matrix_diffz_{tracer}_tracer_zerror_specz{sigma_specz:.3f}_photoz{sigma_photoz:.3f}_kmax{kmax:.2f}_params{params_str}.npz"
    np.savez(ofile, Fisher_matrix_diffz=Fisher_matrix_diffz, zmid=(zlow_bins+zup_bins)/2.0)
        
    ofile = output_dir + f"sigma_alpha_{tracer}_tracer_zerror_specz{sigma_specz:.3f}_photoz{sigma_photoz:.3f}_kmax{kmax:.2f}_params{params_str}.npz"
    np.savez(ofile, alphas_mar = alphas_mar, alphas_unmar = alphas_unmar)


    
if __name__ == '__main__':
    main()




