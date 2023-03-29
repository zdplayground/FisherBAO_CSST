#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -q debug
#PBS -N vary_photoz_Sigmaperp
#PBS -l walltime=3:00:00
#PBS -o ./stdout/${PBS_JOBNAME}.o${PBS_JOBID}
#PBS -e ./stdout/${PBS_JOBNAME}.e${PBS_JOBID}
#PBS -M zhejied@sjtu.edu.cn

## if run it in bash directly, comment out the following three lines
cd $PBS_O_WORKDIR
echo ncores=${PBS_NP}
echo `pwd` 

module load anaconda/anaconda3

source activate miniworkshop

survey_area=17500

kmax=0.3
k_width=0.005

sigma_specz=0.002

#with_recon="True"       # True or False
with_recon="False"      

const_low=0.99
const_up=1.01

#const_low=0.995
#const_up=1.005

f0eff=0.5
#Pspecz_sys=0.0
Pspecz_sys=5.e3

zmin=0.0
zmax=1.6
nzbins=8


params_dir="/home/zjding/csst_bao/fisher_pkmu/"
dir0="/home/zjding/csst_bao/fisher_pkmu/BAO_part/Fisher_sigma_alphas/semi_analytic/"

if [ ${with_recon} = "True" ]; then
    recon_dir="post_recon"
else
    recon_dir="pre_recon"
fi

echo "Result based on $recon_dir"

#for Pspecz_sys in 1.e1 1.e2 5.e2 1.e3; do
#for photoz_Sigma_perp_bf in 1.0 1.05 1.1 1.3 1.5 2.0; do
for photoz_Sigma_perp_bf in 1.0; do
for Sigma_fog in 7.0; do
    echo "Sigma_fog: ${Sigma_fog}"

    output_lnP_dir="$dir0/output/dlnP_dparam/kmax${kmax}/Pspecz_sys${Pspecz_sys}/${recon_dir}/photoz_Sigma_perp_bf${photoz_Sigma_perp_bf}/vary_dlnP_only/"
    output_fisher_dir="$dir0/output/Fisher_params/kmax${kmax}/Pspecz_sys${Pspecz_sys}/${recon_dir}/photoz_Sigma_perp_bf${photoz_Sigma_perp_bf}/vary_dlnP_only/"
#    output_lnP_dir="$dir0/output/dlnP_dparam/kmax${kmax}/vary_dlnk_${const_low}_${const_up}/Sigma_fog_${Sigma_fog}/${recon_dir}/photoz_Sigma_perp_bf${photoz_Sigma_perp_bf}/vary_dlnP_cov_both/"
#    output_fisher_dir="$dir0/output/Fisher_params/kmax${kmax}/vary_dlnk_${const_low}_${const_up}/Sigma_fog_${Sigma_fog}/${recon_dir}/photoz_Sigma_perp_bf${photoz_Sigma_perp_bf}/vary_dlnP_cov_both/"
    echo "output_lnP_dir=${output_lnP_dir}"
    echo "output_fisher_dir=${output_fisher_dir}"

    for sigma_photoz in 0.025; do
        ngalspecz_file=$(printf "/home/zjding/csst_bao/fisher_pkmu/input/nz_bins/nz_specz_zmin%.1f_zmax%.1f_%dzbins.out" $zmin $zmax $nzbins)
        ngalphotoz_file=$(printf "/home/zjding/csst_bao/fisher_pkmu/input/Caoye_data/nz_bins/nz_photoz_zmin%.1f_zmax%.1f_%dzbins_sigmaz%.3f.out" $zmin $zmax $nzbins $sigma_photoz)  

        echo ${ngalspecz_file}
        echo ${ngalphotoz_file}


        time python ./dlnP_dparam.py --survey_area ${survey_area} --zmin $zmin --zmax $zmax --nzbins $nzbins --ngalspecz_file ${ngalspecz_file} --ngalphotoz_file ${ngalphotoz_file} --sigma_specz ${sigma_specz} --sigma_photoz ${sigma_photoz} --Sigma_fog ${Sigma_fog} --const_low ${const_low} --const_up ${const_up} --with_recon ${with_recon} --f0eff $f0eff --Pspecz_sys ${Pspecz_sys} --input_dir ${params_dir} --output_dir ${output_lnP_dir} --photoz_Sigma_perp_bf ${photoz_Sigma_perp_bf}
        
        time python ./cal_inv_cov_lnP.py --survey_area ${survey_area} --kmax $kmax --k_width ${k_width} --zmin $zmin --zmax $zmax --nzbins $nzbins --ngalspecz_file ${ngalspecz_file} --ngalphotoz_file ${ngalphotoz_file} --sigma_specz ${sigma_specz} --sigma_photoz ${sigma_photoz} --Sigma_fog ${Sigma_fog} --with_recon ${with_recon} --f0eff $f0eff --Pspecz_sys ${Pspecz_sys} --input_dir ${params_dir} --output_dir ${output_lnP_dir} --photoz_Sigma_perp_bf 1.0
        
        time python ./fisher_lnPspecz_cross_photoz.py --kmax $kmax --zmin $zmin --zmax $zmax --nzbins $nzbins --Sigma_fog ${Sigma_fog} --input_dir ${output_lnP_dir} --output_dir ${output_fisher_dir}

    done
done
done
#done

conda deactivate
