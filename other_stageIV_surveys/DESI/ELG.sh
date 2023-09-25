#!/bin/bash
#PBS -l nodes=1:ppn=1
##PBS -q debug
#PBS -q small
#PBS -N elg
#PBS -l walltime=24:00:00
#PBS -o ./stdout/${PBS_JOBNAME}.o${PBS_JOBID}
#PBS -e ./stdout/${PBS_JOBNAME}.e${PBS_JOBID}
#PBS -M zhejied@sjtu.edu.cn

## if run it in bash directly, comment out the following three lines
cd $PBS_O_WORKDIR
echo ncores=${PBS_NP}
echo `pwd` 

module load anaconda/anaconda3

source activate miniworkshop

## For DESI

tracer=ELG_LOP
survey_area=14000
# rms of random redshift error and systematics error, Table 5 of arXiv:2306.06307
sigma_specz=0.000026   


const_low=0.99
const_up=1.01

##f0eff=0.5
Pspecz_sys=1.0


zmin=1.1
zmax=1.6
nzbins=5

# the parameter boost factor for the photo-z 
param_bf=1.0   
echo "param_bf: ${param_bf}"

specz_params_list=("specz_Sigma_perp" "specz_Sigma_para" "specz_Sigma_fog" "specz_bg")
photoz_params_list=("photoz_Sigma_perp" "photoz_Sigma_para" "photoz_Sigma_fog" "photoz_bg")

input_nzfile="./input/nz_bias_f_Sigmanl_DESI_${tracer}_Adame_2023a.txt"
params_dir="./"
pk_dir0="./"
sigma_dir0="./"

for k_width in 0.005; do
  for kmax in 0.3 0.5; do
      ##for Sigma_fog in 7.0; do
      for Sigma_fog in 0.1; do
        echo "Sigma_fog: ${Sigma_fog}"
        for sigma_photoz in 0.025; do
           
          echo ${ngalspecz_file}
          echo ${ngalphotoz_file}
          for with_recon in False True; do
            if [ ${with_recon} = "True" ]; then
                dir1="post_recon"
            else
                dir1="pre_recon"
            fi
            echo "Result based on $dir1"
            
            ## ---- for param_bf=1.0
            pkmu_dir="${pk_dir0}/output/pkmu/vary_params_${const_low}_${const_up}/Sigma_fog_${Sigma_fog}/${dir1}/k_width${k_width}/Pspecz_sys${Pspecz_sys}/params_bf1.0/${tracer}/${zmin}z${zmax}_${nzbins}bins/"
            sigalpha_dir="${sigma_dir0}/output/sigma_alpha/vary_params_${const_low}_${const_up}/Sigma_fog_${Sigma_fog}/$dir1/k_width${k_width}/Pspecz_sys${Pspecz_sys}/params_bf1.0/${tracer}/${zmin}z${zmax}_${nzbins}bins/"

            ## ---- for param_bf=1.0
            time python ./pkmu_template.py --survey_area ${survey_area} --kmax $kmax --k_width ${k_width} --zmin $zmin --zmax $zmax --input_nzfile ${input_nzfile} --sigma_specz ${sigma_specz} --sigma_photoz ${sigma_photoz} --Sigma_fog ${Sigma_fog} --const_low ${const_low} --const_up ${const_up} --with_recon ${with_recon} --Pspecz_sys ${Pspecz_sys} --input_dir ${params_dir} --output_dir ${pkmu_dir}
           
            time python ./fisher_singletracer.py --survey_area ${survey_area} --kmax $kmax --zmin $zmin --zmax $zmax --nzbins $nzbins --input_nzfile ${input_nzfile} --sigma_specz ${sigma_specz} --sigma_photoz ${sigma_photoz} --Sigma_fog ${Sigma_fog} --const_low ${const_low} --const_up ${const_up} --tracer specz  --input_dir ${pkmu_dir} --output_dir ${sigalpha_dir}


        done
    done
  done
done
done

conda deactivate
