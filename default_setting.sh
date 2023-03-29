#!/bin/bash
#PBS -l nodes=1:ppn=1
##PBS -q debug
#PBS -q small
#PBS -N default_set
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


survey_area=17500
sigma_specz=0.002

kmax=0.3

const_low=0.99
const_up=1.01

f0eff=0.5
Pspecz_sys=1.0


zmin=0.0
zmax=1.6
nzbins=8

# the parameter boost factor for the photo-z 
param_bf=1.0   
echo "param_bf: ${param_bf}"

specz_params_list=("specz_Sigma_perp" "specz_Sigma_para" "specz_Sigma_fog" "specz_bg")
photoz_params_list=("photoz_Sigma_perp" "photoz_Sigma_para" "photoz_Sigma_fog" "photoz_bg")


params_dir="/home/zjding/csst_bao/fisher_pkmu/"
pk_dir0="/home/zjding/csst_bao/fisher_pkmu/BAO_part/Fisher_sigma_alphas/numerical_method/default/"
sigma_dir0="/home/zjding/csst_bao/fisher_pkmu/BAO_part/Fisher_sigma_alphas/numerical_method/default/"

#for Pspecz_sys in 1.e3 2.e3 3.e3 4.e3 5.e3 6.e3 7.e3 8.e3 9.e3 1.e4; do
#for param in ${photoz_params_list[*]}; do
#for param_bf in 1.05 1.1 1.2 1.5 1.7 2.0; do
#for param in ${specz_params_list[*]}; do
for k_width in 0.005; do
  for kmax in 0.3 0.4 0.5; do
#  for kmax in 0.3; do
      for Sigma_fog in 7.0; do
        echo "Sigma_fog: ${Sigma_fog}"
        for sigma_photoz in 0.025 0.05; do
#        for sigma_photoz in 0.025; do
           
          ngalspecz_file=$(printf "/home/zjding/csst_bao/fisher_pkmu/input/nz_bins/nz_specz_zmin%.1f_zmax%.1f_%dzbins.out" $zmin $zmax $nzbins)
          ngalphotoz_file=$(printf "/home/zjding/csst_bao/fisher_pkmu/input/Caoye_data/nz_bins/nz_photoz_zmin%.1f_zmax%.1f_%dzbins_sigmaz%.3f.out" $zmin $zmax $nzbins $sigma_photoz)  

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
            pkmu_dir="${pk_dir0}/output/pkmu/vary_params_${const_low}_${const_up}/Sigma_fog_${Sigma_fog}/${dir1}/k_width${k_width}/Pspecz_sys${Pspecz_sys}/params_bf1.0/"
            sigalpha_dir="${sigma_dir0}/output/sigma_alpha/vary_params_${const_low}_${const_up}/Sigma_fog_${Sigma_fog}/$dir1/k_width${k_width}/Pspecz_sys${Pspecz_sys}/params_bf1.0/"

            ## ---- for param_bf=1.0
            time python ./pkmu_template.py --survey_area ${survey_area} --kmax $kmax --k_width ${k_width} --zmin $zmin --zmax $zmax --nzbins $nzbins --ngalspecz_file ${ngalspecz_file} --ngalphotoz_file ${ngalphotoz_file} --sigma_specz ${sigma_specz} --sigma_photoz ${sigma_photoz} --Sigma_fog ${Sigma_fog} --const_low ${const_low} --const_up ${const_up} --with_recon ${with_recon} --f0eff $f0eff --Pspecz_sys ${Pspecz_sys} --input_dir ${params_dir} --output_dir ${pkmu_dir}
           
            time python ./fisher_singletracer.py --survey_area ${survey_area} --kmax $kmax --zmin $zmin --zmax $zmax --nzbins $nzbins --sigma_specz ${sigma_specz} --sigma_photoz ${sigma_photoz} --Sigma_fog ${Sigma_fog} --const_low ${const_low} --const_up ${const_up} --tracer specz  --input_dir ${pkmu_dir} --output_dir ${sigalpha_dir}

            time python ./fisher_singletracer.py --survey_area ${survey_area} --kmax $kmax --zmin $zmin --zmax $zmax --nzbins $nzbins --sigma_specz ${sigma_specz} --sigma_photoz ${sigma_photoz} --Sigma_fog ${Sigma_fog} --const_low ${const_low} --const_up ${const_up} --tracer photoz --input_dir ${pkmu_dir} --output_dir ${sigalpha_dir}


            time python ./fisher_specz_add_cross.py --survey_area ${survey_area} --kmax $kmax --zmin $zmin --zmax $zmax --nzbins $nzbins --sigma_specz ${sigma_specz} --sigma_photoz ${sigma_photoz} --Sigma_fog ${Sigma_fog} --const_low ${const_low} --const_up ${const_up}  --input_dir ${pkmu_dir} --output_dir ${sigalpha_dir}

            time python ./fisher_specz_photoz_add_cross.py --survey_area ${survey_area} --kmax $kmax --zmin $zmin --zmax $zmax --nzbins $nzbins --sigma_specz ${sigma_specz} --sigma_photoz ${sigma_photoz} --Sigma_fog ${Sigma_fog} --const_low ${const_low} --const_up ${const_up}  --input_dir ${pkmu_dir} --output_dir ${sigalpha_dir}

        done
    done
  done
done
done
#done
#done
#done

conda deactivate
