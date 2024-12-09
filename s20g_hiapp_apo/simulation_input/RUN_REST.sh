source /dartfs-hpc/rc/home/0/f006f50/labhome/SINGULARITY_BUILDS/.gmx_container.bash
# 20 replicas
nrep=20
# "effective" temperature range
tmin=300
tmax=500

# build geometric progression
list=$(
awk -v n=$nrep \
    -v tmin=$tmin \
    -v tmax=$tmax \
  'BEGIN{for(i=0;i<n;i++){
    t=tmin*exp(i*log(tmax/tmin)/(n-1));
    printf(t); if(i<n-1)printf(",");
  }
}'
)

# clean directory
rm -fr \#*
rm -fr topol*

for((i=1;i<=nrep;i++))
do
rm -rf $i
mkdir $i  
cp plumed.dat $i/plumed.dat
cp ref.pdb $i/ref.pdb

# choose lambda as T[0]/T[i]
# remember that high temperature is equivalent to low lambda
  #echo $list
lambda=$(echo $list | awk 'BEGIN{FS=",";}{print $1/$'$((i))';}') 
temp=$(echo $list | awk 'BEGIN{FS=",";}{print $(('$i'));}')
 
echo "$i $lambda $temp"
plumed partial_tempering $lambda < REST.top > $i/topol.top

# prepare tpr file
# -maxwarn is often needed because box could be charged

gmx_s grompp -maxwarn 2 -c structure/${i}.new.gro -o $i/production.tpr -f production.mdp -p $i/topol.top
#gmx_mpi mdrun -deffnm $i/eq -nsteps 10000 -c $i/eq.gro


done

#krenew -b -a -K60
#nohup mpirun -np 16 mdrun_mpi -deffnm production -multidir 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 -plumed plumed.dat -replex 800 -hrex -dlb no >> AR_R2_R3_REST_16reps_617_turing2AR R

#R2_R3 - turing02 - 6.18.2020

