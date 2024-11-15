#!/bin/bash

# Created by Korey Reid
#
# Equilibrium simulation for use on 20x Ampere GPUS
# Equilibrating 20 systems for REMD 300K to 500K
# MIN -> NVT -> NPT 
# <100   100ps?   100ps?  How is the relaxation? 
#
#
# Purge any external loaded moduled and load based on workflow???? 
# Currently Loaded Modulefiles:
#  1) cuda/11.2           3) plumed/2.7.2            5) gsl/gsl-2.6            7) plumed/2.6.0
#  2) openmpi/4.1.3-gnu   4) GROMACS/2020.6-plumed   6) mpich/3.3.0-gnu2.8.5   8) plumed/2.8.0-crystal
#
#
echo `pwd`
flags=("$@")
if [[ ${flags[0]} == '-h' ]]
then
	echo "flag   action"
	echo "-h Print This help menu"
	echo "-em prep system for energy minimization"
	echo "    use mpirun -np N gmx_mpi mdrun -v -ntomp 4 -deffnm NAME.em -multidir {1..N} -dlb no"
	echo "-eqnvt prep system for equilibration under nvt"
	echo "    use mpirun -np N gmx_mpi mdrun -v -ntomp 4 -deffnm NAME.eqnvt -multidir {1..N} -dlb no"
	echo "-eqnpt prep system for equilibration under npt"
	echo "    use mpirun -np N gmx_mpi mdrun -v -ntomp 4 -deffnm NAME.eqnpt -multidir {1..N} -dlb no"
	echo "-rest Box_L populate structure folder for rest simulation with cubic box"
fi

#module load GROMACS/2019-plumed
#module load vmd 
#module load cuda/11.2
#module unload plumed/2.8.0 cuda/11.6
#module load plumed/2.6.0 cuda/11.2
#module unload GROMACS/2019-plumed
#module load GROMACS/2020.6-plumed

# module list
if [[ ${flags[0]} == '-em' ]]
then
if [[ -d structure ]]
then
rm -r structure
fi
mkdir structure

for((i=1;i<=20;i++))
do

rm -r $i
mkdir $i

cp topol_start.top $i/topol.top
cp input_gro/prot_vac${i}.gro $i/prot.gro

cd $i

echo 1

#
# If packmol was used for input, skip creation of box (assumed already to be in a .gro file 
# with box lengths. Then perform solvation and ion addition (requires ions.mdp)
# 
#
if [[ ! ${flags[1]} == '-packmol' ]]
then
	gmx_mpi editconf -f prot.gro -o prot.box.gro -c -box 6.5 6.5 6.5 -bt cubic
	gmx_mpi solvate -cp prot.box.gro -cs ../a99SBdisp.ff/tip4pd.gro -maxsol 8670 -o prot.solv.gro -p topol.top
	gmx_mpi grompp -f ../mdp/ions.mdp -c prot.solv.gro -p topol.top -o prot.ions.tpr -maxwarn 2

# echo 13 for apo, and apo 15 for ligand

echo 13 | gmx_mpi genion -s prot.ions.tpr -o prot.ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.025

echo -e "1|13\nq\n" | gmx_mpi make_ndx -f prot.ions.gro -o index.ndx

else
	cp prot.gro prot.ions.gro
fi

gmx_mpi grompp -f ../mdp/minim.mdp -c prot.ions.gro -p topol.top -o prot.em.tpr -maxwarn 2

rm \#*
cd ../
done
# echo `pwd`

mpirun -np 20 gmx_mpi mdrun -v -ntomp 4 -deffnm prot.em -multidir {1..20} -dlb no 

fi

if [[ ${flags[0]} == '-eqnvt' ]]
then

for((i=1;i<=20;i++))
do 

   cd $i/
   gmx_mpi grompp -f ../mdp/nvt.mdp -c prot.em.gro -p topol.top -o prot.eqnvt.tpr 
   cd ../

done

mpirun -np 20 gmx_mpi mdrun -v -ntomp 4 -deffnm prot.eqnvt -multidir {1..20} -dlb no 

fi

if [[ ${flags[0]} == '-eqnpt' ]]
then

for((i=1;i<=20;i++))
do
	cd $i/
	gmx_mpi grompp -f ../mdp/npt.mdp -c prot.eqnvt.gro -p topol.top -o prot.eqnpt.tpr 
	cd ../
done

mpirun -np 20 gmx_mpi mdrun -v -ntomp 4 -deffnm prot.eqnpt -multidir {1..20} -dlb no 

fi

if [[ ${flags[0]} == '-eqnpt2' ]]
then

for((i=1;i<=20;i++))
do
	cd $i/
	gmx_mpi grompp -f ../mdp/npt2.mdp -c prot.eqnpt.gro -p topol.top -o prot.eqnpt2.tpr 
	cd ../
done
# only run this last one in the background 
nohup mpirun -np 20 gmx_mpi mdrun -v -deffnm prot.eqnpt2 -multidir {1..20} -dlb no >> eqnpt2_amp02 & 

fi

if [[ ${flags[0]} == '-rest' ]]
then
for((i=1;i<=20;i++))
do
	cd $i/
	gmx_mpi editconf -f prot.eqnpt.gro -o ${i}.new.gro -box ${flags[1]} ${flags[1]} ${flags[1]}
        cd ../
        cp ${i}/${i}.new.gro structure/
done
fi


