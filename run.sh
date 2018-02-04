#!/bin/bash

BIN=./bin
OUT=./out
DXMAX=2048
VERBOSE=1

# stencil size
stsz=( 7p 13p 19p 25p 31p 37p )

# optimization max
OPT_MX=11

# optmizations
declare -A opts
opts[0]="Base"
opts[1]="Sham"
opts[2]="ZintReg"
opts[3]="Zint"
opts[4]="ShamZintReg"
opts[5]="ShamZint"
opts[6]="ShamZintTempReg"
opts[7]="Roc"
opts[8]="ShamRoc"
opts[9]="RocZintReg"
opts[10]="RocZint"
opts[11]="ShamRocZintTempReg"

for st in "${stsz[@]}"
do
	for ((opt=0; opt<=$OPT_MX; opt++))
	do
		FILE=$OUT/${st}-"${opts[$opt]}".out
		echo "dx,dy,dz,gFlops,gpu_comp" > $FILE

		for ((dx=32; dx<=$DXMAX; dx+=32))
		do
			#echo Executing $st test, "${opts[$opt]}" optimization, $dx dx

			$BIN/$st.exe 0 $dx 256 256 5 $VERBOSE >> $FILE
		done
	done
done
