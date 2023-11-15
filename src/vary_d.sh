declare -A Alpha=(["Cora"]="0.9" ["CiteSeer"]="0.6" ["MovieLens"]="0.6" ["Google"]="0.9" ["Amazon"]="0.5")
declare -A Gamma=(["Cora"]="10" ["CiteSeer"]="6" ["MovieLens"]="6" ["Google"]="10" ["Amazon"]="0")

for d in {16,32,54,128,256}
do
	python3 proposed.py --data $1 --alpha ${Alpha[$1]} --gamma ${Gamma[$1]} --d $d  >> ../log/vary_d_$1.log
	python3 eval.py --data $1 --algo TPC  >> ../log/vary_d_$1.log
done
