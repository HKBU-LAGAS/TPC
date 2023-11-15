declare -A Alpha=(["Cora"]="0.9" ["CiteSeer"]="0.6" ["MovieLens"]="0.6" ["Google"]="0.9" ["Amazon"]="0.5")
declare -A Gamma=(["Cora"]="10" ["CiteSeer"]="6" ["MovieLens"]="6" ["Google"]="10" ["Amazon"]="0")
declare -A Dim=(["Cora"]="128" ["CiteSeer"]="32" ["MovieLens"]="-1" ["Google"]="32" ["Amazon"]="64")

for t in {0,5,10,20,40}
do
	python3 proposed.py --data $1 --alpha ${Alpha[$1]} --gamma ${Gamma[$1]} --d ${Dim[$1]} --tg $t  >> ../log/vary_tg_$1.log
	python3 eval.py --data $1 --algo TPC  >> ../log/vary_tg_$1.log
done
