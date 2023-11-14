declare -A Alpha=(["Cora"]="0.9" ["CiteSeer"]="0.6" ["MovieLens"]="0.6" ["LastFM"]="0.5" ["Google"]="0.9" ["Amazon"]="0.5")
declare -A Dim=(["Cora"]="128" ["CiteSeer"]="32" ["MovieLens"]="-1" ["LastFM"]="128" ["Google"]="32" ["Amazon"]="64")

for g in {0..10}
do
	python3.8 proposed.py --data $1  --alpha ${Alpha[$1]} --gamma $g --d ${Dim[$1]}  >> ../log/vary_gamma_$1.log
	python3 eval.py --data $1 --algo TPC  >> ../log/vary_gamma_$1.log
done
