declare -A Gamma=(["Cora"]="10" ["CiteSeer"]="6" ["MovieLens"]="6" ["LastFM"]="7" ["Google"]="10" ["Amazon"]="0")
declare -A Dim=(["Cora"]="128" ["CiteSeer"]="32" ["MovieLens"]="-1" ["LastFM"]="128" ["Google"]="32" ["Amazon"]="64")

for a in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
do
	python3.8 proposed.py --data $1  --alpha $a --gamma ${Gamma[$1]} --d ${Dim[$1]}  >> ../log/vary_alpha_$1.log
	python3 eval.py --data $1 --algo TPC  >> ../log/vary_alpha_$1.log
done
