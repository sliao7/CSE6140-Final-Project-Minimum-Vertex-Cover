for fn in DATA/*.graph; do
	for seed in {1..2}; do
		echo $fn $seed
	 	python Python/main.py -inst DATA/jazz.graph -alg BnB -time 600 -seed $seed >> exec.log
	done  
	break
done