for fn in DATA/*.graph; do
	for seed in {1..10}; do
		echo $fn $seed
	 	python Python/main.py -inst $fn -alg BnB -time 600 -seed $seed >> exec_BnB.log
	done  
done
