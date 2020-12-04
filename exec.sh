for fn in DATA/*.graph; do
	for seed in {1..10}; do
		echo $fn $seed
	 	python Python/main.py -inst $fn -alg SA -time 600 -seed $seed >> exec_SA.log
	done 
done