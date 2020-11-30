for fn in DATA/*.graph; do
	for seed in {1..10}; do
		echo $fn $seed
	 	python temp.py -input $fn -time 600 -seed $seed >> exec.log
	done  
done