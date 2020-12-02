for fn in DATA/*.graph; do
	for seed in {1..10}; do
		echo $fn $seed
<<<<<<< HEAD
	 	python Python/main.py -inst $fn -alg SA -time 600 -seed $seed >> exec_SA.log
	done  
done
=======
	 	python Python/main.py -inst $fn -alg BnB -time 600 -seed $seed >> exec_BnB.log
	done  
done
>>>>>>> caa16cc99a7335c2dd5d9afda951c6a5c535118f
