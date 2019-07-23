launchall = open("all_scripts_tpun_mlp.sh", "w")

for model_type in ["mlp"]:
	for task in ["auto", "rev", "interleave", "sort"]:
		for number in ["0", "1", "2", "3", "4"]:
			for lcs in [(1,0), (1,1), (1,2), (1,3), (1,4), (2,0), (2,1), (2,2), (2,3), (3,0), (3,1), (3,2), (4,0), (4,1), (5,0)]:
				foname = "_".join(["dectpun", model_type, str(lcs[0]) + str(lcs[1]), task, number])
				fo = open(foname + ".scr", "w")
				model_name = "_".join([str(lcs[1]) + "declayers", str(lcs[0]) + "enclayers", model_type, model_type, task, number])
	
				fo.write("#!/bin/bash\n")
				fo.write("#SBATCH --job-name=MyJob\n")
				fo.write("#SBATCH --time=2:0:0\n")
				fo.write("#SBATCH --partition=gpuk80\n")
				fo.write("#SBATCH --gres=gpu:1\n")
				fo.write("#SBATCH --ntasks-per-node=1\n")
				fo.write("#SBATCH --cpus-per-task=6\n")
				fo.write("#SBATCH --mail-type=end\n")
				fo.write("#SBATCH --mail-user=rmccoy20@jhu.edu\n")
				fo.write("#SBATCH --output=" + foname + ".log\n")
				fo.write("#SBATCH --error=" + foname + ".err\n")
				fo.write("#SBATCH --exclude=gpu008,gpu019,gpu037,gpu018\n")
				fo.write("\n")
				fo.write("module load pytorch\n")	
				fo.write("module load cuda\n\n")

				fo.write("python decompose.py --data_prefix " + model_name + " --role_scheme ltr --test_decoder True --decoder mlp --decoder_prefix " + model_name + " --decoder_task " + task + " --filler_dim 20 --role_dim 20 " + " --n_hidden_dec " + str(lcs[1]) + " --decomp_type tpun\n")
				fo.write("python decompose.py --data_prefix " + model_name + " --role_scheme rtl --test_decoder True --decoder mlp --decoder_prefix " + model_name + " --decoder_task " + task + " --filler_dim 20 --role_dim 20 " + " --n_hidden_dec " + str(lcs[1]) + " --decomp_type tpun\n") 
				fo.write("python decompose.py --data_prefix " + model_name + " --role_scheme bi --test_decoder True --decoder mlp --decoder_prefix " + model_name + " --decoder_task " + task + " --filler_dim 20 --role_dim 20 " + " --n_hidden_dec " + str(lcs[1]) + " --decomp_type tpun\n") 
				fo.write("python decompose.py --data_prefix " + model_name + " --role_scheme wickel --test_decoder True --decoder mlp --decoder_prefix " + model_name + " --decoder_task " + task + " --filler_dim 20 --role_dim 20 " + " --n_hidden_dec " + str(lcs[1]) + " --decomp_type tpun\n") 
				fo.write("python decompose.py --data_prefix " + model_name + " --role_scheme tree --test_decoder True --decoder mlp --decoder_prefix " + model_name + " --decoder_task " + task + " --filler_dim 20 --role_dim 20 " + " --n_hidden_dec " + str(lcs[1]) + " --decomp_type tpun\n") 
				fo.write("python decompose.py --data_prefix " + model_name + " --role_scheme bow --test_decoder True --decoder mlp --decoder_prefix " + model_name + " --decoder_task " + task + " --filler_dim 20 --role_dim 20 " + " --n_hidden_dec " + str(lcs[1]) + " --decomp_type tpun\n") 



				launchall.write("sbatch " + foname + ".scr\n")


