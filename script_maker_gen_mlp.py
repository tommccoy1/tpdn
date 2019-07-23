launchall = open("all_scripts_gen_mlp.sh", "w")

for model_type in ["mlp"]:
	for task in ["auto", "rev", "interleave", "sort"]:
		for number in ["0", "1", "2", "3", "4"]:
			for lcs in [(1,0), (1,1), (1,2), (1,3), (1,4), (2,0), (2,1), (2,2), (2,3), (3,0), (3,1), (3,2), (4,0), (4,1), (5,0)]:
				foname = "_".join(["gen", model_type, str(lcs[0]) + str(lcs[1]), task, number])
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
				fo.write("\n")
				fo.write("module load pytorch\n")	
				fo.write("module load cuda\n\n")
				fo.write("python generate_vectors.py --prefix digits_9 --encoder mlp --decoder mlp --task " + task + " --enc_prefix " + model_name + " --dec_prefix " + model_name + " --n_hidden_enc " + str(lcs[0]) + " --n_hidden_dec " + str(lcs[1]) + "\n")	




				launchall.write("sbatch " + foname + ".scr\n")


