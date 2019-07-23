launchall = open("all_scripts_tpun.sh", "w")

for model_type in ["ltr", "bi", "tree"]:
	for task in ["auto", "rev", "interleave", "sort"]:
		for number in ["0", "1", "2", "3", "4"]:
			foname = "_".join(["tpun", model_type, task, number])
			fo = open(foname + ".scr", "w")
			model_name = "_".join([model_type, model_type, task, number])

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
			fo.write("module load pytorch\n\n")	
			fo.write("python decompose.py --data_prefix " + model_name + " --role_scheme ltr --test_decoder True --decoder " + model_type + " --decoder_prefix " + model_name + " --decoder_embedding_size 10 --decoder_task " + task + "  --filler_dim 20 --role_dim 20 --vocab_size 10 --hidden_size 60 --decomp_type tpun\n")
			fo.write("python decompose.py --data_prefix " + model_name + " --role_scheme rtl --test_decoder True --decoder " + model_type + " --decoder_prefix " + model_name + " --decoder_embedding_size 10 --decoder_task " + task + "  --filler_dim 20 --role_dim 20 --vocab_size 10 --hidden_size 60 --decomp_type tpun\n")
			fo.write("python decompose.py --data_prefix " + model_name + " --role_scheme bi --test_decoder True --decoder " + model_type + " --decoder_prefix " + model_name + " --decoder_embedding_size 10 --decoder_task " + task + "  --filler_dim 20 --role_dim 20 --vocab_size 10 --hidden_size 60 --decomp_type tpun\n")
			fo.write("python decompose.py --data_prefix " + model_name + " --role_scheme wickel --test_decoder True --decoder " + model_type + " --decoder_prefix " + model_name + " --decoder_embedding_size 10 --decoder_task " + task + "  --filler_dim 20 --role_dim 20 --vocab_size 10 --hidden_size 60 --decomp_type tpun\n")
			fo.write("python decompose.py --data_prefix " + model_name + " --role_scheme tree --test_decoder True --decoder " + model_type + " --decoder_prefix " + model_name + " --decoder_embedding_size 10 --decoder_task " + task + "  --filler_dim 20 --role_dim 20 --vocab_size 10 --hidden_size 60 --decomp_type tpun\n")
			fo.write("python decompose.py --data_prefix " + model_name + " --role_scheme bow --test_decoder True --decoder " + model_type + " --decoder_prefix " + model_name + " --decoder_embedding_size 10 --decoder_task " + task + "  --filler_dim 20 --role_dim 20 --vocab_size 10 --hidden_size 60 --decomp_type tpun\n")





			launchall.write("sbatch " + foname + ".scr\n")


