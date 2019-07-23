launchall = open("all_scripts_gen_auto_rev.sh", "w")

for model_type in ["ltr"]:
	for task in ["auto", "rev"]:
		for fixedvar in ["fixed6", "fixed6_withheld", "var6", "var6_withheld", "var6_withheld_rtl"]:
			for number in ["0", "1", "2", "3", "4"]:
				model_name = fixedvar + "_".join([model_type, model_type, task, number])
	
				foname = "_".join(["gen_tests", model_name])
				fo = open(foname + ".scr", "w")

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

				for genfile in ["fixed10gen", "fixed10_withheld_gen", "fixed1gen", "fixed2gen", "fixed3gen", "fixed4gen", "fixed5gen", "fixed6gen", "fixed7gen", "fixed8gen", "fixed9gen", "fixed10gen", "fixed11gen", "fixed12gen", "fixed13gen", "fixed14gen", "fixed15gen", "fixed6_withheld_gen", "var10_1", "var10_2", "var10_3", "var10_4", "var10_5", "var10_6", "var10_7", "var10_8", "var10_9", "var10_10", "var10_withheld_gen", "var10_withheld_rtl_gen", "var6_1", "var6_2", "var6_3", "var6_4", "var6_5", "var6_6", "var6_withheld_gen", "var6_withheld_rtl_gen"]:
					fo.write("python model_trainer.py --encoder ltr --decoder ltr --task " + task + " --prefix digits_" + genfile + "  --prefix_prefix " + genfile +  " --train False --train_enc False --train_dec False --enc_file_prefix " + model_name + "  --dec_file_prefix  " + model_name + "\n")
		
				

				launchall.write("sbatch " + foname + ".scr\n")


