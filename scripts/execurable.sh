#task_nums=(22) #(22 21)

###################################### run for shuffle label
# ranks=(32)
# epochs=(8)


# for r in ${ranks[@]}
# do
#   for epoch in ${epochs[@]}
#   do
#     echo $r $epoch
#     # output_dir_1="/home/mila/m/maryam.hashemzadeh/scratch/verifier/gp_verifier/checkpoint/"
#     # sbatch run_lora.sh $r $epoch "/home/mila/m/maryam.hashemzadeh/scratch/verifier/downsampled_prm800k_test.json" "/home/mila/m/maryam.hashemzadeh/scratch/verifier/downsampled_prm800k_train.json" $output_dir_1 False
#     output_dir_1="/home/mila/m/maryam.hashemzadeh/scratch/verifier/gp_verifier/checkpoint_phase2/"
#     sbatch run_lora.sh $r $epoch "/home/mila/m/maryam.hashemzadeh/scratch/verifier/downsampled_prm800k_test_phase2.json" "/home/mila/m/maryam.hashemzadeh/scratch/verifier/downsampled_prm800k_train_phase2.json" $output_dir_1 False
#     # output_dir_2="/home/mila/m/maryam.hashemzadeh/scratch/verifier/gp_verifier/finalanswer_checkpoint_phase2/"
#     # sbatch run_lora.sh $r $epoch "/home/mila/m/maryam.hashemzadeh/scratch/verifier/finalanswer_downsampled_prm800k_test_phase2.json" "/home/mila/m/maryam.hashemzadeh/scratch/verifier/finalanswer_downsampled_prm800k_train_phase2.json" $output_dir_2 True
#   done
# done

# sbatch run.sh 'QuantFactory/Llama-3-8B-Instruct-Finance-RAG-GGUF' ### tokenizer is not found 
# sbatch run.sh 'HPAI-BSC/Llama3-Aloe-8B-Alpha'
sbatch run.sh 'AdaptLLM/medicine-LLM-13B'
sbatch run.sh 'camel-ai/CAMEL-13B-Combined-Data'



# sbatch run.sh 'meta-llama/Llama-3.2-3B-Instruct'
# sbatch run.sh 'meta-llama/Llama-3.2-3B'