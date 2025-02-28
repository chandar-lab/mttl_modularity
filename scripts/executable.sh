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
# sbatch run.sh 'AdaptLLM/medicine-LLM-13B'
# sbatch run.sh 'camel-ai/CAMEL-13B-Combined-Data'
# sbatch run.sh 'meta-llama/Llama-3.2-3B-Instruct'
# sbatch run.sh 'meta-llama/Llama-3.2-3B'

# sbatch run.sh 'meta-llama/Llama-3.1-8B'
# sbatch run.sh 'meta-llama/Llama-3.1-8B-Instruct'
# sbatch run.sh 'meta-llama/Meta-Llama-3-8B'

# sbatch run.sh 'meta-llama/Llama-2-13b-hf'
# sbatch run.sh 'lmsys/vicuna-13b-v1.5'
# sbatch run.sh 'meta-llama/Llama-3.2-1B'
# sbatch run.sh 'meta-llama/Llama-3.2-1B-Instruct'

# sbatch run.sh 'meta-llama/Llama-2-7b-chat-hf' 'fwnlp'
# sbatch run.sh 'meta-llama/Llama-2-7b-chat-hf' 'llm_attack'


# tasks=("a-safety" "adversarial_qa_droberta_answer_the_following_q" "camel-biology" "beaverTails_unsafe_animal_abuse" "beaverTails_unsafe_child_abuse" "beaverTails_unsafe_controversial_topics_politics" "beaverTails_unsafe_discrimination_stereotype_injustice" "beaverTails_unsafe_drug_abuse_weapons_banned_substance" "beaverTails_unsafe_financial_crime_property_crime_theft" "beaverTails_unsafe_hate_speech_offensive_language" "beaverTails_unsafe_misinformation_regarding_ethics_laws_and_safety" "beaverTails_unsafe_non_violent_unethical_behavior" "beaverTails_unsafe_privacy_violation" "beaverTails_unsafe_self_harm" "beaverTails_unsafe_sexually_explicit_adult_content" "beaverTails_unsafe_terrorism_organized_crime_unsafe" "beaverTails_unsafe_violence_aiding_and_abetting_incitement")

tasks=("beaverTails_unsafe_controversial_topics_politics" "beaverTails_unsafe_hate_speech_offensive_language" "beaverTails_unsafe_financial_crime_property_crime_theft" "beaverTails_unsafe_misinformation_regarding_ethics_laws_and_safety")
for task in ${tasks[@]}
do
    echo $task
    sbatch run.sh $task
done




