task="StairClimberSparse"
search_method="Scheduled_HillClimbing"
search_space="ProgrammaticSpace"
start_k=32
end_k=2048
starting_seed=0
interval=2
ending_seed=$(($starting_seed + $interval - 1))

# Activate conda environment and run the script
# conda activate llm_gs_env
# export OPENROUTER_API_KEY="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"

for seed in $(seq $starting_seed $ending_seed); do
    python stairclimber.py --seed ${seed} --task ${task} --start_k ${start_k} --end_k ${end_k} --output_name "LLM-GS" --search_method ${search_method} --search_space ${search_space} --llm_program_num 5
done
