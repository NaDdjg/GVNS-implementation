INSTANCES=(
    "vrptws/100/R110_100_5.txt"
    "vrptws/100/C101_100_10.txt"
)

TIMESTEPS=50000
MAX_STEPS=50
BVND_ITERS=500
SEED=42
VEHICLES=30

for inst in "${INSTANCES[@]}"; do
    name=$(basename "$inst" .txt)
    OUT="models/ppo/ppo_${name}.zip"

    echo "Training PPO on $inst ..."
    python trainers/train_ppo.py \
        --instance "$inst" \
        --save_path "$OUT" \
        --timesteps $TIMESTEPS \
        --max_steps $MAX_STEPS \
        --bvnd_max_iters $BVND_ITERS \
        --vehicles $VEHICLES \
        --seed $SEED
done

echo "All PPO trainings finished."

# run like ; TIMESTEPS=100000 bash scripts/run_train_ppo.sh besh nbdl variables 