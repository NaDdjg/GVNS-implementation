INSTANCES=(
    "vrptws/100/R110_100_5.txt"
    "vrptws/100/C101_100_10.txt"
)

EPISODES=300
OUT_DIR="models/qlearning"

mkdir -p $OUT_DIR

for inst in "${INSTANCES[@]}"; do
    name=$(basename "$inst" .txt)
    echo "Training Q-Learning on $inst ..."
    python trainers/train_qlearning.py \
        --instance "$inst" \
        --episodes $EPISODES \
        --out "$OUT_DIR/qlearning_${name}.pkl"
done

echo "Q-learning training finished for all instances."
