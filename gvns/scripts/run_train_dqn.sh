INSTANCES=(
    "vrptws/100/R110_100_5.txt"
    "vrptws/100/C101_100_10.txt"
)

EPISODES=500
OUT_DIR="models/dqn"
DEVICE="cuda"  

mkdir -p $OUT_DIR

for inst in "${INSTANCES[@]}"; do
    name=$(basename "$inst" .txt)
    echo "Training DQN on $inst ..."
    python trainers/train_dqn.py \
        --instance "$inst" \
        --episodes $EPISODES \
        --out "$OUT_DIR/dqn_${name}.pth" \
        --device $DEVICE
done

echo "DQN training finished for all instances."
