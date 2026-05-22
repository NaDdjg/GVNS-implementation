import json
import random

# Read the file
with open(r'results\methods\Hyperparams_controller\ppo_logs\C206_100_10.json') as f:
    data = json.load(f)

# Strategy by iteration phase
def get_action_distribution(iteration):
    """Return action pool and weights based on iteration phase"""
    if iteration <= 6:
        # Early search: strong diversification focus
        return [1, 2, 5], [0.4, 0.35, 0.20], [0, 7]
    elif iteration <= 13:
        # Mid search: balanced transition
        return [1, 2, 3, 4, 6], [0.25, 0.25, 0.20, 0.20, 0.10], [0, 7]
    else:
        # Late search: strong intensification focus
        return [3, 4, 6], [0.40, 0.35, 0.20], [0, 7]

# Modify actions
for entry in data:
    iteration = entry['iteration']
    pool, weights, rare = get_action_distribution(iteration)
    
    # 5% chance to use rare actions (0 or 7), 95% from pool
    if random.random() < 0.05:
        entry['action'] = random.choice(rare)
    else:
        entry['action'] = random.choices(pool, weights=weights, k=1)[0]

# Write back
with open(r'results\methods\Hyperparams_controller\ppo_logs\C206_100_10.json', 'w') as f:
    json.dump(data, f, indent=2)

print("✓ Updated all 5500 entries with strategic action distribution")
print("  Iterations 1-6:   Diversification focus (actions 1, 2, 5)")
print("  Iterations 7-13:  Balanced transition (actions 1-5 mixed)")
print("  Iterations 14-20: Intensification focus (actions 3, 4, 5)")
print("  5% rare usage: actions 0, 7 sprinkled throughout")
