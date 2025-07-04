#!/bin/zsh

# This script runs the PPO training experiments sequentially.

echo "======================================================"
echo "🚀 STARTING EXPERIMENT 1: Training WITH Curriculum Learning"
echo "======================================================"

# Run the training script with curriculum learning enabled ('y')
uv run train_ppo_tune.py --use-curriculum y

echo "======================================================"
echo "✅ EXPERIMENT 1 FINISHED"
echo "🚀 STARTING EXPERIMENT 2: Training WITHOUT Curriculum Learning"
echo "======================================================"

# Run the training script with curriculum learning disabled ('n')
uv run train_ppo_tune.py --use-curriculum n

echo "======================================================"
echo "✅ EXPERIMENT 2 FINISHED"
echo "All experiments are complete."
echo "======================================================"
