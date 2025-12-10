# DQN Training for JetPack JoyRide

This guide explains how to train a Deep Q-Network (DQN) agent to play your JetPack JoyRide game.

## 📁 File Structure

```
your_project/
├── game.py              # Original game (unchanged)
├── agent.py             # Agent class (unchanged)
├── obstacle.py          # Obstacle class (unchanged)
├── coins.py             # Coin class (unchanged)
├── spawner.py           # Spawner class (unchanged)
├── platform.py          # Platform class (unchanged)
├── environment.py       # Game constants (unchanged)
├── hud.py              # HUD display (unchanged)
├── score_manager.py    # Score management (unchanged)
├── state_extractor.py  # State extraction (unchanged)
├── game_gym.py         # NEW: Gym wrapper for RL
├── dqn_agent.py        # NEW: DQN implementation
└── train.py            # NEW: Training script
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy matplotlib pygame
```

### 2. Start Training

**Basic training (1000 episodes):**
```bash
python train.py
```

**Custom training:**
```bash
python train.py --episodes 500 --render-every 25 --save-every 50
```

### 3. Watch Training Progress

The game will render every 50 episodes by default. You'll see:
- Episode number
- Score achieved
- Coins collected
- Average score (last 100 episodes)
- Epsilon (exploration rate)
- Loss value

### 4. Evaluate Trained Model

```bash
python train.py --eval checkpoints/dqn_final.pth
```

## 📊 Understanding the Output

During training, you'll see output like:
```
Ep  100/1000 | Score:  234 | Coins:  3 | Avg:  156.2 | ε: 0.605 | Loss: 0.1234
```

- **Ep**: Current episode / Total episodes
- **Score**: Score achieved in this episode
- **Coins**: Coins collected in this episode
- **Avg**: Average score over last 100 episodes (main metric!)
- **ε (epsilon)**: Exploration rate (1.0 = random, 0.01 = mostly exploit)
- **Loss**: Training loss (should decrease over time)

## 📈 Training Progress

### Checkpoints
- Saved every 100 episodes to `checkpoints/dqn_episode_N.pth`
- Final model saved as `checkpoints/dqn_final.pth`
- Interrupted training saved as `checkpoints/dqn_interrupted.pth`

### Plots
Training plots saved to `plots/` directory:
- Score progression over episodes
- Coins collected over episodes

## 🎮 How It Works

### State Representation (27 features)
The agent observes:
1. **Speed multiplier** (1 value)
2. **Agent position & velocity** (3 values)
3. **2 nearest obstacles** (16 values):
   - Position, size, rotation, rotation speed
   - Time to collision
4. **3 nearest coins** (12 values):
   - Position, distance

### Actions (2 choices)
- **Action 0**: Do nothing (gravity pulls down)
- **Action 1**: Press space (fly up)

### Reward Function
- **+0.1** per frame survived
- **+10.0** per coin collected
- **-100.0** on collision (death)

## 🔧 Advanced Options

### Continue Training from Checkpoint
```bash
python train.py --continue checkpoints/dqn_episode_500.pth --episodes 500
```

### Adjust Hyperparameters
```bash
python train.py --lr 0.0005 --gamma 0.95 --epsilon-decay 0.99
```

Available parameters:
- `--lr`: Learning rate (default: 0.001)
- `--gamma`: Discount factor (default: 0.99)
- `--epsilon-decay`: Exploration decay (default: 0.995)

### Render Less/More Often
```bash
# Render every 100 episodes (faster training)
python train.py --render-every 100

# Render every 10 episodes (watch learning closely)
python train.py --render-every 10
```

## 📖 Training Stages

### Stage 1: Random Exploration (Episodes 0-200)
- ε = 1.0 → 0.5
- Agent takes mostly random actions
- Learns basic obstacle avoidance
- Average score: 50-200

### Stage 2: Learning Basics (Episodes 200-500)
- ε = 0.5 → 0.2
- Agent starts avoiding obstacles consistently
- Average score: 200-500

### Stage 3: Optimization (Episodes 500-1000)
- ε = 0.2 → 0.05
- Agent learns to collect coins efficiently
- Average score: 500-1500+

### Stage 4: Mastery (Episodes 1000+)
- ε = 0.05 → 0.01
- Near-optimal play
- Average score: 1500-3000+

## 🐛 Troubleshooting

### "Module not found" errors
```bash
# Make sure all original game files are in the same directory
# Install dependencies:
pip install torch numpy matplotlib pygame
```

### Training is slow
```bash
# Render less frequently:
python train.py --render-every 100

# Or don't render at all (fastest):
# Modify game_gym.py and comment out the render calls
```

### Agent not improving
- Check if loss is decreasing
- Try lower learning rate: `--lr 0.0005`
- Try different epsilon decay: `--epsilon-decay 0.99`
- Train for more episodes: `--episodes 2000`

### Game window freezes
This is normal - the window only updates when rendering.
Press Ctrl+C to stop training gracefully.

## 💾 File Sizes

- Each checkpoint: ~50 KB
- Training for 1000 episodes: ~5 MB total
- Plots: ~100 KB each

## 🎯 Expected Results

After 1000 episodes of training:
- Random agent: 50-100 average score
- Trained DQN: 500-2000+ average score
- The agent should:
  - Avoid obstacles consistently
  - Collect coins when safe
  - Balance risk vs reward

## 📚 Next Steps

### Improve Performance
1. **Tune hyperparameters**: Try different learning rates, gamma values
2. **Adjust reward function**: In `dqn_agent.py`, modify `compute_reward()`
3. **Add more training**: Continue from checkpoint for another 1000 episodes
4. **Try Double DQN**: Implement Double DQN for more stable learning

### Advanced Techniques
1. **Prioritized Experience Replay**: Sample important transitions more often
2. **Dueling DQN**: Separate value and advantage streams
3. **Multi-step Returns**: Use n-step TD learning
4. **Rainbow DQN**: Combine all improvements

### Alternative Algorithms
- **PPO** (Proximal Policy Optimization): More stable than DQN
- **A3C** (Asynchronous Actor-Critic): Faster training with parallel agents
- **SAC** (Soft Actor-Critic): Best for continuous control

## 🤝 Tips for Success

1. **Be patient**: Good results take 500-1000 episodes
2. **Monitor average score**: This is your main metric
3. **Save checkpoints**: You can always continue training later
4. **Experiment**: Try different reward functions and hyperparameters
5. **Use GPU**: Training is 5-10x faster with CUDA

## 📞 Getting Help

If the agent isn't learning:
1. Check that all files are in place
2. Verify state extraction is working (run `game_gym.py` alone)
3. Make sure rewards are being calculated correctly
4. Try training for longer (2000+ episodes)

Good luck with your training! 🚀