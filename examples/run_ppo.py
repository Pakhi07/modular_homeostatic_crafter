import argparse
import homeostatic_crafter
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import numpy as np
from collections import defaultdict
import os
import torch

class AnalysisCallback(BaseCallback):
    def __init__(self, log_interval=4096, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.visited_states = defaultdict(int)
        self.actions = []
        self.positions = []
        self.rewards = []
        self.healths = []
        self.place_stone_actions = 0
        self.zombies_defeated = 0
        self.skeletons_defeated = 0
        self.wake_ups = 0
        self.death_count = 0
        self.achievements_unlocked = defaultdict(int)
        self.episodes = 0

    def _on_step(self):
        # Access info from the environment
        info = self.locals['infos'][-1]
        action = self.locals['actions'][-1]
        action_name = self.training_env.get_attr('action_names')[0][action]
        
        if action_name == 'place_stone':
            self.place_stone_actions += 1
        
        if info.get('discount') == 0.0:
            self.death_count += 1

        # Check for newly unlocked achievements in this step
        current_achievements = info.get('achievements', {})
        if current_achievements.get('defeat_zombie', 0) > self.achievements_unlocked.get('defeat_zombie', 0):
            self.zombies_defeated += 1
        
        if current_achievements.get('defeat_skeleton', 0) > self.achievements_unlocked.get('defeat_skeleton', 0):
            self.skeletons_defeated += 1

        if current_achievements.get('wake_up', 0) > self.achievements_unlocked.get('wake_up', 0):
            self.wake_ups += 1
            
        # Update our running count of achievements for the next step (only one line needed)
        self.achievements_unlocked.update(current_achievements)

        self.rewards.append(info.get('reward', 0))
        self.actions.append(action) # Use the action variable we already have
        
        obs = self.locals['new_obs']['obs'][0]
        state_key = str(obs.tobytes())
        self.visited_states[state_key] += 1
        
        self.healths.append(info.get('player_health', 0))
        self.positions.append(info.get('player_pos', (0, 0)))
        if 'episodes' in info:
            self.episodes = max(self.episodes, info['episodes'])  
        
        if self.n_calls > 0 and self.n_calls % self.log_interval == 0:
            metrics = self.compute_metrics()
            for key, value in metrics.items():
                self.logger.record(f'custom/{key}', value)
            print(f"Step {self.n_calls}: Logged metrics to TensorBoard.")
            
            self.rewards.clear()
            self.healths.clear()
            self.positions.clear()
            self.actions.clear()
            torch.cuda.empty_cache()

        return True

    def compute_metrics(self):
        positions = np.array(self.positions)
        exploration_variance = np.var(positions, axis=0).sum() if len(positions) > 1 else 0
        state_counts = np.array(list(self.visited_states.values()))
        state_probs = state_counts / (state_counts.sum() + 1e-10)
        state_entropy = -np.sum(state_probs * np.log2(state_probs + 1e-10))
        action_counts = np.bincount(self.actions, minlength=self.training_env.get_attr('action_space')[0].n)
        action_probs = action_counts / (action_counts.sum() + 1e-10)
        action_entropy = -np.sum(action_probs * np.log2(action_probs + 1e-10))
        reward_mean = np.mean(self.rewards)
        health_mean = np.mean(self.healths) if self.healths else 0
        return {
            'exploration_variance': exploration_variance,
            'state_entropy': state_entropy,
            'action_entropy': action_entropy,
            'reward_homeostatic_mean': reward_mean,
            'health_mean': health_mean,
            'total_zombie_defeated': self.zombies_defeated,
            'total_skeleton_defeated': self.skeletons_defeated,
            'total_wake_ups': self.wake_ups,
            'total_stones_placed': self.place_stone_actions,
            'total_deaths': self.death_count,
            'total_episodes': self.episodes,
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='homeostatic', choices=['crafter', 'homeostatic'])
    parser.add_argument('--outdir', type=str, default='logdir/homeostatic_reward-ppo/0')
    parser.add_argument('--steps', type=float, default=250000)  # Updated to 250k
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,  # save every 100k steps (adjust as needed)
        save_path=args.outdir,
        name_prefix=f"{args.env}_seed{args.seed}_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    env_class = homeostatic_crafter.Env
    env = env_class(seed=args.seed)
    env = homeostatic_crafter.Recorder(
        env, 
        f"{args.outdir}/{args.env}_eval/seed_{args.seed}",
        save_stats=True,
        save_episode=False,
        save_video=False,
    )

    env = DummyVecEnv([lambda: env])
    # env = VecTransposeImage(env)

    model = stable_baselines3.PPO(
        'MultiInputPolicy', 
        env, 
        verbose=1, 
        tensorboard_log=args.outdir,
        seed=args.seed)

    analysis_callback = AnalysisCallback(log_interval=4096)
    callback = CallbackList([analysis_callback, checkpoint_callback])
    
    print(f"Starting {args.env} training with seed {args.seed}. Logs in {args.outdir}")  
    
    model.learn(
        total_timesteps=int(args.steps), 
        callback=callback
    )
    print(f"{args.env} training finished.")

    print("\n--- Checking Model Parameters ---")
    for name, param in model.policy.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Shape: {param.shape}")
    print("-----------------------------\n")

    metrics = analysis_callback.compute_metrics()
    print(f"Final {args.env} Seed {args.seed} Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    try:
        final_log_path = model.logger.dir
    
        os.makedirs(final_log_path, exist_ok=True)
        
        with open(f"{final_log_path}/{args.env}_seed{args.seed}_metrics.txt", 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
                
        model.save(f"{final_log_path}/{args.env}_seed{args.seed}_model")
        
        print(f"Metrics and model saved successfully to {final_log_path}")
    except Exception as e:
        print(f"Warning: Could not save to {args.outdir}: {e}")

if __name__ == '__main__':
    main()