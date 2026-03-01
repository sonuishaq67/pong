import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from model import Model
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
import ale_py
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, TransformObservation
import numpy as np
import torch

torch.backends.cudnn.benchmark = True

N_ENVS = 256
N_EPISODES = 10000
HIDDEN = 512


def crop_frame(obs):
    return obs[34:, :]


def make_env():
    def _init():
        env = gym.make("ALE/Pong-v5", frameskip=4)
        env = GrayscaleObservation(env, keep_dim=False)
        cropped_obs_space = spaces.Box(low=0, high=255, shape=(176, 160), dtype=np.uint8)
        env = TransformObservation(env, crop_frame, cropped_obs_space)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, 4)
        return env
    return _init


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device} | Envs: {N_ENVS} | Episodes: {N_EPISODES}")

vec_env = AsyncVectorEnv([make_env() for _ in range(N_ENVS)])
obs_shape = vec_env.single_observation_space.shape
n_actions = vec_env.single_action_space.n

model = Model(action_dim=n_actions, hidden_dim=HIDDEN, observation_shape=obs_shape).to(device)
model = torch.compile(model, mode='reduce-overhead')
model._orig_mod.load_model()
model.eval()

obs_np, _ = vec_env.reset()
obs = torch.tensor(np.asarray(obs_np), dtype=torch.uint8, device=device)

ep_reward = np.zeros(N_ENVS)
ep_agent_pts = np.zeros(N_ENVS)
ep_opp_pts = np.zeros(N_ENVS)

margins = []
agent_scores = []
opp_scores = []
episodes_done = 0

print(f"\n{'Ep':>6}  {'Result':<6}  {'Score':^7}  {'Margin':>7}")
print("-" * 36)

with torch.inference_mode():
    while episodes_done < N_EPISODES:
        q_vals = model(obs.float())
        actions = torch.argmax(q_vals, dim=1).cpu().numpy()

        next_obs_np, rewards, terminated, truncated, _ = vec_env.step(actions)
        dones = terminated | truncated

        ep_reward += rewards
        ep_agent_pts += np.maximum(rewards, 0)
        ep_opp_pts += np.maximum(-rewards, 0)

        for i in range(N_ENVS):
            if dones[i] and episodes_done < N_EPISODES:
                margin = ep_reward[i]
                a_pts = int(ep_agent_pts[i])
                o_pts = int(ep_opp_pts[i])

                margins.append(margin)
                agent_scores.append(a_pts)
                opp_scores.append(o_pts)

                result = "WIN" if margin > 0 else ("LOSS" if margin < 0 else "DRAW")
                print(f"{episodes_done+1:>6}  {result:<6}  {a_pts:>2} - {o_pts:<2}  {margin:>+7.0f}")

                ep_reward[i] = 0.0
                ep_agent_pts[i] = 0.0
                ep_opp_pts[i] = 0.0
                episodes_done += 1

        obs = torch.tensor(np.asarray(next_obs_np), dtype=torch.uint8, device=device)

margins = np.array(margins)
agent_scores = np.array(agent_scores)
opp_scores = np.array(opp_scores)

wins = int((margins > 0).sum())
losses = int((margins < 0).sum())
draws = int((margins == 0).sum())
best_idx = margins.argmax()
worst_idx = margins.argmin()

print("\n" + "=" * 50)
print(f"  EVALUATION SUMMARY  ({N_EPISODES} games)")
print("=" * 50)
print(f"  Win rate:         {wins / N_EPISODES * 100:6.2f}%  ({wins} wins)")
print(f"  Loss rate:        {losses / N_EPISODES * 100:6.2f}%  ({losses} losses)")
print(f"  Draw rate:        {draws / N_EPISODES * 100:6.2f}%  ({draws} draws)")
print(f"  Avg margin:       {margins.mean():+.3f}  (std {margins.std():.3f})")
print(f"  Median margin:    {np.median(margins):+.1f}")
print(f"  Avg agent score:  {agent_scores.mean():.3f}")
print(f"  Avg opp score:    {opp_scores.mean():.3f}")
print(f"  Best game:        {margins[best_idx]:+.0f}  ({agent_scores[best_idx]} - {opp_scores[best_idx]})")
print(f"  Worst game:       {margins[worst_idx]:+.0f}  ({agent_scores[worst_idx]} - {opp_scores[worst_idx]})")
print(f"  Perfect wins:     {int((agent_scores == 21).sum())}  ({(agent_scores == 21).mean()*100:.1f}%)")
print(f"  Perfect losses:   {int((opp_scores == 21).sum())}  ({(opp_scores == 21).mean()*100:.1f}%)")
print("=" * 50)
