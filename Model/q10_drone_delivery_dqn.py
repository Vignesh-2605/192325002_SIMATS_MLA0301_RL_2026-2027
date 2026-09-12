import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple("Transition", "state action reward next_state done")


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
class DroneDeliveryEnv:
    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "CHARGE"]
    DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    CHARGE = 4

    # reward constants
    STEP_COST = -0.02
    SHAPING = 0.20
    WASTED_ACTION = -0.10
    PICKUP_REWARD = 2.0
    DELIVERY_REWARD = 15.0
    CRASH_PENALTY = -40.0

    def __init__(self, size=8, battery_max=30.0, max_steps=150, seed=None):
        self.size = size
        self.battery_max = battery_max
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.obs_dim = 11
        self.n_actions = len(self.ACTIONS)

        self.drain_empty = 1.0
        self.drain_loaded = 1.5
        self.drain_idle = 0.5        # hovering still costs power
        self.recharge_rate = 30.0      # one CHARGE at the depot fills the pack

    # -------------------------------------------------------------- helpers
    def _random_cell(self, exclude=()):
        while True:
            cell = (int(self.rng.integers(self.size)), int(self.rng.integers(self.size)))
            if cell not in exclude:
                return cell

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _target(self):
        return self.customer if self.carrying else self.pickup

    def _observe(self):
        s = self.size - 1
        tgt = self._target()
        drain = self.drain_loaded if self.carrying else self.drain_empty
        slack = (self.battery / drain) - self._manhattan(self.drone, self.depot)
        return np.array([
            self.drone[0] / s,
            self.drone[1] / s,
            tgt[0] / s,
            tgt[1] / s,
            (tgt[0] - self.drone[0]) / s,
            (tgt[1] - self.drone[1]) / s,
            (self.depot[0] - self.drone[0]) / s,
            (self.depot[1] - self.drone[1]) / s,
            self.battery / self.battery_max,
            1.0 if self.carrying else 0.0,
            np.clip(slack / (2 * s), -1.0, 1.0),
        ], dtype=np.float32)

    # ---------------------------------------------------------------- reset
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.depot = (self.size // 2, self.size // 2)
        self.pickup = self._random_cell(exclude=(self.depot,))
        self.customer = self._random_cell(exclude=(self.depot, self.pickup))
        self.drone = self.depot
        self.battery = self.battery_max
        self.carrying = False
        self.deliveries = 0
        self.charges = 0
        self.t = 0
        return self._observe()

    # ----------------------------------------------------------------- step
    def step(self, action):
        self.t += 1
        reward = self.STEP_COST
        info = {}

        if action == self.CHARGE:
            if self.drone == self.depot and self.battery < self.battery_max:
                # docked, so no drain this tick
                self.battery = min(self.battery_max, self.battery + self.recharge_rate)
                self.charges += 1
                info["charged"] = True
                # deliberately no bonus: charging is worth doing only because a
                # flat battery is catastrophic, not because it pays directly
            else:
                # hovering off-pad or topping up a full battery: wasted tick,
                # and the rotors still draw power, so waiting is never free
                self.battery -= self.drain_idle
                reward += self.WASTED_ACTION

        else:
            prev_target_dist = self._manhattan(self.drone, self._target())

            dr, dc = self.DELTAS[action]
            nr = int(np.clip(self.drone[0] + dr, 0, self.size - 1))
            nc = int(np.clip(self.drone[1] + dc, 0, self.size - 1))
            if (nr, nc) == self.drone:
                reward += self.WASTED_ACTION        # flew into the boundary
            self.drone = (nr, nc)

            self.battery -= self.drain_loaded if self.carrying else self.drain_empty

            # pickup and delivery resolve on arrival
            if not self.carrying and self.drone == self.pickup:
                self.carrying = True
                reward += self.PICKUP_REWARD
                info["picked_up"] = True
            elif self.carrying and self.drone == self.customer:
                self.carrying = False
                self.deliveries += 1
                reward += self.DELIVERY_REWARD
                info["delivered"] = True
                self.pickup = self._random_cell(exclude=(self.depot, self.drone))
                self.customer = self._random_cell(
                    exclude=(self.depot, self.drone, self.pickup))
            else:
                # progress shaping only while the target is unchanged, so the
                # switch of target at pickup does not read as a jump backwards
                new_dist = self._manhattan(self.drone, self._target())
                reward += self.SHAPING * (prev_target_dist - new_dist)

        done = False
        if self.battery <= 0.0:
            self.battery = 0.0
            reward += self.CRASH_PENALTY
            done = True
            info["crashed"] = True

        if self.t >= self.max_steps:
            done = True

        return self._observe(), reward, done, info


# --------------------------------------------------------------------------
# network and replay
# --------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(128, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size, rng):
        idx = rng.integers(len(self.buf), size=batch_size)
        batch = [self.buf[int(i)] for i in idx]
        return (
            torch.as_tensor(np.array([b.state for b in batch]), dtype=torch.float32),
            torch.as_tensor([b.action for b in batch], dtype=torch.int64),
            torch.as_tensor([b.reward for b in batch], dtype=torch.float32),
            torch.as_tensor(np.array([b.next_state for b in batch]), dtype=torch.float32),
            torch.as_tensor([float(b.done) for b in batch], dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


# --------------------------------------------------------------------------
# agent
# --------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.99, batch_size=64,
                 buffer_size=50_000, target_sync=400, warmup=1_000,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=12_000,
                 device=None, seed=0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.warmup = warmup

        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps

        self.q = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_q = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()

        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.updates = 0

    @property
    def epsilon(self):
        frac = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.q(t).argmax(dim=1).item())

    def learn(self):
        if len(self.buffer) < max(self.batch_size, self.warmup):
            return None

        s, a, r, s2, d = self.buffer.sample(self.batch_size, self.rng)
        s, a, r, s2, d = (x.to(self.device) for x in (s, a, r, s2, d))

        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + self.gamma * (1.0 - d) * self.target_q(s2).max(dim=1).values

        loss = F.smooth_l1_loss(q, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return float(loss.item())


# --------------------------------------------------------------------------
# training and evaluation
# --------------------------------------------------------------------------
def evaluate(env, agent, episodes=10, base_seed=10_000):
    returns, deliveries, crashes = [], [], 0
    for i in range(episodes):
        obs = env.reset(seed=base_seed + i)
        total, done = 0.0, False
        while not done:
            action = agent.act(obs, greedy=True)
            obs, reward, done, info = env.step(action)
            total += reward
            crashes += int(info.get("crashed", False))
        returns.append(total)
        deliveries.append(env.deliveries)
    return float(np.mean(returns)), float(np.mean(deliveries)), crashes


def train(episodes=1200, seed=0, eval_every=100):
    torch.manual_seed(seed)
    random.seed(seed)

    env = DroneDeliveryEnv(seed=seed)
    eval_env = DroneDeliveryEnv(seed=seed + 999)
    agent = DQNAgent(env.obs_dim, env.n_actions, seed=seed)

    history = []
    for ep in range(1, episodes + 1):
        obs = env.reset()
        total, done = 0.0, False

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.buffer.push(obs, action, reward, next_obs, done)
            agent.steps += 1
            agent.learn()
            obs = next_obs
            total += reward

        history.append((ep, total, env.deliveries))

        if ep % eval_every == 0:
            ret, dlv, crashes = evaluate(eval_env, agent)
            recent = np.mean([h[1] for h in history[-eval_every:]])
            print(f"episode {ep:>5} | eps {agent.epsilon:.3f} "
                  f"| train return {recent:>8.2f} "
                  f"| eval return {ret:>8.2f} | eval deliveries {dlv:>5.2f} "
                  f"| crashes {crashes}/10")

    return env, agent, history


def demo_episode(env, agent, seed=12345, max_print=45):
    """Print one greedy episode step by step."""
    obs = env.reset(seed=seed)
    print("\nGreedy demonstration episode")
    print(f"depot {env.depot}   first pickup {env.pickup}   first customer {env.customer}")
    done, total, step = False, 0.0, 0

    while not done:
        action = agent.act(obs, greedy=True)
        obs, reward, done, info = env.step(action)
        total += reward
        step += 1

        tag = ""
        if info.get("picked_up"):
            tag = "  <- parcel collected"
        elif info.get("delivered"):
            tag = "  <- parcel delivered"
        elif info.get("charged"):
            tag = "  <- recharged at depot"
        if info.get("crashed"):
            tag = "  <- battery flat, drone down"

        if step <= max_print:
            print(f"  step {step:>3}  {DroneDeliveryEnv.ACTIONS[action]:<7} "
                  f"pos {env.drone}  battery {env.battery:5.1f}  "
                  f"carrying {int(env.carrying)}  reward {reward:+6.2f}{tag}")
        elif step == max_print + 1:
            print("  ...")

    print(f"  total reward {total:.2f} over {step} steps, "
          f"{env.deliveries} deliveries, {env.charges} recharges")


def main():
    print("Deep Q-Network — autonomous drone delivery under a battery constraint")
    print("=" * 72)
    env = DroneDeliveryEnv()
    print(f"Grid            : {env.size} x {env.size}, depot at the centre")
    print(f"Battery         : {env.battery_max:.0f} units, "
          f"{env.drain_empty} per move empty, {env.drain_loaded} per move loaded")
    print(f"Recharge        : {env.recharge_rate:.0f} units per CHARGE at the depot")
    print(f"Episode length  : {env.max_steps} steps")
    print(f"Observation dim : {env.obs_dim}")
    print(f"Actions         : {', '.join(DroneDeliveryEnv.ACTIONS)}")
    print()

    env, agent, history = train(episodes=1200, seed=0, eval_every=100)

    ret, dlv, crashes = evaluate(env, agent, episodes=20)
    print("\nFinal greedy evaluation over 20 held-out seeds")
    print(f"  mean return     : {ret:.2f}")
    print(f"  mean deliveries : {dlv:.2f}")
    print(f"  crashes         : {crashes}/20")

    demo_episode(env, agent)

    torch.save(agent.q.state_dict(), "drone_dqn.pt")
    print("\nSaved trained weights to drone_dqn.pt")


if __name__ == "__main__":
    main()
