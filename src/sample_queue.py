import ray
import time
import jax
import jax.lax as lax
import jax.numpy as np
from functools import partial
from jax import random
import asyncio


@ray.remote(max_restarts=-1, max_task_retries=-1)
class MemorySampler:
    def __init__(self, queue, memory_actor, batch_size, rollout_size=5, n_step=10, discount_rate=0.995) -> None:
        self.queue = queue
        self.memory_actor = memory_actor
        self.batch_size = batch_size
        self.rollout_size = rollout_size
        self.n_step = n_step
        self.discount_rate = discount_rate
        self.memory_batches = 8

    async def run_queue(self, key):
        training_device_count = 8
        game_count = 0
        cpu = jax.devices("cpu")[0]
        device = jax.devices()[0]
        item_count = await self.memory_actor.item_count.remote()
        while item_count < self.batch_size:
            if item_count != game_count:
                # print("GAMES", item_count)
                game_count = item_count
            await asyncio.sleep(30)
            item_count = await self.memory_actor.item_count.remote()

        while True:
            # print("SAMPLE")
            with jax.default_device(cpu):
                # import code; code.interact(local=dict(globals(), **locals()))
                full_data = None
                game_indices = None
                game_keys = None
                key, full_data, game_keys = await self.fetch_games(
                    jax.device_put(key, cpu), self.batch_size)

                # (observations, actions, values, policies, rewards,
                #  priorities, game_indices, choices) = data
                # data = (np.stack(observations), np.stack(actions), np.stack(values), np.stack(
                #     policies), np.stack(rewards), priorities, game_indices, choices)
                memories = memory_sample(jax.device_put(
                    full_data, device), self.rollout_size, self.n_step, self.discount_rate)
                observations = np.reshape(memories["observations"], (training_device_count, int(
                    self.batch_size / training_device_count), 32, 96, 96, 3))
                actions = np.reshape(memories["actions"], (training_device_count, int(
                    self.batch_size / training_device_count), 6, 32))
                policies = np.reshape(memories["policies"], (training_device_count, int(
                    self.batch_size / training_device_count), 6, 18))
                values = np.reshape(memories["values"], (training_device_count, int(
                    self.batch_size / training_device_count), 6))
                rewards = np.reshape(memories["rewards"], (training_device_count, int(
                    self.batch_size / training_device_count), 6))
                game_indices = np.array(memories["game_indices"])
                step_indices = np.array(memories["step_indices"])
                priorities = np.reshape(memories["priority"], (training_device_count, int(
                    self.batch_size / training_device_count)))
            await self.queue.put_async((observations, actions, policies, values,
                                        rewards, game_indices, step_indices, priorities))

    async def fetch_games(self, key, n):
        relevant_priorities = await self.memory_actor.get_priorities.remote()
        relevant_priorities = np.array(relevant_priorities)
        length = len(relevant_priorities)
        if relevant_priorities == None:
            relevant_priorities = np.ones(length)
        else:
            relevant_priorities = relevant_priorities.at[relevant_priorities == 0].set(
                relevant_priorities[relevant_priorities == 0] + 1)
        key, subkey = random.split(key)
        print("PRIORITIES SHAPE", relevant_priorities.shape)
        game_indices = random.choice(subkey, np.array(range(length)), replace=False, shape=(
            1, n), p=relevant_priorities).squeeze()
        # print("GAME INDICES", game_indices)
        keys = random.split(key, num=n + 1)
        key = keys[0]
        keys = keys[1:]

        observations = []
        actions = []
        values = []
        policies = []
        rewards = []

        full_priorities = []
        full_observations = []
        full_actions = []
        full_values = []
        full_policies = []
        full_rewards = []
        number_per_batch = int(len(game_indices) / self.memory_batches)
        for i in range(self.memory_batches):
            split_start = number_per_batch * i
            split_end = number_per_batch * i + number_per_batch
            # print("FETCHING INDICES", game_indices[split_start: split_end])
            (priorities, observations, actions, values, policies, rewards) = await self.memory_actor.get_games.remote(game_indices[split_start: split_end])
            # print("PRIOR", priorities)
            full_priorities += priorities
            full_observations += observations
            full_actions += actions
            full_values += values
            full_policies += policies
            full_rewards += rewards

        priorities = np.array(full_priorities)
        # for i in game_indices:
        #     # TODO deal with reset memory here
        #     game = await self.memory_actor.get_games.remote(i)
        #     priorities.append(game.priorities)
        #     observations.append(game.observations)
        #     actions.append(game.actions)
        #     values.append(game.values)
        #     policies.append(game.policies)
        #     rewards.append(game.rewards)
        choices, priorities = jax.vmap(self.choice, (0, 0))(
            priorities, keys[0: self.batch_size])
        observations = np.array(full_observations)
        actions = np.array(full_actions)
        values = np.array(full_values)
        policies = np.array(full_policies)
        rewards = np.array(full_rewards)

        return key, (observations, actions, values, policies, rewards, np.array(priorities), game_indices, np.array(choices)), keys

    def choice(self, priorities, key):
        available_indices = list(range(0, priorities.shape[0]))
        starting_index = 32
        available_indices = np.stack(
            available_indices[starting_index: -(self.rollout_size + self.n_step)])
        priorities = np.stack(
            priorities[starting_index: -(self.rollout_size + self.n_step)])
        priorities = np.where(priorities == 0, 1, priorities)
        sum = np.sum(priorities)
        # TODO check paper to understand why this is happening
        priorities = lax.cond(np.all(sum == 0), lambda: priorities +
                              (1 / priorities.shape[0]), lambda: priorities / sum)
        index = random.choice(key, available_indices, p=priorities).squeeze()
        return index, priorities


@ partial(jax.jit, static_argnums=(1, 2, 3,))
def memory_sample(data, rollout_size, n_step, discount_rate):
    (observations, actions, values, policies, rewards,
        priorities, game_indices, choices) = data
    game_sample = jax.vmap(
        sample_from_game, (0, 0, 0, 0, 0, 0, 0, None, None, None))
    observations, actions, rewards, values, policies, _, step_indices = game_sample(
        observations,
        actions,
        values,
        policies,
        rewards,
        game_indices,
        choices,
        rollout_size,
        n_step,
        discount_rate
    )

    priority_result = jax.vmap((lambda i, j: priorities[i][j]), (0, 0))(
        game_indices, step_indices)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "values": values,
        "policies": policies,
        "step_indices": step_indices,
        "game_indices": np.stack(game_indices),
        "priority": np.stack(priority_result),
    }

# TODO move to fetching out of bounds blank states on sample


@ partial(jax.jit, static_argnums=(7, 8, 9))
def sample_from_game(observations, actions, values, policies, rewards, game_index, step_index, rollout_size, n_step, discount_rate):
    k_step_actions = []
    k_step_rewards = []
    k_step_values = []
    k_step_policies = []
    for k_step in range(rollout_size + 1):
        k_step_actions.append(lax.dynamic_slice_in_dim(
            actions, step_index - 32 + k_step, 32))
        k_step_rewards.append(rewards[step_index + k_step])
        _, k_step_value, _, _, _ = lax.fori_loop(0, n_step - 1, compute_nstep_value, (step_index + k_step,
                                                                                      rewards[step_index + k_step] - values[step_index + k_step], rewards, n_step, discount_rate))
        k_step_values.append(k_step_value)
        k_step_policies.append(policies[step_index + k_step])

    observations = lax.dynamic_slice_in_dim(
        observations, step_index - 32, 32)
    return (np.array(observations), np.stack(k_step_actions), np.array(k_step_rewards), np.array(k_step_values), np.array(k_step_policies), game_index, step_index)


def compute_nstep_value(i, data):
    (starting_index, value, rewards, n_step, discount_rate) = data

    def update_value(): return value + \
        rewards[starting_index + n_step + 1] * discount_rate ** n_step
    value = lax.cond(starting_index + n_step + 1 <
                     rewards.shape[0], update_value, lambda: value)
    return (starting_index, value, rewards, n_step, discount_rate)
