import ray
import time
import jax
import jax.lax as lax
import jax.numpy as np
import numpy as onp
from functools import partial
from jax import random
import asyncio


@ray.remote(max_restarts=-1, max_task_retries=-1, resources={"TPU_VM_CPU": 0.5})
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

        with jax.default_device(cpu):
            item_count = await self.memory_actor.item_count.remote()
            while item_count < self.batch_size:
                if item_count != game_count:
                    # print("GAMES", item_count)
                    game_count = item_count
                await asyncio.sleep(30)
                item_count = await self.memory_actor.item_count.remote()
            count = 0
            while True:
                print("COUNT", count)
                # print("SAMPLE")
                # import code; code.interact(local=dict(globals(), **locals()))
                full_data = None
                game_indices = None
                game_keys = None
                print("BEFORE FETCH")
                key, full_data, game_keys = await self.fetch_games(
                    jax.device_put(key, cpu), self.batch_size)
                print("AFTER FETCH")

                # # (observations, actions, values, policies, rewards,
                # #  priorities, game_indices, choices) = data
                # # data = (np.stack(observations), np.stack(actions), np.stack(values), np.stack(
                # #     policies), np.stack(rewards), priorities, game_indices, choices)
                # memories = memory_sample(
                #     full_data, self.rollout_size, self.n_step, self.discount_rate)
                # observations = np.reshape(memories["observations"], (training_device_count, int(
                #     self.batch_size / training_device_count), 32, 96, 96, 3))
                # actions = np.reshape(memories["actions"], (training_device_count, int(
                #     self.batch_size / training_device_count), 6, 32))
                # policies = np.reshape(memories["policies"], (training_device_count, int(
                #     self.batch_size / training_device_count), 6, 18))
                # values = np.reshape(memories["values"], (training_device_count, int(
                #     self.batch_size / training_device_count), 6))
                # rewards = np.reshape(memories["rewards"], (training_device_count, int(
                #     self.batch_size / training_device_count), 6))
                # game_indices = np.array(memories["game_indices"])
                # step_indices = np.array(memories["step_indices"])
                # priorities = np.reshape(memories["priority"], (training_device_count, int(
                #     self.batch_size / training_device_count)))
                await self.queue.put_async(full_data)
                # print("SAMPLE QUEUE AMOUNT", self.queue.qsize())
                count += 1

    async def fetch_games(self, key, n):
        cpu = jax.devices("cpu")[0]

        with jax.default_device(cpu):
            relevant_priorities = await self.memory_actor.get_priorities.remote()
            length = len(relevant_priorities)
            relevant_priorities = np.array(relevant_priorities)
            if relevant_priorities == None:
                relevant_priorities = np.ones(length)
            else:
                relevant_priorities = relevant_priorities.at[relevant_priorities == 0].set(
                    relevant_priorities[relevant_priorities == 0] + 1)
            key, subkey = random.split(key)
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
            game_generator = self.memory_actor.get_games.remote(game_indices)
            for item in game_generator:
                data = await item
                (priorities, observations, actions,
                 values, policies, rewards) = jax.device_put(data, cpu)
                full_priorities.append(priorities)
                full_observations.append(observations)
                full_actions.append(actions)
                full_values.append(values)
                full_policies.append(policies)
                full_rewards.append(rewards)

            # for i in range(self.memory_batches):
            #     split_start = number_per_batch * i
            #     split_end = number_per_batch * i + number_per_batch
            #     # print("FETCHING INDICES", game_indices[split_start: split_end])
            #     (priorities, observations, actions, values, policies, rewards) = await self.memory_actor.get_games.remote(game_indices[split_start: split_end])
            #     # print("PRIOR", priorities)
            #     full_priorities += priorities
            #     full_observations += observations
            #     full_actions += actions
            #     full_values += values
            #     full_policies += policies
            #     full_rewards += rewards

            priorities = onp.array(full_priorities)
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
            observations = onp.array(full_observations)
            actions = onp.array(full_actions)
            values = onp.array(full_values)
            policies = onp.array(full_policies)
            rewards = onp.array(full_rewards)

            return key, (observations, actions, values, policies, rewards, onp.array(priorities), game_indices, onp.array(choices)), keys

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


@ray.remote(max_restarts=-1, max_task_retries=-1, resources={"TPU_VM_CPU": 0.5})
class RolloutWorker:
    def __init__(self, context_queue, rollout_queue, batch_size, rollout_size=5, n_step=10, discount_rate=0.995):
        self.context_queue = context_queue
        self.batch_size = batch_size
        self.rollout_size = rollout_size
        self.n_step = n_step
        self.discount_rate = discount_rate
        self.memory_batches = 8
        self.rollout_queue = rollout_queue

    async def rollout(self):
        cpu = jax.devices("cpu")[0]

        with jax.default_device(cpu):

            training_device_count = 8
            while True:
                full_data = await self.context_queue.get_async()
                memories = memory_sample(
                    full_data, self.rollout_size, self.n_step, self.discount_rate)
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
                data = (onp.array(observations), onp.array(actions), onp.array(policies), onp.array(values),
                        onp.array(rewards), onp.array(game_indices), onp.array(step_indices), onp.array(priorities))
                # data = jax.device_put(data, cpu)
                await self.rollout_queue.put_async(data)

                # print("ROLLOUT QUEUE AMOUNT", self.rollout_queue.qsize())


@partial(jax.jit, static_argnums=(1, 2, 3,))
def memory_sample(data, rollout_size, n_step, discount_rate):
    (observations, actions, values, policies, rewards,
        priorities, game_indices, choices) = data

    observations = np.array(observations)
    actions = np.array(actions)
    values = np.array(values)
    policies = np.array(policies)
    rewards = np.array(rewards)
    priorities = np.array(priorities)
    game_indices = np.array(game_indices)
    choices = np.array(choices)
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
