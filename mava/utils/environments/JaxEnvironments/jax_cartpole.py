import math
from typing import Dict, Tuple

import chex
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey
from dm_env import specs
from flax.struct import dataclass
from gym import spaces
from jax import jit, random
from PIL import Image, ImageDraw

from mava.types import OLT
from mava.utils.id_utils import EntityId

GRAVITY = 9.82
CART_MASS = 0.5
POLE_MASS = 0.5
POLE_LEN = 0.6
FRICTION = 0.1
FORCE_SCALING = 10.0
DELTA_T = 0.01
CART_X_LIMIT = 2.4

SCREEN_W = 600
SCREEN_H = 600
CART_W = 40
CART_H = 20
VIZ_SCALE = 100
WHEEL_RAD = 5


@chex.dataclass
class State:
    cartpole_data: jnp.ndarray
    step: int
    key: PRNGKey


class JaxCartPole:
    def __init__(self):
        self.gravity = GRAVITY
        self.masscart = CART_MASS
        self.masspole = POLE_MASS
        self.total_mass = POLE_MASS + CART_MASS
        self.length = POLE_LEN  # actually half the pole's length
        self.polemass_length = POLE_MASS * POLE_LEN
        self.force_mag = FORCE_SCALING
        self.tau = DELTA_T  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.step_limit = 500

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = CART_X_LIMIT
        self.random_limit = 0.05

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        self.high = jnp.array(
            [
                self.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                self.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ],
            dtype=jnp.float32,
        )
        self.action_space = spaces.Discrete(2)

    @property
    def possible_agents(self):
        return [EntityId(type=0, id=0)]

    def step(self, env_state: State, action):
        action = action[EntityId(type=0, id=0)]
        state, key = env_state.cartpole_data, env_state.key
        x, x_dot, theta, theta_dot = state
        force = self.force_mag * (2 * action - 1)
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        done = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta > self.theta_threshold_radians)
            | (theta < -self.theta_threshold_radians)
            | (env_state.step > self.step_limit)
        )

        reward = jnp.float32(1)

        cartpole_data = jnp.array([x, x_dot, theta, theta_dot])
        new_state = State(cartpole_data=cartpole_data, step=env_state.step + 1, key=key)

        step_type = jax.lax.cond(
            done, lambda: dm_env.StepType.LAST, lambda: dm_env.StepType.MID
        )
        discount = jax.lax.cond(done, lambda: 0.0, lambda: 1.0)

        timestep = dm_env.TimeStep(
            step_type,
            {EntityId(type=0, id=0): reward},
            {EntityId(type=0, id=0): discount},
            {
                EntityId(type=0, id=0): OLT(
                    self._get_obsv(new_state), jnp.ones(2), jnp.bool_(done)
                )
            },
        )

        return new_state, timestep, {}

    def _get_obsv(self, state: State):
        return state.cartpole_data

    def _reset(self, key):
        new_state = random.uniform(
            key, minval=-self.random_limit, maxval=self.random_limit, shape=(4,)
        )
        new_key = random.split(key)[0]
        new_state = State(cartpole_data=new_state, step=0, key=new_key)
        return new_state

    def reset(self, key):
        env_state = self._reset(key)

        timestep = dm_env.TimeStep(
            dm_env.StepType.FIRST,
            {EntityId(type=0, id=0): 0.0},
            {EntityId(type=0, id=0): 1.0},
            {
                EntityId(type=0, id=0): OLT(
                    self._get_obsv(env_state), jnp.ones(2), jnp.bool_(False)
                )
            },
        )

        return env_state, timestep, {}

    def observation_spec(self) -> Dict[str, OLT]:
        observation_specs = {}
        for agent in self.possible_agents:

            # Legals spec
            legals = jnp.ones(
                self.action_space.n,
                dtype=jnp.float32,
            )

            observation_specs[agent] = OLT(
                observation=jnp.ones(4, jnp.float32),
                legal_actions=legals,
                terminal=jnp.bool_(True),
            )
        return observation_specs

    def action_spec(self):
        action_specs = {}
        for agent in self.possible_agents:
            action_specs[agent] = specs.DiscreteArray(
                num_values=self.action_space.n, dtype=jnp.int32
            )
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        """Reward spec.

        Returns:
            Dict[str, specs.Array]: spec for rewards.
        """
        reward_specs = {}
        for agent in self.possible_agents:
            reward_specs[agent] = specs.Array((), jnp.float32)

        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        """Discount spec.

        Returns:
            Dict[str, specs.BoundedArray]: spec for discounts.
        """
        discount_specs = {}
        for agent in self.possible_agents:
            discount_specs[agent] = specs.BoundedArray(
                (), jnp.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        return {}

    def get_agent_mask(self, state, agent_info):
        return jnp.zeros(self.action_space.n)

    def get_possible_agents(self):
        return self.possible_agents

    def get_observation(self, env_state: State, agent_info):
        return env_state.cartpole_data

    @staticmethod
    def render(state: State, mode="rgb_array") -> Image:
        """Render a specified task."""
        img = Image.new("RGB", (SCREEN_W, SCREEN_H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        x, _, theta, _ = np.array(state.cartpole_data)
        cart_y = SCREEN_H // 2 + 100
        cart_x = x * VIZ_SCALE + SCREEN_W // 2
        # Draw the horizon.
        draw.line(
            (
                0,
                cart_y + CART_H // 2 + WHEEL_RAD,
                SCREEN_W,
                cart_y + CART_H // 2 + WHEEL_RAD,
            ),
            fill=(0, 0, 0),
            width=1,
        )
        # Draw the cart.
        draw.rectangle(
            (
                cart_x - CART_W // 2,
                cart_y - CART_H // 2,
                cart_x + CART_W // 2,
                cart_y + CART_H // 2,
            ),
            fill=(255, 0, 0),
            outline=(0, 0, 0),
        )
        # Draw the wheels.
        draw.ellipse(
            (
                cart_x - CART_W // 2 - WHEEL_RAD,
                cart_y + CART_H // 2 - WHEEL_RAD,
                cart_x - CART_W // 2 + WHEEL_RAD,
                cart_y + CART_H // 2 + WHEEL_RAD,
            ),
            fill=(220, 220, 220),
            outline=(0, 0, 0),
        )
        draw.ellipse(
            (
                cart_x + CART_W // 2 - WHEEL_RAD,
                cart_y + CART_H // 2 - WHEEL_RAD,
                cart_x + CART_W // 2 + WHEEL_RAD,
                cart_y + CART_H // 2 + WHEEL_RAD,
            ),
            fill=(220, 220, 220),
            outline=(0, 0, 0),
        )
        # Draw the pole.
        draw.line(
            (
                cart_x,
                cart_y,
                cart_x + POLE_LEN * VIZ_SCALE * np.cos(theta - np.pi / 2),
                cart_y + POLE_LEN * VIZ_SCALE * np.sin(theta - np.pi / 2),
            ),
            fill=(0, 0, 255),
            width=6,
        )
        return img
