import math
from typing import Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pymunk
from pymunk.vec2d import Vec2d


class DoublePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, reward_type: str = "shaped", render_mode: str = None):
        super().__init__()
        assert reward_type in ["baseline", "shaped"], "reward_type must be 'baseline' or 'shaped'"
        self.reward_type = reward_type

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 800
        self.screen_height = 600

        # Physics parameters
        self.dt = 1.0 / 60.0
        self.gravity = (0.0, -980.0)  # pixels/s^2
        self.track_y = 100

        # Create pymunk space
        self.space = pymunk.Space()
        self.space.gravity = self.gravity

        # Observation: [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
        high = np.array(
            [
                2.4,  # cart x
                np.finfo(np.float32).max,  # cart v
                np.pi,  # theta1
                np.finfo(np.float32).max,  # theta1_dot
                np.pi,  # theta2
                np.finfo(np.float32).max,  # theta2_dot
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Action: force on cart, scaled later
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Bodies / joints placeholders
        self.cart_body = None
        self.pole1_body = None
        self.pole2_body = None
        self.joints = []

        self.max_steps = 1000
        self.steps = 0

        self._setup_pygame_if_needed()
        self.reset()

    def _setup_pygame_if_needed(self):
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Double Inverted Pendulum - Pymunk")
            self.clock = pygame.time.Clock()

    def _clear_space(self):
        for shape in self.space.shapes[:]:
            self.space.remove(shape)
        for body in self.space.bodies[:]:
            self.space.remove(body)
        for constraint in self.space.constraints[:]:
            self.space.remove(constraint)
        self.joints = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._clear_space()

        # Cart
        mass_cart = 5.0
        width_cart, height_cart = 80.0, 30.0
        moment_cart = pymunk.moment_for_box(mass_cart, (width_cart, height_cart))
        self.cart_body = pymunk.Body(mass_cart, moment_cart)
        self.cart_body.position = (self.screen_width / 2, self.track_y)
        cart_shape = pymunk.Poly.create_box(self.cart_body, (width_cart, height_cart))
        cart_shape.friction = 0.5

        # Constrain cart movement along horizontal line (GrooveJoint with static body)
        static_body = self.space.static_body
        groove = pymunk.GrooveJoint(
            static_body,
            self.cart_body,
            (50, self.track_y),
            (self.screen_width - 50, self.track_y),
            (0, 0),
        )

        # Pole 1
        mass_pole1 = 1.0
        length_pole1 = 150.0
        moment_pole1 = pymunk.moment_for_segment(
            mass_pole1, (0, 0), (0, length_pole1), 5
        )
        self.pole1_body = pymunk.Body(mass_pole1, moment_pole1)
        self.pole1_body.position = self.cart_body.position + (0, height_cart / 2)
        pole1_shape = pymunk.Segment(
            self.pole1_body,
            (0, 0),
            (0, length_pole1),
            5,
        )
        pole1_shape.friction = 0.5

        # Pivot between cart and pole1 at top of cart
        pivot1 = pymunk.PinJoint(
            self.cart_body,
            self.pole1_body,
            (0, height_cart / 2),
            (0, 0),
        )

        # Pole 2
        mass_pole2 = 1.0
        length_pole2 = 120.0
        moment_pole2 = pymunk.moment_for_segment(
            mass_pole2, (0, 0), (0, length_pole2), 5
        )
        self.pole2_body = pymunk.Body(mass_pole2, moment_pole2)
        self.pole2_body.position = self.pole1_body.position + (0, length_pole1)
        pole2_shape = pymunk.Segment(
            self.pole2_body,
            (0, 0),
            (0, length_pole2),
            5,
        )
        pole2_shape.friction = 0.5

        # Pivot between pole1 and pole2 at end of pole1
        pivot2 = pymunk.PinJoint(
            self.pole1_body,
            self.pole2_body,
            (0, length_pole1),
            (0, 0),
        )

        # Add to space
        self.space.add(
            self.cart_body,
            cart_shape,
            self.pole1_body,
            pole1_shape,
            self.pole2_body,
            pole2_shape,
            groove,
            pivot1,
            pivot2,
        )
        self.joints = [groove, pivot1, pivot2]

        # Slight randomization around upright
        self.cart_body.velocity = (0, 0)
        self.pole1_body.angle = math.pi + self.np_random.uniform(-0.05, 0.05)
        self.pole2_body.angle = math.pi + self.np_random.uniform(-0.05, 0.05)
        self.pole1_body.angular_velocity = 0.0
        self.pole2_body.angular_velocity = 0.0

        self.steps = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self) -> np.ndarray:
        x = (self.cart_body.position.x - self.screen_width / 2) / (self.screen_width / 2)

        vx = self.cart_body.velocity.x / 1000.0

        theta1 = self._wrap_angle(self.pole1_body.angle - math.pi)
        theta2 = self._wrap_angle(self.pole2_body.angle - math.pi)
        omega1 = self.pole1_body.angular_velocity / 10.0
        omega2 = self.pole2_body.angular_velocity / 10.0

        return np.array([x, vx, theta1, omega1, theta2, omega2], dtype=np.float32)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        # wrap to [-pi, pi]
        return ((angle + math.pi) % (2 * math.pi)) - math.pi

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        _, _, theta1, omega1, theta2, omega2 = obs

        # baseline upright reward
        baseline = math.cos(theta1) + math.cos(theta2)

        if self.reward_type == "baseline":
            return baseline

        # shaped reward
        x = obs[0]
        center_penalty = -0.1 * abs(x)
        vel_penalty = -0.01 * (abs(omega1) + abs(omega2))
        action_penalty = -0.001 * float(action[0] ** 2)

        return baseline + center_penalty + vel_penalty + action_penalty

    def step(self, action: np.ndarray):
        # clip action to [-1, 1]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        force_mag = 2000.0
        force = float(action[0]) * force_mag

        # Apply horizontal force on cart
        self.cart_body.apply_force_at_local_point((force, 0.0), (0.0, 0.0))

        # Step physics
        self.space.step(self.dt)
        self.steps += 1

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)

        x = obs[0]
        theta1 = obs[2]
        theta2 = obs[4]

        terminated = False
        truncated = False

        # Episode ends when cart leaves range or poles fall
        if abs(x) > 1.2 or abs(theta1) > math.radians(60) or abs(theta2) > math.radians(60):
            terminated = True

        # Time limit
        if self.steps >= self.max_steps:
            truncated = True

        info: Dict[str, Any] = {"reward_type": self.reward_type}
        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human"):
        if self.screen is None or self.clock is None:
            self.render_mode = "human"
            self._setup_pygame_if_needed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((255, 255, 255))

        # Draw track
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (50, self.screen_height - self.track_y),
            (self.screen_width - 50, self.screen_height - self.track_y),
            4,
        )

        def to_pygame(p: Vec2d):
            return int(p.x), int(self.screen_height - p.y)

        # Cart
        cart_pos = self.cart_body.position
        width_cart, height_cart = 80.0, 30.0
        cart_rect = pygame.Rect(0, 0, width_cart, height_cart)
        cart_rect.center = to_pygame(cart_pos)
        pygame.draw.rect(self.screen, (0, 0, 255), cart_rect)

        # Pole1
        p1_start = self.pole1_body.position
        p1_end = self.pole1_body.position + Vec2d(0, 150).rotated(self.pole1_body.angle)
        pygame.draw.line(
            self.screen,
            (255, 0, 0),
            to_pygame(p1_start),
            to_pygame(p1_end),
            6,
        )

        # Pole2
        p2_start = p1_end
        p2_end = p2_start + Vec2d(0, 120).rotated(self.pole2_body.angle)
        pygame.draw.line(
            self.screen,
            (0, 150, 0),
            to_pygame(p2_start),
            to_pygame(p2_end),
            6,
        )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None