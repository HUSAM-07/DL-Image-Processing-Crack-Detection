import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import logging

# Setup logging
logging.basicConfig(
    filename='rl_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Custom Feature Extractor for normalized images
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class CrackEnv(gym.Env):
    """Custom Environment for Crack Propagation using Gymnasium"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, initial_image, target_image, image_size=(224, 224), render_mode=None):
        super().__init__()
        
        # Image processing
        self.image_size = image_size
        self.initial_image = self.preprocess_image(initial_image)
        self.target_image = self.preprocess_image(target_image)
        self.current_image = self.initial_image.copy()
        self.render_mode = render_mode
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(8,), dtype=np.float32
        )
        
        # Modified observation space for normalized images
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(1, *image_size),
            dtype=np.float32
        )
        
        # Metrics
        self.max_steps = 100
        self.current_step = 0
        self.best_similarity = -float('inf')
        
        # Window for rendering
        self.window = None
        self.clock = None

    def preprocess_image(self, image):
        """Preprocess image to normalized grayscale"""
        if isinstance(image, str):
            image = Image.open(image).convert('L')
        image = image.resize(self.image_size)
        return np.array(image).astype(np.float32) / 255.0

    def apply_action(self, action):
        """Apply transformation based on action parameters"""
        # Extract transformation parameters
        brightness, contrast = action[0:2]
        angle = action[2] * 30  # Rotation angle
        scale = 1 + action[3] * 0.2  # Scale factor
        tx, ty = action[4:6] * 10  # Translation
        blur = abs(action[6])  # Blur factor
        threshold = (action[7] + 1) / 2  # Threshold value
        
        # Apply transformations
        image = self.current_image.copy()
        
        # Adjust brightness and contrast
        image = np.clip(image * (1 + contrast) + brightness, 0, 1)
        
        # Apply geometric transformations
        rows, cols = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        image = cv2.warpAffine(image, M, (cols, rows))
        
        # Apply blur and threshold
        if blur > 0:
            kernel_size = int(blur * 5) * 2 + 1
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        image = np.where(image > threshold, 1, 0)
        
        return image

    def compute_reward(self):
        """Compute reward based on similarity and intrinsic motivation"""
        # Structural similarity
        similarity = ssim(self.current_image, self.target_image)
        
        # Intrinsic motivation (novelty)
        edge_current = cv2.Canny((self.current_image * 255).astype(np.uint8), 100, 200)
        edge_target = cv2.Canny((self.target_image * 255).astype(np.uint8), 100, 200)
        edge_similarity = ssim(edge_current, edge_target)
        
        # Combined reward
        reward = similarity * 0.7 + edge_similarity * 0.3
        
        # Bonus for improvement
        if reward > self.best_similarity:
            reward += 0.1
            self.best_similarity = reward
            
        return reward

    def step(self, action):
        """
        Apply action and return new state, reward, terminated, truncated, and info
        """
        self.current_step += 1
        
        # Apply action and get new state
        self.current_image = self.apply_action(action)
        
        # Calculate reward
        reward = self.compute_reward()
        
        # Check if episode is terminated or truncated
        terminated = reward > 0.95  # Success condition
        truncated = self.current_step >= self.max_steps  # Time limit
        
        # Additional info
        info = {
            'similarity': reward,
            'step': self.current_step
        }
        
        if self.render_mode == "human":
            self._render_frame()
        
        return (
            self.current_image.reshape(1, *self.image_size),
            reward,
            terminated,
            truncated,
            info
        )

    def reset(self, seed=None, options=None):
        """Reset environment to initial state with Gymnasium API"""
        super().reset(seed=seed)  # Seed the random number generator
        
        self.current_image = self.initial_image.copy()
        self.current_step = 0
        self.best_similarity = -float('inf')
        
        if self.render_mode == "human":
            self._render_frame()
        
        return (
            self.current_image.reshape(1, *self.image_size),
            {"initial_state": True}  # Info dict
        )

    def render(self):
        """Render the environment"""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Helper function for rendering"""
        if self.render_mode == "human":
            cv2.imshow("Crack Propagation", self.current_image)
            cv2.waitKey(1)
        
        return (self.current_image * 255).astype(np.uint8)

    def close(self):
        """Clean up resources"""
        if self.window is not None:
            cv2.destroyAllWindows()
            self.window = None

def create_env(initial_image, target_image):
    """Create vectorized environment"""
    def _init():
        env = CrackEnv(initial_image, target_image)
        return env
    return DummyVecEnv([_init])

def train_rl_model(dataset_path, save_path='rl_crack_model'):
    """Train the RL model on the crack propagation dataset"""
    logging.info("Starting RL model training...")
    
    # Get image pairs
    image_pairs = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('first_frame.png'):
            number = filename.split('_')[0]
            first_frame = filename
            last_frame = f"{number}_last_frame.png"
            if os.path.exists(os.path.join(dataset_path, last_frame)):
                image_pairs.append((first_frame, last_frame))
    
    if not image_pairs:
        logging.error("No image pairs found in dataset!")
        return None
    
    logging.info(f"Found {len(image_pairs)} image pairs")
    
    # Create environment with first pair
    first_pair = image_pairs[0]
    env = create_env(
        os.path.join(dataset_path, first_pair[0]),
        os.path.join(dataset_path, first_pair[1])
    )
    
    # Initialize PPO model with custom policy
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False  # Images are already normalized
    )
    
    model = PPO(
        "MlpPolicy",  # Changed from CnnPolicy to MlpPolicy
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./rl_tensorboard/"
    )
    
    # Training loop with curriculum learning
    total_timesteps = 100000
    logging.info(f"Training for {total_timesteps} timesteps...")
    
    try:
        # Curriculum learning loop
        for difficulty in range(len(image_pairs)):
            current_pair = image_pairs[difficulty]
            env = create_env(
                os.path.join(dataset_path, current_pair[0]),
                os.path.join(dataset_path, current_pair[1])
            )
            model.set_env(env)
            
            # Train on current difficulty
            logging.info(f"Training on image pair {difficulty + 1}/{len(image_pairs)}")
            model.learn(
                total_timesteps=total_timesteps // len(image_pairs),
                reset_num_timesteps=False,
                tb_log_name=f"training_stage_{difficulty}"
            )
            
            # Evaluate current performance
            eval_env = create_env(
                os.path.join(dataset_path, current_pair[0]),
                os.path.join(dataset_path, current_pair[1])
            )
            mean_reward = 0
            n_eval_episodes = 5
            
            for _ in range(n_eval_episodes):
                obs = eval_env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs)
                    obs, reward, done, _ = eval_env.step(action)
                    mean_reward += reward
            
            mean_reward /= n_eval_episodes
            logging.info(f"Mean reward on difficulty {difficulty}: {mean_reward}")
        
        # Save the final model
        model.save(save_path)
        logging.info(f"Model saved to {save_path}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        return None
    
    return model

def predict_crack_propagation(model_path, initial_image_path):
    """Use trained model to predict crack propagation"""
    try:
        # Load model
        model = PPO.load(model_path)
        
        # Create environment
        env = CrackEnv(initial_image_path, initial_image_path)
        obs = env.reset()
        
        # Generate prediction
        done = False
        total_reward = 0
        max_steps = 100
        step = 0
        
        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            step += 1
            
            # Store best prediction
            if reward > env.best_similarity:
                best_prediction = env.current_image.copy()
        
        logging.info(f"Prediction completed with total reward: {total_reward}")
        return best_prediction if 'best_prediction' in locals() else env.current_image
        
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return None

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated_frames")
    model = train_rl_model(dataset_path)
    
    if model:
        logging.info("Training completed successfully!")
        
        # Test prediction
        test_image = os.path.join(dataset_path, "1_first_frame.png")
        if os.path.exists(test_image):
            prediction = predict_crack_propagation("rl_crack_model", test_image)
            if prediction is not None:
                cv2.imwrite("predicted_crack.png", prediction * 255)
                logging.info("Prediction saved as 'predicted_crack.png'")
