Core Projects Utilized:
- Stable Baselines3
- Gymnasium
- PyTorch
- TensorFlow
- Donkey Kong Environment

For those unfamiliar, the objective of Nintendo’s original Donkey Kong is to navigate the player through a series of platforms to rescue the princess at the top most platform. All the while, an ape-like creature throws barrels down the platforms towards the player, which carry with them a good deal of random movements to thwart the player’s progress towards the top. The player must combine careful timing, avoidance of obstacles, and navigation to have success.

The game is simple enough to define and understand for reinforcement learning algorithms, but challenging enough to test the limits of current technologies. Modern reinforcement learning algorithms work by defining a set of parameters from the environment, along with rewards for actions made by the agent in the environment. These definitions for rewards are key for agent success in the environment, as they ultimately establish the player’s goal. For this environment, a number of rewards could be utilized, and I have certainly tried my share:

- Rewarding the player for getting closer to the princess
- Rewarding the player for navigating to greater Y-axis values
- Rewarding the player for climbing ladders
- Rewarding the player more for being on higher platforms
- Punishing the player for dying
- Punishing the player for going down a ladder
- Rewarding distance traveled from the starting point on each platform
- Rewarding attempting to climb complete ladders
- Punishing attempting to climb ladders that have been discovered broken
- Rewarding for staying grounded on a platform to limit the amount of jump attempts by the player
- Rewarding for points earned in game
  
Through the use of the above reward systems, along with a few others, the agent quickly learns to progress towards the bottommost complete ladder, where it quickly discovers its first enemy entities. The randomness of enemy entities makes this a challenging set of interactions for the player. The agent must decipher entity locations and velocities to determine the correct direction and time to jump over or avoid these obstacles. This learning takes a good deal of time, and demonstrates both the power of these algorithms and the overall cost of training them.

With many fewer enemies, and a more predictable environment, the agent succeeds at reaching the princess. Without removing these barriers, modern agents struggle to progress up the platforms, after early success on lower platforms. Perhaps with more training time, these agents can discover complete success.
