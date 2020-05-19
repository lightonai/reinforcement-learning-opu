"""
Fork of https://github.com/astier/model-free-episodic-control (MIT-Licensed)
"""

import random
import argparse
import time
import matplotlib.pyplot as plt
import os
import shutil

import gym

from mfec.agent import MFECAgent
from mfec.utils import Utils

parser = argparse.ArgumentParser()
parser.add_argument('--knn', type=int, default=9,
                    help="Number of nearest neighbors taken into account (default: 9)")
parser.add_argument('--dim', type=int, default=32,
                    help="Number of random features extracted by the OPU (default: 32)")
parser.add_argument('--discount', type=float, default=0.99,
                    help="Discount rate for the return (default: 0.99)")
parser.add_argument('--epsilon', type=float, default=0.001,
                    help="Percentage of random exploration (default: 0.001)")
parser.add_argument('--env', type=str, default="MsPacman-v0",
                    help="Game to play (more games at https://gym.openai.com/envs/#atari)")
parser.add_argument('-v', '--volatile', action="store_true",
                    help="Prevent frames from being recorded")
parser.add_argument('-s', '--save', action="store_true",
                    help="Save the trained agent for later use")
args = parser.parse_args()

ENVIRONMENT = args.env
RENDER = not args.volatile
SAVE = args.save

EPOCHS = 4
FRAMES_PER_EPOCH = 100000

ACTION_BUFFER_SIZE = 200000  # Number of states that can be stored for each action
K = args.knn

DISCOUNT = args.discount
EPSILON = args.epsilon

FRAMESKIP = 3  # Default gym-setting is (2, 5), see notes in the README
REPEAT_ACTION_PROB = 0.0  # Default gym-setting is .25

SCALE_DIMS = None  #(58, 40)  # Dimensions to rescale the inputs to, None means no rescaling
STATE_DIMENSION = args.dim


def main():
    """Learns to play ENVIRONMENT. Initializes the environment and the agent.

    """
    random.seed(None)

    # Creates folder to store some of the frames
    try:
        shutil.rmtree("videos")
    except FileNotFoundError:
        pass
    os.mkdir("videos")

    # Initialize utils, environment and agent
    utils = Utils(FRAMES_PER_EPOCH, EPOCHS * FRAMES_PER_EPOCH)
    env = gym.make(ENVIRONMENT)

    try:
        env.frameskip = FRAMESKIP
        env.ale.setFloat("repeat_action_probability", REPEAT_ACTION_PROB)
        agent = MFECAgent(
            ACTION_BUFFER_SIZE,
            K,
            DISCOUNT,
            EPSILON,
            SCALE_DIMS,
            STATE_DIMENSION,
            range(env.action_space.n)
        )

        run_algorithm(agent, env, utils)

        exploit_score = exploit(agent, env)

        if SAVE:
            import pickle
            with open("agent.pkl", 'wb') as file:
                pickle.dump(agent, file, 2)

        if RENDER:  # Creates the video of the best recorded run
            # https://askubuntu.com/questions/610903/how-can-i-create-a-video-file-from-a-set-of-jpg-images
            if bestrun[2] > exploit_score:
                os.system(
                    'ffmpeg -framerate 25 -i videos/{}-{}-%00000d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p bestrun.mp4'.format(
                        bestrun[0], bestrun[1]))
            else:
                os.system(
                    'ffmpeg -framerate 25 -i videos/exploit-%00000d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p bestrun.mp4')

            print('\n\nBest:', max(bestrun[2], exploit_score), '\n')

    finally:
        utils.close()
        env.close()


def run_algorithm(agent, env, utils):
    """Runs the algorithm.

    """
    global epi, epo, bestrun  # Variables used to store the frames
    frames_left = 0
    successive_wins = 0
    for _ in range(EPOCHS):
        frames_left += FRAMES_PER_EPOCH
        epi = 1
        while frames_left > 0:
            # if os.path.exists('terminate_cron.txt'):  # Workaround to terminate the process at a certain time
            #     os.remove('terminate_cron.txt')
            #     successive_wins = 6
            #     break
            episode_frames, episode_reward = run_episode(agent, env)
            if (epi == 1 or epi % 10 == 0) and episode_reward > bestrun[2]:
                bestrun = (epo, epi, episode_reward)
            epi += 1
            frames_left -= episode_frames
            utils.end_episode(episode_frames, episode_reward)  # Console display

            if episode_reward > threshold:  # This is a means of terminating the script before all epochs are completed
                successive_wins += 1
            else:
                successive_wins = 0
            if successive_wins > 5:
                break
        if successive_wins > 5:
            print('Solved!\n\n')
            break
        utils.end_epoch()  # Console display
        epo += 1


def run_episode(agent, env):
    """Finds the right action depending on the observed state of the ENVIRONMENT and sends
    it to the ENVIRONMENT until the ENVIRONMENT returns a 'done' signal.

    """
    episode_frames = 0
    episode_reward = 0

    env.seed(random.randint(0, 1000000))
    observation = env.reset()
    max_lives = env.ale.lives()

    done = False
    frame = 0  # Used if RENDER is True
    while not done:  # While not game over

        # if RENDER:  # Live display
        #     env.render()
        #     time.sleep(RENDER_SPEED)
        if RENDER and (epi == 1 or epi % 10 == 0):
            plt.imsave(os.path.join('videos', '-'.join((str(epo), str(epi), str(frame))) + '.png'),
                       env.render(mode='rgb_array'))

        action = agent.choose_action(observation)
        observation, reward, done, info = env.step(action)
        if info['ale.lives'] < max_lives:  # No revive
            done = True
        frame += 1
        agent.receive_reward(reward)

        episode_reward += reward
        episode_frames += FRAMESKIP

    agent.train()
    return episode_frames, episode_reward


def exploit(agent, env):
    """Same as run_episode but EPSILON is considered to be 0, i.e. there is no exploration.

    """
    episode_reward = 0
    env.seed(random.randint(0, 1000000))
    observation = env.reset()
    max_lives = env.ale.lives()

    frame = 0
    done = False
    while not done:
        if RENDER:
            plt.imsave(os.path.join('videos', '-'.join(('exploit', str(frame))) + '.png'), env.render(mode='rgb_array'))

        action = agent.choose_action(observation, explore=False)
        observation, reward, done, info = env.step(action)
        if info['ale.lives'] < max_lives:  # No revive
            done = True
        frame += 1

        episode_reward += reward

    print('\nExploitation run: score', episode_reward, '\n')
    return episode_reward


if __name__ == "__main__":
    threshold = 10000  # If the score of an episode reaches this threshold 5 times in a row, the scipts stops.
    # Next three variables are used if RENDER is True
    epo = 1
    epi = 1
    bestrun = (1, 1, 0)

    main()
