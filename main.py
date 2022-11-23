import argparse
import os
import sys
import logging
import numpy as np
from collections import deque

import rospy
from mpi4py import MPI
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import ppoAgent
from stage_world1 import StageWorld


parser = argparse.ArgumentParser(description='Point-Navi RL')
parser.add_argument('--id', default="train_PPO", type=str)
parser.add_argument('--laser_beam', default=512, type=int, metavar='N', help='number of laser beams')
parser.add_argument('-actor_lr', default=5e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-critic_lr', default=5e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lmbda', default=0.95, type=float, help='lambda for calculate advantage')
parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to update')
parser.add_argument('--eps', default=0.1, type=float, help='clip value')
parser.add_argument('--gamma', default=0.99, type=float, help='learning rate gamma')
parser.add_argument('--max_episodes', default=50000, type=int, metavar='N', help='the max episodes of training')
parser.add_argument('--horizon', default=128, type=int, metavar='N', help='the max steps the agent can take')
parser.add_argument('--test', default=0, type=int, metavar='N', help='1 -> True')
parser.add_argument('--train', default=1, type=int, metavar='N', help='1 ->True')
parser.add_argument('--resume', default=0, type=int, metavar='N', help='1 ->True')


def main():
    global args
    args = parser.parse_args()

    # log configuration
    log_dir = os.path.join("./logs", args.id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + '/run.log'
    logger = logging.getLogger('run_logger')
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    file_hander = logging.FileHandler(log_file, mode='a')
    file_hander.setLevel(logging.INFO)
    file_hander.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(stdout_handler)
    logger.addHandler(file_hander)

    # model saving dir
    save_model_dir = os.path.join("./save_models", args.id)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # tensorboard setting
    return_dir = os.path.join("./returns", args.id)
    tsbd = SummaryWriter(log_dir=return_dir)
    loss_dir = os.path.join("./losses", args.id)
    loss_tsbd = SummaryWriter(log_dir=loss_dir)

    # mpi4py setting
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # initialize the environment
    env = StageWorld(args.laser_beam, index=rank, num_env=1)
    reward = None

    # setting of cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    robot = ppoAgent(actor_lr=args.actor_lr, critic_lr=args.critic_lr, lmbda=args.lmbda, epochs=args.epochs,
                     eps=args.eps, gamma=args.gamma, device=device, loss_tensorboard=loss_tsbd)

    if args.train == 1:
        train_agent(env, robot, args.max_episodes, args.horizon, save_model_dir, tsbd, logger)
    if args.resume == 1:
        resume_agent(env, robot, args.max_episodes, save_model_dir, tsbd, logger)
    if args.test == 1:
        test_agent(env, robot, args.max_episodes, save_model_dir, tsbd, logger)


def train_agent(env, agent, num_episodes, horizon, model_path, writer, logger):
    print ('Go Training !!!')
    transition_dict = {'scans': [], 'goals': [], 'speeds': [], 'actions': [], 'next_scans': [],
                       'next_goals': [], 'next_speeds': [], 'rewards': [], 'dones': []}
    global_step = 0
    global_update = 0

    # reset the environment
    if env.index == 0:
        env.reset_world()

    for episode in range(num_episodes):
        episode_return = 0
        step = 1
        done = False

        env.reset_pose()
        env.generate_goal_point()

        obs = env.get_laser_observation()           # (512,)
        obs_stack = deque([obs, obs, obs])
        scan = obs_stack
        goal = np.asarray(env.get_local_goal())     # (2,)
        speed = np.asarray(env.get_self_speed())    # (2,)

        while not done and not rospy.is_shutdown():
            action, scaled_action = agent.take_action(scan=scan, goal=goal, speed=speed)
            env.control_vel(scaled_action)
            rospy.sleep(0.001)

            reward, done, result = env.get_reward_and_terminate(step)

            episode_return += reward
            global_step += 1

            # get next state
            obs_next = env.get_laser_observation()
            left = obs_stack.popleft()
            obs_stack.append(obs_next)
            scan_next = obs_stack
            goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            if env.index == 0:
                transition_dict['scans'].append(scan)
                transition_dict['goals'].append(goal)
                transition_dict['speeds'].append(speed)
                transition_dict['actions'].append(action)
                transition_dict['next_scans'].append(scan_next)
                transition_dict['next_goals'].append(goal_next)
                transition_dict['next_speeds'].append(speed_next)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                if len(transition_dict['rewards']) > horizon * 2:
                    logger.info('Start updating !!!')
                    agent.update(transition_dict, count=global_update+1)
                    logger.info('Updating done !!!')
                    transition_dict = {'scans': [], 'goals': [], 'speeds': [], 'actions': [], 'next_scans': [],
                                       'next_goals': [], 'next_speeds': [], 'rewards': [], 'dones': []}
                    global_update += 1
            step += 1
            scan = scan_next
            goal = goal_next
            speed = speed_next

        # save the model
        if env.index == 0:
            if global_update != 0 and global_step % 20 == 0:
                torch.save(agent.actor.state_dict(), model_path + 'train_actor{}'.format(global_update/20))
                torch.save(agent.critic.state_dict(), model_path + 'train_critic{}'.format(global_update/20))
        logger.info('Episode %05d, step %03d, Goal (%05.1f, %05.1f), Return %-5.1f, %s'
                    % (episode + 1, step, env.goal_point[0], env.goal_point[1], episode_return, result))
        writer.add_scalar('train/return', episode_return, episode + 1)


def resume_agent(env, agent, num_episodes,  model_path, writer, logger):
    actor_dict = torch.load(model_path + 'train_actor{}'.format(1))
    critic_dict = torch.load(model_path + 'train_critic{}'.format(1))
    agent.actor.load_state_dict(actor_dict)
    agent.critic.load_state_dict(critic_dict)
    print ('Go Resuming !!!')
    transition_dict = {'scans': [], 'points': [], 'goals': [], 'speeds': [], 'actions': [], 'next_scans': [],
                       'next_points': [], 'next_goals': [], 'next_speeds': [], 'rewards': [], 'dones': []}
    global_step = 0
    global_update = 100

    # reset the environment
    if env.index == 0:
        env.reset_world()

    for episode in range(num_episodes):
        episode_return = 0
        step = 1
        done = False

        env.reset_pose()
        env.generate_goal_point()

        obs = env.get_laser_observation()           # (512,)
        pos = to_point(obs)                         # (512, 2)
        obs_stack = deque([obs, obs])
        scan = obs_stack
        goal = np.asarray(env.get_local_goal())     # (2,)
        speed = np.asarray(env.get_self_speed())    # (2,)

        while not done and not rospy.is_shutdown():
            action, scaled_action = agent.take_action(scan=scan, point=pos, goal=goal, speed=speed)
            env.control_vel(scaled_action)
            rospy.sleep(0.001)

            reward, done, result = env.get_reward_and_terminate(step)

            episode_return += reward
            global_step += 1

            # get next state
            obs_next = env.get_laser_observation()
            pos_next = to_point(obs_next)
            left = obs_stack.popleft()
            obs_stack.append(obs_next)
            scan_next = obs_stack
            goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            if env.index == 0:
                transition_dict['scans'].append(scan)
                transition_dict['points'].append(pos)
                transition_dict['goals'].append([goal])
                transition_dict['speeds'].append([speed])
                transition_dict['actions'].append(action)
                transition_dict['next_scans'].append(scan_next)
                transition_dict['next_points'].append(pos_next)
                transition_dict['next_goals'].append([goal_next])
                transition_dict['next_speeds'].append([speed_next])
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
            step += 1
            pos = pos_next
            scan = scan_next
            goal = goal_next
            speed = speed_next

        if len(transition_dict['rewards']) > 256:
            logger.info('Start updating !!!')
            agent.update(transition_dict)
            logger.info('Updating done !!!')

            global_update += 1
            transition_dict = {'scans': [], 'points': [], 'goals': [], 'speeds': [], 'actions': [], 'next_scans': [],
                               'next_points': [], 'next_goals': [], 'next_speeds': [], 'rewards': [], 'dones': []}

        # save the model
        if env.index == 0:
            if global_update != 0 and global_step % 20 == 0:
                torch.save(agent.actor.state_dict(), model_path + 'train_actor{}'.format(global_update/20))
                torch.save(agent.critic.state_dict(), model_path + 'train_critic{}'.format(global_update/20))
        logger.info('Episode %05d, step %03d, Goal (%05.1f, %05.1f), Return %-5.1f, %s'
                    % (episode + 1, step, env.goal_point[0], env.goal_point[1], episode_return, result))
        writer.add_scalar('train/return', episode_return, episode + 1)


def test_agent(env, agent, num_episodes, model_path, writer, logger):
    print ('Go Testing !!!')
    actor_dict = torch.load(model_path + 'train_actor{}'.format())
    critic_dict = torch.load(model_path + 'train_critic{}'.format())
    agent.actor.load_state_dict(actor_dict)
    agent.critic.load_state_dict(critic_dict)

    global_step = 0
    if env.index == 0:
        env.reset_world()

    for episode in range(num_episodes):
        episode_return = 0
        step = 1
        done = False

        env.reset_pose()
        env.generate_goal_point()

        obs = env.get_laser_observation()           # (512,)
        pos = to_point(obs)                         # (512, 2)
        obs_stack = deque([obs, obs])
        scan = obs_stack
        goal = np.asarray(env.get_local_goal())     # (2,)
        speed = np.asarray(env.get_self_speed())    # (2,)

        while not done and not rospy.is_shutdown():
            action = agent.take_action(scan=scan, point=pos, goal=goal, speed=speed)
            env.control_vel(action)
            rospy.sleep(0.001)
            reward, done, result = env.get_reward_and_terminate(step)
            episode_return += reward
            global_step += 1
            # get next state
            obs_next = env.get_laser_observation()
            pos_next = to_point(obs_next)
            left = obs_stack.popleft()
            obs_stack.append(obs_next)
            scan_next = obs_stack
            goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            step += 1
            pos = pos_next
            scan = scan_next
            goal = goal_next
            speed = speed_next
            logger.info('Episode %05d, step %03d, Reward %-5.1f, %s' % (episode + 1, step, reward, result))
            print ('-------------------------')
            print (agent)
        writer.add_scalar('test/return', episode_return, episode + 1)


if __name__ == '__main__':
    main()


