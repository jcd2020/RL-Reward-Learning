#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
from gym_minigrid.envs.foodworld import FoodEnv
def redraw(img):
    if not args['agent_view']:
        img = env.render('rgb_array', tile_size=args['tile_size'])
    window.set_caption(env.get_mission())
    window.show_img(img)

def reset():
    if args['seed'] != -1:
        env.seed(args['seed'])

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.get_mission())
        window.set_caption(env.get_mission())

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)

    if done:
        if reward == 1.:
            print('you win!')
        else:
            print('you lose :\'(')
        reset()
    else:
        redraw(obs)

def key_handler(event):

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return




args = {'env': 'MiniGrid-FoodWorld-16x16-N3-v0',
        'seed': -1,
        'tile_size': 32,
        'agent_view': False}

env = FoodEnv(m=30, n=5, max_steps=110, min_nutrients=300)

print('loaded env')
if args['agent_view']:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args['env'])
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
