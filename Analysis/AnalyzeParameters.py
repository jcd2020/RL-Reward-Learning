from gym_minigrid.envs.foodworld import FoodEnv
from random import randint
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import os

"""
This file is used to analyze the properties of different configurations of food types/distributions, to give a sense
of how the parameters of the RL world should be set.

For a given parametrization it tells you the average time required to acquire the resources needed under a couple of different
models: pursue closest food, pursue highest calorie food within windows size d, other strategies TBD
"""






class Simulation:
    k = 0
    Nutrients_i = []
    p_i = []
    m = 15
    n = 4

    def __init__(self):
        self.m = Simulation.m
        self.n = Simulation.n
        self.agent_location = randint(0, self.m - 1), randint(0, self.m - 1)

        self.gen_grid()

    def gen_grid(self):
        food_coordinates = []
        draws = self.get_foods()

        while len(food_coordinates) < self.n:
            x, y = randint(0, self.m - 1), randint(0, self.m - 1)
            while (x, y) in food_coordinates or (x, y) == self.agent_location:
                x, y = randint(0, self.m - 1), randint(0, self.m - 1)
            # that will make sure we don't add (7, 0) to cords_set

            food_coordinates.append((x, y))
        self.food_placement = list(zip(food_coordinates, draws))

    @staticmethod
    def set_params(m, n, k, Nutrients_i, p_i):
        Simulation.k = k
        Simulation.Nutrients_i = Nutrients_i
        Simulation.p_i = p_i
        Simulation.m = m
        Simulation.n = n

    def get_foods(self):
        draws = np.random.multinomial(self.n, Simulation.p_i)

        foods = []
        for idx, count in enumerate(draws):
            foods.extend([(idx, -1)]*count)

        np.random.shuffle(foods)

        return foods

    def dist(self, k):
        return abs(k[0] - self.agent_location[0]) + abs(k[1] - self.agent_location[1])

    def sort_foods_by_distance(self):
        dic = {(k, (v[0], self.dist(k))) for (k,v) in self.food_placement}
        res = sorted(dic, key=lambda t: t[1][1])
        self.food_placement = res


def greedy_navigation():
    accumulated_nutrients_t = np.asarray([])
    curr_accum_nutrients = np.reshape(np.asarray([0]*(1+len(Simulation.Nutrients_i[0]))), (1,3))
    s = Simulation()

    while len(s.food_placement) > 0:
        s.sort_foods_by_distance()
        nearest_food = s.food_placement.pop(0)[1]
        food_item = nearest_food[0]
        dist = nearest_food[1]

        nutrients = [1]
        nutrients.extend(Simulation.Nutrients_i[food_item])
        new_nutrients_transcript = np.asarray([curr_accum_nutrients[0]] * (dist - 1))

        curr_accum_nutrients = np.reshape(np.add(nutrients, curr_accum_nutrients), (1,3))
        if dist > 1:
            new_nutrients_transcript = np.append(new_nutrients_transcript, curr_accum_nutrients, axis=0)
        else:
            new_nutrients_transcript = curr_accum_nutrients
        if len(accumulated_nutrients_t) == 0:
            accumulated_nutrients_t = new_nutrients_transcript
        else:
            accumulated_nutrients_t = np.append(accumulated_nutrients_t, new_nutrients_transcript, axis=0)
    return np.append(accumulated_nutrients_t, np.asarray([curr_accum_nutrients[0]] * (250 - len(accumulated_nutrients_t))),
              axis=0)


def run_greedy_simulation(iters=1000):
    transcripts = np.asarray([])
    for i in range(iters):
        transcript = greedy_navigation()
        transcript = np.asarray(list(map(lambda x: [x], transcript)))
        if len(transcripts) == 0:
            transcripts = transcript
        else:
            transcripts = np.append(transcripts, transcript, axis=1)

    percentile1 = np.percentile(transcripts, .4, axis=1)
    mean = np.mean(transcripts, axis=1)
    return percentile1, mean

def make_a_plot(x, y, title, y_axis, x_axis, results_dir):
    tick_spacing = 20
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    title = '{} {}'.format(title, y_axis)
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    fig_title = title.replace(' ', '_') + '.png'
    plt.savefig(results_dir + '/' + fig_title)

def plots(data, name, results_dir):
    x = np.arange(0, len(data))
    y0 = np.transpose(data)[0]

    make_a_plot(x, y0, name, 'number of food items', 'time step', results_dir)

    for i in range(1, len(np.transpose(data))):
        y_i = np.transpose(data)[i]
        make_a_plot(x, y_i, name, 'nutrient {}'.format(i), 'time step', results_dir)


def run_experiment(m, n, k, Nutrients_i, p_i, results_dir):
    Simulation.set_params(m, n, k, Nutrients_i, p_i)
    percentile1, mean = run_greedy_simulation()


    plots(percentile1, '40th percentile', results_dir)
    plots(mean, 'average', results_dir)


def uniform_experiment_1():
    k = 6
    m = 20
    n = 5
    Nutrients_i = [[100, 200]] * k
    p_i = [1 / k] * k

    results_dir = '{0}x{0}_{1}_items_{2}_{3}_categories'.format(m, n, k, 'uniform')

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    run_experiment(m, n, k, Nutrients_i, p_i, results_dir)

if __name__ == "__main__":
    uniform_experiment_1()
