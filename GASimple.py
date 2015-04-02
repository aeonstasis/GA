__author__ = 'Aaron'
import random
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from math import *
from types import *

# Define number of bits to use in the chromosome and number of generations
numBits = 20
num_gens = 200
cross = 0.9
mut = 0.05


# noinspection PyMethodMayBeStatic
class GA:
    """
    Encapsulates the methods needed to solve an optimization problem using a genetic
    algorithm.
    """
    function = None
    n = None
    constraints = None
    problem = None

    avgs = num_gens * [None]                               # Record data and keep track of which run
    bests = num_gens * [None]
    data = [avgs, bests]
    cnt = 0

    def __init__(self, function, n, constraints, problem):
        """
        Initialize a GA problem
        :param function: the function describing the problem
        :param n: size of initial population
        :param constraints: restrictions on the value of an individual
        :param problem: takes arguments "min" and "max"
        :return:
        """
        assert(isinstance(function, FunctionType))
        self.function = function
        self.n = n
        self.constraints = constraints
        self.problem = problem
        Elem.init_args(function, constraints, problem)

    def __select(self, pop_subset, t):
        """
        Implements tournament-style selection of the population.
        :param pop_subset: some subset of the population
        :param t: size of the tournament
        :return: fittest individual from some subset of the population.
        """
        assert(t >= 1)
        best = pop_subset[random.randrange(0, len(pop_subset))]
        for i in range(1, t):
            next_ind = pop_subset[random.randrange(0, len(pop_subset))]
            if self.problem == 'max':
                if next_ind.fitness > best.fitness:
                    best = next_ind
            else:
                if next_ind.fitness < best.fitness:
                    best = next_ind
        return best

    @staticmethod
    def __mutate(elem):
        """
        Implements bit-flip mutation.
        """
        chromosome = elem.chromosome
        if mut is None:
            p = 1.0 / len(chromosome)
        else:
            p = mut
        for i in range(len(chromosome)):
            if p >= random.random():
                bit = 1 if chromosome[i] == 0 else 1
                chromosome = chromosome[0:i] + str(bit) + chromosome[i+1:]
        elem.set_chromosome(chromosome)

    @staticmethod
    def __crossover(e1, e2):
        """
        Implement uniform crossover, given two parent individuals.
        Return two children constructed from the chromosomes.
        :type e1: Elem
        :type e2: Elem
        :param e1: first parent
        :param e2: second parent
        :return: tuple containing two chromosome bitstrings
        """
        assert(isinstance(e1, Elem) and isinstance(e2, Elem))
        assert(len(e1.chromosome) == len(e2.chromosome))
        if cross is None:
            p = 1 / len(e1.chromosome)
        else:
            p = cross
        for i in range(len(e1.chromosome)):
            if p >= random.random():
                temp = e1.chromosome[i]
                e1.set_chromosome(e1.chromosome[0:i] + e2.chromosome[i] + e1.chromosome[i+1:])
                e2.set_chromosome(e2.chromosome[0:i] + temp + e2.chromosome[i+1:])
        return Elem(e1.chromosome), Elem(e2.chromosome)

    def update_data(self, avg_fitness, best_fitness, i):
        avg_list = GA.avgs[i]
        if avg_list is None:
            GA.avgs[i] = [avg_fitness]
        else:
            avg_list.append(avg_fitness)
            GA.avgs[i] = avg_list
        best_list = GA.bests[i]
        if best_list is None:
            GA.bests[i] = [best_fitness]
        else:
            best_list.append(best_fitness)
            GA.bests[i] = best_list

    def optimize(self):
        """
        Implements a genetic algorithm in order to solve an
        optimization problem.
        :return: the solution to the optimization problem
        """
        population = self.n * [None]                                    # Initialize population of random individuals
        for i in range(0, self.n):
            bit_string = random.randint(0, pow(2, numBits))             # Bitstring treated as unsigned integer
            format_string = '0' + str(numBits) + 'b'
            bit_string = format(bit_string, format_string)
            ind = Elem(bit_string)
            population[i] = ind

        best = None

        # Run for num_gens generations
        for i in range(num_gens):                                       # Control number of generations
            total_fitness = 0
            for elem in population:                                     # Identify current most fit individual
                elem_fitness = elem.fitness
                total_fitness += elem_fitness
                if self.problem == "max":
                    if best is None or elem_fitness > best.fitness:
                        best = elem
                else:
                    if best is None or elem_fitness < best.fitness:
                        best = elem

            avg_fitness = total_fitness / len(population)               # Record fitness value of population for run
            best_fitness = best.fitness
            self.update_data(avg_fitness, best_fitness, i)

            population_next = 0 * [None]
            for j in range(self.n // 2):                                # Generate next generation of individuals
                p_a = GA.__select(self, population, 2)                        # Tournament select two parents
                p_b = GA.__select(self, population, 2)
                (c_a, c_b) = GA.__crossover(p_a.copy(), p_b.copy())     # Create two children through uniform crossover
                GA.__mutate(c_a)                                        # Mutate both children using uniform bit-flip
                GA.__mutate(c_b)
                population_next.append(c_a)                             # Add children to population pool for next gen
                population_next.append(c_b)
            population = population_next                                # Advance one generation
        assert(isinstance(best, Elem))

        GA.cnt += 1                                                     # Advance run count

        return best.decode(best.chromosome)                             # Return decoded value of best individual


# noinspection PyMethodMayBeStatic
class Elem:
    """
    Represents an individual in a population.
    Contains its own chromosome (bitstring) and a calculated numerical fitness.
    """

    function = None
    constraints = None
    problem = None

    def __init__(self, chromosome):
        self.chromosome = chromosome
        if not(hasattr(self, 'fitness')):
            self.fitness = self.__calc_fitness()

    def set_chromosome(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.__calc_fitness()

    def __calc_fitness(self):
        """
        Calculate the fitness based off the objective function
        and chromosome.
        :return: an individual's fitness
        """
        val = self.decode(self.chromosome)
        if isinstance(Elem.function, FunctionType):
            return Elem.function(val)

    def copy(self):
        """
        Return deep copy of an individual.
        :param elem: individual to be cloned
        :return: a deep copy
        """
        copy = Elem(self.chromosome)
        return copy

    def decode(self, chromosome):
        """
        Decode the chromosome bitstring to a value.
        :param chromosome: individual's chromosome
        :return: numerical equivalent of the chromosome
        """
        value = int(chromosome, 2)
        c_len = len(chromosome)
        c_range = self.constraints[1] - self.constraints[0]
        value = (value / (pow(2, c_len) - 1)) * c_range + self.constraints[0]
        return value

    @staticmethod
    def init_args(function, constraints, problem):
        """
        Used to initiate the function and constraints variables within the Elem scope
        :param function: objective function
        :param constraints: constraints on domain
        """
        Elem.function = function
        Elem.constraints = constraints
        Elem.problem = problem


def plot(ga_data):
    """
    Plot the average and best fitness values for each generation and run
    """
    avgs = ga_data[0]
    bests = ga_data[1]

    x_vals = np.arange(1, num_gens + 1, 1)

    avgs_means = np.mean(avgs, axis=1)
    avgs_errors = np.std(avgs, axis=1)

    bests_means = np.mean(bests, axis=1)
    bests_errors = np.std(avgs, axis=1)

    ax1 = plt.subplot("211")                                            # Plot average fitness values
    ax1.errorbar(x_vals, avgs_means, -1 * avgs_errors, 1 * avgs_errors, fmt='o')
    ax1.set_title("Average Generational Fitness")
    ax1.set_ylabel("Fitness Function")
    ax1.xaxis.set_ticks(np.arange(0, num_gens, int(num_gens / 10)))
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    ax2 = plt.subplot("212", sharex=ax1)                                # Plot best fitness values
    ax2.errorbar(x_vals, bests_means, -1 * bests_errors, 1 * bests_errors, fmt='o')
    ax2.xaxis.set_ticks(np.arange(0, num_gens, int(num_gens / 10)))
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax2.set_title("Best Generational Fitness")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness Function")

    #plt.show()
    plt.savefig("../Plots/GA_1.pdf")


if __name__ == '__main__':
    """
    Define functions to test on
    """
    """
    # Print metadata
    print("*****FUN WITH GENETIC ALGORITHMS*****AARON ZOU*****")
    print("Chromosomes are " + str(numBits) + " bits.")
    print("There are " + str(num_gens) + " generations used.\n")

    # Lab test case
    def h(x):
        return pow(6*x-2, 2) * sin(12*x-4)

    Problem3 = GA(h, 20, [0, 1], "min")
    print("For h(x) = (6x-2)^2*sin(12x-4), [0, 1]:")

    for _ in range(10):
        x3 = Problem3.optimize()
        print("\tminimum (x, y) was: (" + format(x3, ".5f")
              + ", " + format(h(x3), ".5f") + ")")

    # Plot and save data
    pickle.dump(GA.data, open("data.p", "wb"), -1)
    plot(GA.data, 200)
    """

    #Load data and plot
    data = pickle.load(open("data.p", "rb"))
    plot(data)
