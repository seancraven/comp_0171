"""
File implementing forwards backwards algorithm and viterbi on toy problem.


The toy problem in question is inference of the location of a burglar in
a kitchen. The kitchen is described by a grid, where each grid point has
a chance of squeaking or creaking. Using the sequence of squeaks and creaks
heard from the kitchen we can perform inference on a Hidden Markov Model.

Additionally, I am avoiding using numpy to improve my familiarity with
torch.

Outline of the probelm structure:
A sequence of creaks and squeaks are retured in adition to a known kitchen
layout. This is in the form of a List[str].

Each Kitchen is a new object, the user can pass a pre-determined layout
or a random layout can be generated.
"""
import torch
from torch import distributions as dist
import matplotlib.pyplot as plt
from typing import Tuple, List
import random


class Kitchen:
    """
    The kitchen class defines the problem space.

    An instance of a kitchen defines a new kitchen with a size and creaky
    and squeaky floorboards.
    The kitchen is represented by a grid of discrete points. In this grid
    some points "creak" and some "squeak".
    By default the creaking and squeaking is controlled by Bernoulli samples.
    Coustom input kitchens can be made and passed to the class.
    """

    def __init__(self, size: Tuple[int, int] = (5, 5), rng: int = 0, **cr_sq):
        """
        Initalise kitchen, by default generates random 5,5 gird.

        Default behaviour can be adjusted using args:
        Args:
            size: kitchen matrix size as a tuple.
            rng: Random seed for torch
            **cr_sq: "squeak", "creak" one hot torch tensors defining
        """
        torch.manual_seed(rng)
        self.squeak = dist.Bernoulli(0.2).sample(size)
        self.p_creak = 1
        self.p_squeak = 1
        self.creak = dist.Bernoulli(0.2).sample(size)
        self.dim = self.squeak.shape
        if cr_sq:
            try:
                self.squeak = cr_sq["squeak"]
                self.creak = cr_sq["creak"]
                self.p_creak = cr_sq["p_creak"]
                self.p_squeak = cr_sq["p_squeak"]
                assert self.squeak.shape == self.creak.shape, \
                    " creaks and squeaks must be the same shape."
                assert 0 <= p_creak <= 1, "p_creak must be a probability."
                assert 0 <= p_squeak <= 1, "p_squeak must be a probability."
            except KeyError("invalid Kwargs, using Defaults"):
                pass

    def show(self):
        plt.matshow(self.squeak)
        plt.xlabel("squeaks")
        plt.show()
        plt.matshow(self.creak)
        plt.xlabel("creaks")
        plt.show()


class Burglar:
    """
    A random walker that proceedes through the space defined by kitchen.
    """

    def __init__(self, kitchen: Kitchen, walk_length: int):
        self.walk_length = walk_length
        self.kitchen = kitchen
        self.path = self._walk()

    def _walk(self):
        """
        Performs the burglars random walk behaviour.

        Returns:
            path_mat: torch tensor of the same shape as kitchen, where the one
            entries are the locations the burglar has walked over.

        """
        path_mat = torch.zeros_like(self.kitchen.squeak)
        position = self.random_start()
        positions = [position]
        for step in range(self.walk_length):
            position = self.take_step(position)
            positions.append(position)
        return positions

    def walk_sounds(self) -> List[str]:
        """
        Method to record the creaks and squeaks of the burglar.

        Returns:
            sounds: list of sounds made.

        """
        squeaks = self.kitchen.squeak
        creaks = self.kitchen.creak
        p_squeak = self.kitchen.p_squeak
        p_creak = self.kitchen.p_creak
        sounds = []
        ## Current sound can be "squeak", "creak", "squeakcreak", ""
        for step in self.path:
            current_sound = ""
            if squeaks[step[0], step[1]]:
                u = random.uniform(0,1)
                if u < p_squeak:
                    current_sound +="squeak"
            if creaks[step[0], step[1]]:
                u = random.uniform(0,1)
                if u < p_creak:
                    current_sound += "creak"
            sounds.append(current_sound)
        return sounds


    def random_start(self) -> List[int]:

        """
        Produces Random start position for the burglar in the kitchen.
        Returns:
            start: Start position index for th

        """
        kitchen_x = self.kitchen.dim[0]
        kitchen_y = self.kitchen.dim[1]
        x = random.randint(0, kitchen_x - 1)
        y = random.randint(0, kitchen_y - 1)
        possible_start = [[x, 0], [x, kitchen_y - 1], [0, y], [kitchen_x - 1, y]]
        random.shuffle(possible_start)
        return possible_start[0]

    def take_step(self, current_pos: List[int]) -> List[int]:
        """
        Performs a random walk step in the kitchen. Only move one unit
        in one direction(no diagonal movement).
        Conceptually takes a step in a bounded 2d random walk.

        Args:
            current_pos: position index.

        Returns:
            new_pos
        """
        # if on edge dof is reduced:
        new_pos = list(current_pos)
        steps = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        random.shuffle(steps)
        while new_pos == current_pos:
            p_step = steps.pop(-1)
            p_pos = [c_p + st for c_p, st in zip(current_pos, p_step)]
            if all(
                [0 <= p and p <= (dim - 1) for p, dim in zip(p_pos, self.kitchen.dim)]
            ):
                new_pos = p_pos
        return new_pos

    def show_walk(self):
        positions = torch.zeros(self.kitchen.dim)
        for pos in self.path:
            positions[pos[0], pos[1]] += 1
        positions[positions > 1] = 1
        plt.matshow(positions.detach())
        plt.xlabel("burglar")
        plt.show()


if __name__ == "__main__":
    test = Kitchen()
    test.show()
    t_b = Burglar(test, 5)
    t_b.show_walk()
    print(t_b.walk_sounds())