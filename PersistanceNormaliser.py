"""
Author: Cameron Hargreaves

This simple class takes in a list of persistence points and "normalises" these
so that the total sum of points is equal to one

We include a method for the modified bottleneck distance on these points that
carries forward a recursive bipartite matching on the two, returning the shared
counts and the distance between these, and uses this as a modified score metric

The returned value can be used as a distance metric between each homology group

"""

import os
import pickle as pk

from collections import Counter
from copy import deepcopy

import numpy as np

from scipy.spatial.distance import cdist, squareform, euclidean
from scipy.optimize import linear_sum_assignment

def main():
    test_folder =
    pers_points = pk.load(open(test_string1, "rb"))
    x = PersistenceNorm(pers_points)
    pers_points = pk.load(open(test_string2, "rb"))
    y = PersistenceNorm(pers_points)

    score = x.normalised_bottleneck(y)
    print(score)

class PersistenceNorm():
    def __init__(self, points):
        self.points = points
        self.counter_list = []
        self._count_points()
        self._normalise_points()

    def normalised_bottleneck(self, other, freq_self=None):
        """
        Perform a bartitite maximal matching of two frequency counts,
        recursively called until all points are matched together
        This only takes into account the homology groups of self and will not
        match with higher dimensions in other and currently will break if self
        has more dimensions than other
        """
        if freq_self == None:
            freq_self = deepcopy(self.norm_list)

        if type(other) == PersistenceNorm:
            other = deepcopy(other.norm_list)

        scores = []
        for i, group in enumerate(freq_self):
            matched_pairs = []
            other_group = other[i]

            # Recursively match the closest points in each group, calc their
            # distance and shared cardinality, and append to matched_pairs
            matching = self._bipartite_match(group, other_group, matched_pairs)

            # For each of these points sum the product of their distance and
            # shared cardinality
            scores.append(sum(x[2] * x[3] for x in matching))

        return scores

    def _count_points(self, dp=5):
        """
        Save a list of Counters to self in the form:
        self.point_counter = [Counter(H_0 points), Counter(H_1 points), ...]

        Parameters:
        dp: Int, default 5. The number of decimal places to round to
        """
        counter_list = []

        points = self.points
        # In case it has already been reduced
        if type(points) is Counter:
            counter_list = points

        # Standard output should be a list of 2D numpy arrays
        elif type(points) is list:
            # Strip the infinite points
            for i, diagram in enumerate(points):
                points[i] = diagram[~np.isinf(diagram).any(axis=1)]

            # Round each of these to 5dp, cast to a list of tuples
            # and apply a Counter
            for homology_group in points:
                homology_group = [tuple(x) for x in np.round(homology_group,dp)]
                count = Counter(homology_group)
                counter_list.append(count)

        self.counter_list = counter_list

    def _normalise_points(self, counter_list=None):
        """Performs standard normalisation in each dimension"""
        if counter_list == None:
            if self.counter_list == None:
                self._count_points()
                counter_list = self.counter_list
            else:
                counter_list = self.counter_list

        norm_list = []

        for i, point_counter in enumerate(counter_list):
            total = sum(point_counter.values(), 0.0)
            for key in point_counter:
                point_counter[key] /= total
            norm_list.append(point_counter)

        self.norm_list = norm_list

    def _bipartite_match(self, freq, other, matched_pairs):
        """
        Possibly inefficient implementation to run max bipartite matching
        TODO: Optimise with Hopcroft-Karp algorithm?
        """
        # If we have reached the base case simply add the remaining points
        # unmatched
        if len(freq) == 0:
            for point in other.keys():
                matched_pairs.append(('_', point, np.linalg.norm(point),
                                                  other[point]))
            return matched_pairs

        elif len(other) == 0:
            for point in freq.keys():
                matched_pairs.append((point, '_', np.linalg.norm(point),
                                                  freq[point]))
            return matched_pairs

        # Unpack the dictionaries into lists of points
        x = tuple(freq.keys())
        y = tuple(other.keys())

        # Compute a distance matrix of these and then use the hungarian
        # algorithm in linear_sum_assignment() to create the minimal bipartite
        # matching of these two
        dist_matrix = cdist(x, y)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        x_matched = [x[i] for i in row_ind]
        y_matched = [y[i] for i in col_ind]

        for (x_i, y_i) in zip(x_matched, y_matched):
            if freq[x_i] == other[y_i]:
                matched_pairs.append((x_i, y_i, euclidean(x_i, y_i), freq[x_i]))
                freq.pop(x_i)
                other.pop(y_i)

            elif freq[x_i] > other[y_i]:
                matched_pairs.append((x_i, y_i, euclidean(x_i, y_i),other[y_i]))
                freq[x_i] -= other[y_i]
                other.pop(y_i)

            elif freq[x_i] < other[y_i]:
                matched_pairs.append((x_i, y_i, euclidean(x_i, y_i), freq[x_i]))
                other[y_i] -= freq[x_i]
                freq.pop(x_i)

        # After we have reduced the dictionaries sizes, call this again on the
        # remaining values
        self._bipartite_match(freq, other, matched_pairs)

        return matched_pairs

if __name__ == "__main__":
    main()
