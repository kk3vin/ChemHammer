"""
A class to compute a normalised vector of atomic counts and calculate the
Hamming distance to another chemical composition.

Python Parser Source: https://github.com/Zapaan/python-chemical-formula-parser
Periodic table JSON data: https://github.com/Bowserinator/Periodic-Table-JSON
All additional work: Cameron Hargreaves

TODO: Remove duplicates in the pairwise distances matrix
TODO: Refine the hamming distance metric
"""

import re
import json
import os
import sys
import urllib.request

from copy import deepcopy
from collections import Counter

import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment


class ChemHammer():
    ATOM_REGEX = '([A-Z][a-z]*)(\d*)'
    OPENERS = '({['
    CLOSERS = ')}]'


    def __init__(self, formula):
        self.formula = formula
        self.periodic_tab = self.get_periodic_tab()
        self.composition = self.parse_formula(formula)
        self.normed_composition = self.normalise_composition(self.composition)

    def get_periodic_tab(self):
        """
        Attempt to load periodic data from the same folder, else download
        it from the web
        """
        try:
            with open('Periodic-Table-JSON.json') as json_data:
                periodic_data = json.load(json_data)
            return periodic_data

        except FileNotFoundError as e:
            print(f"File failed to load due to {e}")
            print("Attempting to download from the web, please allow firewall access")
            url = 'https://raw.githubusercontent.com/SurgeArrester/ChemHammer/master/Periodic-Table-JSON.json'
            response = urllib.request.urlopen(url)
            data = response.read()      # a `bytes` object
            data = data.decode('utf-8')
            periodic_data = json.loads(data)
            return periodic_data

        except Exception as e:
            print(f"Failed due to {e}")


    def is_balanced(self, formula):
        """Check if all sort of brackets come in pairs."""
        # Very naive check, just here because you always need some input checking
        c = Counter(formula)
        return c['['] == c[']'] and c['{'] == c['}'] and c['('] == c[')']


    def _dictify(self, tuples):
        """Transform tuples of tuples to a dict of atoms."""
        res = dict()
        for atom, n in tuples:
            try:
                res[atom] += int(n or 1)
            except KeyError:
                res[atom] = int(n or 1)
        return res


    def _fuse(self, mol1, mol2, w=1):
        """
        Fuse 2 dicts representing molecules. Return a new dict.
        """
        return {atom: (mol1.get(atom, 0) + mol2.get(atom, 0)) * w for atom in set(mol1) | set(mol2)}


    def _parse(self, formula):
        """
        Return the molecule dict and length of parsed part.
        Recurse on opening brackets to parse the subpart and
        return on closing ones because it is the end of said subpart.
        """
        q = []
        mol = {}
        i = 0

        while i < len(formula):
            # Using a classic loop allow for manipulating the cursor
            token = formula[i]

            if token in self.CLOSERS:
                # Check for an index for this part
                m = re.match('\d+', formula[i+1:])
                if m:
                    weight = int(m.group(0))
                    i += len(m.group(0))
                else:
                    weight = 1

                submol = self._dictify(re.findall(self.ATOM_REGEX, ''.join(q)))
                return self._fuse(mol, submol, weight), i

            elif token in self.OPENERS:
                submol, l = self._parse(formula[i+1:])
                mol = self._fuse(mol, submol)
                # skip the already read submol
                i += l + 1
            else:
                q.append(token)

            i+=1

        # Fuse in all that's left at base level
        return self._fuse(mol, self._dictify(re.findall(self.ATOM_REGEX, ''.join(q)))), i


    def parse_formula(self, formula):
        """Parse the formula and return a dict with occurences of each atom."""
        if not self.is_balanced(formula):
            raise ValueError("Your brackets not matching in pairs ![{]$[&?)]}!]")

        return self._parse(formula)[0]

    def normalise_composition(self, composition):
        """
        Sum up the numbers in our counter to get total atom count
        """
        composition = deepcopy(composition)
        # check it has been processed
        if isinstance(composition, str):
            composition = self.parse_formula(composition)

        atom_count =  sum(composition.values(), 0.0)

        for atom in composition:
            composition[atom] /= atom_count

        return composition

    def get_atomic_num(self, element_string):
        """
        Return atomic number from element string
        """
        for i, element in enumerate(self.periodic_tab['elements']):
            if element['symbol'] == element_string:
                return i

    def euclidean_dist(self, comp2, comp1 = None):
        """
        Simply take the euclidean distance between two vectors excluding atom
        similarity. Here we take the normalised vector and assume that the input
        is a string formula
        """
        comp1 = comp1 if comp1 is not None else self.normed_composition

        if isinstance(comp2, str):
            comp2 = self.parse_formula(comp2)
            comp2 = self.normalise_composition(comp2)

        dist = 0

        # Loop through the union of  keys
        for key in list(comp1.keys()) + list(comp2.keys()):
            # Simply take the distance between if element is present in both
            if key in comp1 and key in comp2:
                dist += abs(comp1[key] - comp2[key])

            #If its' not a shared element try and take distance from each dict
            elif key in comp1:
                dist += comp1[key]

            elif key in comp2:
                dist += comp2[key]

            else:
                print("Key not in either, strange bug occurred")

        return dist

    def position(self, element):
        atomic_num = self.get_atomic_num(element)
        atom_info = self.periodic_tab['elements'][atomic_num]

        return (atom_info['xpos'], atom_info['ypos'])

    def return_positions(self, composition):
        element_pos = {}

        for element in composition:
            element_pos[element] = self.position(element)

        return element_pos

    def pairwise_dist(self, comp1_orig, comp2_orig):
        """
        Return matched pairs of closest elements
        TODO: Refactor the code to use a simpler datastructure than dictionarys
        of dictionarys
        """
        # Make copies of these to avoid changing original
        comp1 = deepcopy(comp1_orig)
        comp2 = deepcopy(comp2_orig)

        pairing_dict = {}

        # Check every element in the table, add to pair and pop from both lists if theres an exact match
        for element in list(comp1.keys()):
            if element in comp2:
                # Remove from both lists if we have a shared element and update with distance 0
                pairing_dict[element] = {element: 0}
                comp1.pop(element)
                comp2.pop(element)

        # List the elements we're using for use in a square matrix, craete
        # a list of the coordinates of these on the periodic table, and make
        # a square distance matrix
        comp1_elements = list(comp1.keys())
        comp2_elements = list(comp2.keys())
        elements = comp1_elements + comp2_elements

        coords = list(map(self.position, elements))
        dist_matrix = np.array(squareform(pdist(coords, metric="cityblock")))

        # As we only want to match those from opposing compositions we will
        # look at the top right quadrant only
        dist_cost = dist_matrix[:len(comp1), len(comp1):]

        # Use the minimum weight matching algorithm for bipartite graphs to find
        # the best combination of these
        row_ind, col_ind = linear_sum_assignment(dist_cost)

        for i, _ in enumerate(row_ind):
            pairing_dict[comp1_elements[row_ind[i]]] = {comp2_elements[col_ind[i]] : dist_cost[row_ind[i]][col_ind[i]]}

        return pairing_dict


    def hamming_dist(self, comp2, comp1 = None):
        """
        This is similar to euclidean distance, however adds a further distance
        metric depending on manhattan distance between closest neighbours
        """
        comp1 = comp1 if comp1 is not None else self.normed_composition

        if isinstance(comp2, str):
            comp2 = self.parse_formula(comp2)
            comp2 = self.normalise_composition(comp2)

        if len(comp1) != len(comp2):
            print("Must have equal numbers of elements for Hamming distance, use levenshtein distance")
            return None

        dist = 0

        comp1_pos = self.return_positions(comp1)
        comp2_pos = self.return_positions(comp2)

        pairwise_matches = self.pairwise_dist(comp1_pos, comp2_pos)

        # Loop through the union of  keys
        for key, value in pairwise_matches.items():
            dist += abs(comp1[key] - comp2[value[0]]) + value[1]

        return dist

    def levenshtein_dist(self, comp2, comp1 = None):
        comp1 = comp1 if comp1 is not None else self.normed_composition

        if isinstance(comp2, str):
            comp2 = self.parse_formula(comp2)
            comp2 = self.normalise_composition(comp2)

        dist = 0

        comp1_pos = self.return_positions(comp1)
        comp2_pos = self.return_positions(comp2)

        # Find the closest matching letters for the first composition
        pairwise_matches = self.pairwise_dist(comp1_pos, comp2_pos)



if __name__ == "__main__":
    # For personal use as cba configuring environments in vscode
    os.chdir('/home/cameron/Dropbox/University/PhD/ChemHammer')

    x = ChemHammer("Li4(CO)6Rb")
    print(x.composition)
    print(x.normed_composition)
    print(x.euclidean_dist("Li4(C6O6)"))
    print(x.hamming_dist("K6(C6O6)Sr"))
    print()
