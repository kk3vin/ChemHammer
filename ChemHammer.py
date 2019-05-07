"""
A class to compute a normalised vector of atomic counts and calculate the
Hamming distance to another chemical composition.

Python Parser Source: https://github.com/Zapaan/python-chemical-formula-parser
Periodic table JSON data: https://github.com/Bowserinator/Periodic-Table-JSON
All additional work: Cameron Hargreaves
"""

import re
import json
import os
import sys
import urllib.request

from copy import deepcopy
from collections import Counter

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

if __name__ == "__main__":
    # For personal use as cba configuring environments in vscode
    os.chdir('/home/cameron/Dropbox/University/PhD/ChemHammer')

    x = ChemHammer("Li2(CO)3")
    print(x.composition)
    print(x.normed_composition)
