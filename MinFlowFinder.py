"""
Take the frequency table and pettifor distances from ChemHammer and apply 
a minimal cost multi-commodity flow algorithm to find the optimal "distance"
between the two. All combinatorial optimization algorithms are taken from the 
Google Operations Research Tools library (OR-Tools) 

Copyright 2019 Google
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from ortools.graph import pywrapgraph

class MinFlowFinder():
    def __init__(self, source, sink):
        self.source = source
        self.sink = sink
        self.start_nodes = self._generate_parameters(source, sink)
        self.end_nodes = self._generate_end_nodes(source, sink)
        self.capacities = self._generate_capacities(source, sink)
        print(self.capacities)
        self.costs = self._generate_costs(source, sink)
        self.dist = 5

    def _generate_parameters(self, source, sink):
        start_nodes = []
        start_labels = []
        end_nodes = []
        end_labels = []

        capacities = [] 
        costs = [] 
        supplies = []

        for i, key_value_source in enumerate(source.items()):
            for j, key_value_sink in enumerate(sink.items()):
                start_nodes.append(i)
                start_labels.append(key_value_source[0])
                end_nodes.append(j + len(source))
                end_labels.append(key_value_sink[0])
                capacities.append(min(key_value_source[1], key_value_sink[1]))
                costs.append(abs(_get_atomic_num))
        return start_nodes

    def _generate_end_nodes(self, source, sink):
        end_nodes = []
        for _, _ in enumerate(source):
            for i, _ in enumerate(sink):
                end_nodes.append(i + len(source))
        return end_nodes

    def _generate_capacities(self, source, sink):
        capacities = []
        for _, v1 in source.items():
            for _, v2 in sink.items():
                capacities.append(min(v1, v2))
        return capacities

    def _generate_costs(self, source, sink):
        return [4]