from __future__ import annotations

import heapq
from typing import Tuple, Dict, List
from math import fabs


class Node:
    neighbors: List[Node]
    parent: Node | None

    def __init__(
            self,
            label: str,
            coordinates: Tuple[float, float],
            current_cost: float = float('inf'),
            heuristic: float = 0.0,
    ):
        self.label = label
        self.coordinates = coordinates
        self.current_cost = current_cost
        self.heuristic = heuristic
        self.neighbors = []
        self.parent = None

    def add_neighbor(self, neighbor: Node):
        self.neighbors.append(neighbor)

    def __str__(self):
        neighbor_labels = [neighbor.label for neighbor in self.neighbors]
        return f"Node '{self.label}' with coordinates ({self.coordinates[0]}, {self.coordinates[1]}) and neighbors: {neighbor_labels}"


def get_manhattan_distance(start: Node, end: Node):
    x1, y1 = start.coordinates
    x2, y2 = end.coordinates
    return fabs(x1 - x2) + fabs(y1 - y2)


# heuristic based on manhattan distance
def calculate_heuristic(start_node: Node, finish_nodes: List[Node]):
    heuristic = float('inf')
    for finish_node in finish_nodes:
        distance = get_manhattan_distance(start_node, finish_node)
        heuristic = min(heuristic, distance)
    return heuristic


class Graph:
    nodes: Dict[str, Node]
    steps: int

    def __init__(self, nodes: List[Tuple[str, float, float]], edges: List[Tuple[str, str]]):
        self.nodes = dict()

        for node in nodes:
            label, x, y = node
            node = Node(label, (x, y))
            self.nodes[label] = node

        for edge in edges:
            source, destination = edge
            source_node = self.get_node_with_label(source)
            destination_node = self.get_node_with_label(destination)
            source_node.add_neighbor(destination_node)
            destination_node.add_neighbor(source_node)

    def get_node_with_label(self, label: str):
        return self.nodes[label]

    def a_star(self, start_label: str, finish_labels: List[str], step_count: int) -> str | List[Tuple[float, str]]:
        start_node = self.nodes[start_label]
        finish_nodes = [self.get_node_with_label(finish_label) for finish_label in finish_labels]

        start_node.current_cost = 0
        start_node.heuristic = calculate_heuristic(start_node, finish_nodes)

        open_list = [(start_node.current_cost, start_label)]
        open_dict: Dict[str, Node] = {start_label: start_node}
        closed_set = set()

        self.steps = 0
        while open_list:
            self.steps += 1
            if self.steps > step_count:
                break

            _, current_node_label = heapq.heappop(open_list)
            if current_node_label in finish_labels:
                return current_node_label
            closed_set.add(current_node_label)

            current_node = self.get_node_with_label(current_node_label)
            for neighbor in current_node.neighbors:
                if neighbor.label in closed_set:
                    continue

                tentative_cost = current_node.current_cost + get_manhattan_distance(current_node, neighbor)

                if neighbor.label not in open_dict:
                    neighbor.current_cost = tentative_cost
                    neighbor.heuristic = calculate_heuristic(neighbor, finish_nodes)
                    neighbor.parent = current_node
                    heapq.heappush(open_list, (neighbor.current_cost + neighbor.heuristic, neighbor.label))
                    open_dict[neighbor.label] = neighbor
                elif tentative_cost < open_dict[neighbor.label].current_cost:
                    neighbor.current_cost = tentative_cost
                    neighbor.parent = current_node

        return open_list

    def ida_star(self, start_label: str, finish_labels: List[str], step_count: int) -> None | str | Tuple[float, List[Node]]:
        start_node = self.nodes[start_label]
        finish_nodes = [self.nodes[label] for label in finish_labels]
        threshold = calculate_heuristic(start_node, finish_nodes)

        start_node.current_cost = 0
        start_node.heuristic = calculate_heuristic(start_node, finish_nodes)

        self.steps = 0
        while True:
            path = [start_node]
            visited = {}

            search_result = self._ida_star(start_node, finish_nodes, threshold, visited, path, step_count)

            if search_result == 'stopped':
                return threshold, path
            elif search_result == float('inf'):
                return None
            elif isinstance(search_result, float):
                threshold = search_result
            else:
                return search_result

    def _ida_star(self, current_node: Node, finish_nodes: List[Node], threshold: float, visited: Dict[str, float], path: List[Node], step_count: int) -> float | str:
        self.steps += 1
        if self.steps > step_count:
            return 'stopped'

        tentative_cost = current_node.current_cost + calculate_heuristic(current_node, finish_nodes)
        if tentative_cost > threshold:
            return tentative_cost

        if current_node.label in [node.label for node in finish_nodes]:
            return current_node.label

        if current_node.label in visited and visited[current_node.label] <= current_node.current_cost:
            return float('inf')

        visited[current_node.label] = current_node.current_cost

        mn = float('inf')
        for neighbor in current_node.neighbors:
            neighbor.parent = current_node

            previous_current_cost = neighbor.current_cost
            neighbor.current_cost = current_node.current_cost + get_manhattan_distance(current_node, neighbor)
            path += [neighbor]
            search_result = self._ida_star(neighbor, finish_nodes, threshold, visited, path, step_count)

            if isinstance(search_result, str):
                return search_result

            path.pop()
            neighbor.current_cost = previous_current_cost
            mn = min(mn, search_result)
        return mn


nodes = [('1', -3, 3), ('2', 0, 3), ('3', 3, 3), ('4', -2, 2), ('5', 0, 2), ('6', 2, 2),
         ('7', -1, 1), ('8', 0, 1), ('9', 1, 1), ('10', -3, 0), ('11', -2, 0), ('12', -1, 0),
         ('13', 1, 0), ('14', 2, 0), ('15', 3, 0), ('16', -1, -1), ('17', 0, -1), ('18', 1, -1),
         ('19', -2, -2), ('20', 0, -2), ('21', 2, -2), ('22', -3, -3), ('23', 0, -3), ('24', 3, -3)]
edges = [('1', '2'), ('1', '10'), ('2', '3'), ('2', '5'), ('3', '15'), ('4', '5'),
         ('4', '11'), ('5', '6'), ('5', '8'), ('6', '14'), ('7', '8'), ('7', '12'),
         ('8', '9'), ('9', '13'), ('10', '11'), ('10', '22'), ('11', '12'), ('11', '19'),
         ('12', '16'), ('13', '14'), ('13', '18'), ('14', '15'), ('14', '21'), ('15', '24'),
         ('16', '17'), ('17', '18'), ('17', '20'), ('19', '20'), ('20', '21'), ('20', '23'),
         ('22', '23'), ('23', '24')]
graph = Graph(nodes, edges)

# test code here

# Am ales sa utilizez distanta Manhattan deoarece pentru orice
# doua noduri alese, observam ca ne putem deplasa intre ele ori
# vertical ori orizontal. Astfel, euristica bazata pe distanta
# Manhattan este una valida, deoarece va fi intotdeauna mai
# mica sau egala cu distanta reala.