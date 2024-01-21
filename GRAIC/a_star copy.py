"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math

import matplotlib.pyplot as plt
import heapq
show_animation = True

class AStarPlanner:
    def __init__(self, ox, oy, resolution, rr):
        self.resolution = resolution
        self.rr = rr
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.obstacle_map = [[False for _ in range(self.y_width)] for _ in range(self.x_width)]
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()


    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        start_node = self.Node(self.calc_xy_index(sx, self.min_x), self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x), self.calc_xy_index(gy, self.min_y), 0.0, -1)
        open_set = dict()
        closed_set = dict()
        open_set[self.calc_grid_index(start_node)] = start_node
        priority_queue = []
        heapq.heappush(priority_queue, (start_node.cost + self.calc_heuristic(goal_node, start_node), self.calc_grid_index(start_node)))

        while priority_queue:
            _, current_index = heapq.heappop(priority_queue)
            if current_index in open_set:
                current = open_set[current_index]
                del open_set[current_index]
                closed_set[current_index] = current

                if current.x == goal_node.x and current.y == goal_node.y:
                    return self.calc_final_path(goal_node, closed_set)

                for move_x, move_y, move_cost in self.motion:
                    node = self.Node(current.x + move_x, current.y + move_y, current.cost + move_cost, current_index)
                    node_index = self.calc_grid_index(node)
                    if not self.verify_node(node):
                        continue
                    if node_index in closed_set:
                        continue
                    if node_index not in open_set or open_set[node_index].cost > node.cost:
                        open_set[node_index] = node
                        heapq.heappush(priority_queue, (node.cost + self.calc_heuristic(goal_node, node), node_index))

        return None  # Failed to find a path

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion





def main():
    print(__file__ + " start!!")
    print("Start " + __file__)
    file_path = 'boundary2.txt'

    # This will store the tuples.
    obstacle_list = []

    # Open the file for reading.
    with open(file_path, 'r') as file:
    # Read each line in the file one by one.
        for line in file:
            # Strip whitespace and then split the line into two parts at the comma.
            # This assumes that there is no space after the comma.
            left_part, right_part = line.strip().split('),(')
            
            # Remove any remaining parentheses.
            left_part = left_part.strip('(')
            right_part = right_part.strip(')')
            
            # Now split by comma to separate the values and convert them to integers or floats as needed.
            left_x, left_y = map(float, left_part.split(','))
            right_x, right_y = map(float, right_part.split(','))
            size = 0.5
            # Add the tuple to the list. It assumes size1 and size2 are the same, as per your structure.
            # If size1 and size2 are different, adjust accordingly.
            obstacle_list.append(((left_x, left_y)))
            obstacle_list.append((right_x, right_y))
    # start and goal position
    sx = 101.0  # [m]
    sy = 93.0# [m]
    gx = 99.0  # [m]
    gy = -6.0  # [m]
    grid_size = 1.0  # [m]
    robot_radius = 1.0  # [m]
    # print(obstacle_list)
    # set obstacle positions
    ox, oy = [], []
    for obstacle in obstacle_list:
        ox.append(obstacle[0])
        oy.append(obstacle[1])

    # for i in range(-10, 60):
    #     ox.append(i)
    #     oy.append(-10.0)
    # for i in range(-10, 60):
    #     ox.append(60.0)
    #     oy.append(i)
    # for i in range(-10, 61):
    #     ox.append(i)
    #     oy.append(60.0)
    # for i in range(-10, 61):
    #     ox.append(-10.0)
    #     oy.append(i)
    # for i in range(-10, 40):
    #     ox.append(20.0)
    #     oy.append(i)
    # for i in range(0, 40):
    #     ox.append(40.0)
    #     oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()
