# https://medium.com/@msgold/using-python-to-create-and-solve-mazes-672285723c96

import random
from queue import Queue
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Maze:
    
    def __init__(self,dim):
        self.dim = dim
        self.maze = self.create_maze()
        self.path = self.find_path()
        self.graph = self.maze_to_graph()
        self.n_nodes = self.graph.number_of_nodes()
        self.entry = 0
        self.exit = self.n_nodes-1

    def create_maze(self):
        # Create a grid filled with walls
        maze = np.ones((self.dim*2+1, self.dim*2+1))
        # Define the starting point
        x, y = (0, 0)
        maze[2*x+1, 2*y+1] = 0

        # Initialize the stack with the starting point
        stack = [(x, y)]
        while len(stack) > 0:
            x, y = stack[-1]

            # Define possible directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if nx >= 0 and ny >= 0 and nx < self.dim and ny < self.dim and maze[2*nx+1, 2*ny+1] == 1:
                    maze[2*nx+1, 2*ny+1] = 0
                    maze[2*x+1+dx, 2*y+1+dy] = 0
                    stack.append((nx, ny))
                    break
            else:
                stack.pop()

        # Create an entrance and an exit
        maze[1, 0] = 0
        maze[-2, -1] = 0
        return maze

    def find_path(self):
        # BFS algorithm to find the shortest path
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        start = (1, 1)
        end = (self.maze.shape[0]-2, self.maze.shape[1]-2)
        visited = np.zeros_like(self.maze, dtype=bool)
        visited[start] = True
        queue = Queue()
        queue.put((start, []))
        while not queue.empty():
            (node, path) = queue.get()
            for dx, dy in directions:
                next_node = (node[0]+dx, node[1]+dy)
                if (next_node == end):
                    return path + [next_node]
                if (next_node[0] >= 0 and next_node[1] >= 0 and 
                    next_node[0] < self.maze.shape[0] and next_node[1] < self.maze.shape[1] and 
                    self.maze[next_node] == 0 and not visited[next_node]):
                    visited[next_node] = True
                    queue.put((next_node, path + [next_node]))
        return path

    def maze_to_graph(self):
        rows, cols = self.maze.shape
        G = nx.Graph()
        # Iterate through each cell in the maze
        for i in range(rows):
            for j in range(cols):
                if self.maze[i, j] == 0:  # If the cell is open (1)
                    # Add node for the cell
                    G.add_node((i, j))
                    # Check adjacent cells and add edges
                    if i > 0 and self.maze[i - 1, j] == 0:
                        G.add_edge((i, j), (i - 1, j))
                    if i < rows - 1 and self.maze[i + 1, j] == 0:
                        G.add_edge((i, j), (i + 1, j))
                    if j > 0 and self.maze[i, j - 1] == 0:
                        G.add_edge((i, j), (i, j - 1))
                    if j < cols - 1 and self.maze[i, j + 1] == 0:
                        G.add_edge((i, j), (i, j + 1))
        return G

    def draw_maze(self, path=None):
        fig, ax = plt.subplots(figsize=(10,10))

        # Set the border color to white
        fig.patch.set_edgecolor('white')
        fig.patch.set_linewidth(0)

        ax.imshow(self.maze, cmap=plt.cm.binary, interpolation='nearest')
        
        # Draw the solution path if it exists
        if path is not None:
            x_coords = [x[1] for x in path]
            y_coords = [y[0] for y in path]
            ax.plot(x_coords, y_coords, color='red', linewidth=2)

        ax.set_xticks([])
        ax.set_yticks([])

        # Draw entry and exit arrows
        #ax.arrow(0, 1, .4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
        #ax.arrow(self.maze.shape[1] - 1, self.maze.shape[0]  - 2, 0.4, 0, fc='blue', ec='blue', head_width=0.3, head_length=0.3)
        plt.show()
        
    def draw_graph(self,entry=False,exit=False,save=False):
        node_colors = ['black'] * self.graph.number_of_nodes()
        if entry:
            node_colors[0] = 'blue' # entry
        if exit:
            node_colors[-1] = 'red' # exit
        rows, cols = len(self.maze), len(self.maze[0])
        pos = {(i, j): (j, -i) for i in range(rows) for j in range(cols) if self.maze[i][j] == 0}
        nx.draw(self.graph,pos, with_labels=False,node_color=node_colors, edge_color='black', node_size=100, font_size=1)
        if save:
            plt.savefig("maze_graph.png", format="PNG",dpi=300)
        plt.show()
