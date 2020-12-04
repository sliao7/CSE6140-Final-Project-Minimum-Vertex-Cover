# undirected and unweighted graph object

from collections import deque,defaultdict
class vertex:
    def __init__(self,node):
        self.id = node
        self.adjacent = set()

    def __str__(self):
        # for print out result
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor):
    
        self.adjacent.add(neighbor)

    def remove_neighbor(self, neighbor):
        if neighbor in self.adjacent:
            self.adjacent.remove(neighbor)

    def is_connected(self,neighbor):
        return neighbor in self.adjacent

    def get_connections(self):
        return self.adjacent



class graph:
    # unweighted undirected graph
    # can be connected or not
    edges = None
    def __init__(self, edge_list = False):
        self.vert_dict = {} # vertex_id (int) : vertex
        self.num_vertices = 0
        self.num_edges = 0
        
        if edge_list:
            self.edges = [] # list of edge tuples

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self,node):
        self.num_vertices += 1
        new_vertex = vertex(node)
        self.vert_dict[node] = new_vertex

    def get_vertex(self,node):
        if node in self.vert_dict:
            return self.vert_dict[node]
        else:
            return None

    def add_edge(self, frm, to):
        # for new vertices
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        if not self.vert_dict[frm].is_connected(self.vert_dict[to]):
            self.num_edges += 1

        self.vert_dict[frm].add_neighbor(self.vert_dict[to])
        self.vert_dict[to].add_neighbor(self.vert_dict[frm])
        if self.edges != None:
            self.edges.append((frm, to))

    def remove_vertex(self,node,inplace = False):
        # remove vertex(node) in the graph and all the edges connected to it
        G = self.copy() if not inplace else self
        
        for neighbor in list(G.vert_dict[node].get_connections()):
            G.remove_edge(node, neighbor.id)
            if not neighbor.get_connections():
                # remove resulting isolated vertex
                del G.vert_dict[neighbor.id]
        del G.vert_dict[node]
        
        return G


    def remove_edge(self, frm, to):
        self.vert_dict[frm].remove_neighbor(self.vert_dict[to])
        self.vert_dict[to].remove_neighbor(self.vert_dict[frm])
        self.num_edges -= 1
        if self.edges is not None:
            self.edges = [e for e in self.edges if not e == (frm, to) and not e == (to, frm)]


    def get_vertices(self):
        # return a list of ints, the id of vertices
        return list(self.vert_dict.keys())

    def get_vertex_degree(self,node):
        # return the degree of a vertex in the graph
        return len(self.get_vertex(node).adjacent)

    def copy(self):
        # return a copy of the graph
        copy = graph()
        for v in self.vert_dict:
            for u in self.vert_dict[v].get_connections():
                copy.add_edge(v,u.id)
        return copy



def main():
    data = ['6 7 0', '2 3', '1 3 4', '1 2 5','2 5 6','3 4','4']
    g = graph()
    num_ver, num_edges, _ = map(int, data[0].split(' '))

    for i in range(1, num_ver + 1):
        neighbors = map(int, data[i].split(' ')) 
        for neighbor in neighbors:
            if neighbor > i:
                g.add_edge(i, neighbor)


    for v in g.get_vertices():
        print(v)
        print(g.get_vertex(v))

    print('Number of edges: ', g.num_edges)
    print('Number of vertices, ', g.num_vertices)
   

if __name__ == '__main__':
        main()






