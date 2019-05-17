"""
adjacancy.py

author: Colin Clement
date: 2017-10-26

Implementation of adjacancy-list graph structure and basic graph traversal
algorithms as described in Skiena chapter 5
"""


class Edgenode(object):
    def __init__(self, y=None, w=1):
        self.y = None
        self.w = w
        self.nextedge = None


class Graph(object):
    def __init__(self, nvertices, directed=False):
        self.nv = nvertices
        self.ne = 0
        self.directed = directed
        self.maxint = 9999999

        self.degrees = [0 for i in range(self.nv)]
        self.edges = [None for i in range(self.nv)]

        self.reset()

    def reset(self):
        self.process_early = lambda x: x
        self.process_edge = lambda x, y: x
        self.process_late = lambda x: x

    def insert(self, x, y, w=1, directed=False):
        p = Edgenode(y, w)
        p.y = y
        p.nextedge = self.edges[x]
        self.edges[x] = p
        self.degrees[x] += 1
        if not directed:
            self.insert(y, x, w, True)
        else:
            self.ne += 1

    def show(self):
        for i in range(self.nv):
            if self.degrees[i]:
                p = self.edges[i]
                edgestr = "vertex {}: ".format(i)
                while p:
                    edgestr += "{} ".format(p.y)
                    p = p.nextedge
                print(edgestr)

    def bfs(self, s):
        self._discovered = [False for i in range(self.nv)]
        self._processed = [False for i in range(self.nv)]
        parent = [None for i in range(self.nv)]

        self._discovered[s] = True
        q = [s]
        while q:
            v = q.pop(0)  # fifo
            self.process_early(v)
            self._processed[v] = True

            p = self.edges[v]
            while p:
                y = p.y
                if not self._processed[y] or self.directed:  # accurate edge count
                    self.process_edge(v, y)
                if not self._discovered[y]:
                    q.append(y)
                    self._discovered[y] = True
                    parent[y] = v
                p = p.nextedge
            self.process_late(v)
        return parent

    def findpath(self, start, end, parents, path=None):
        path = path if path is not None else [end]
        if start == end:
            return path
        elif end is None:
            return []  # no path exists!
        else:
            return self.findpath(start, parents[end], parents, [parents[end]] + path)

    def shortestpath(self, start, end):
        parents = self.bfs(start)  # must have start as root
        return self.findpath(start, end, parents)

    def connected_components(self):
        def addcomponents(x, comp):
            comp += [x]

        comp = []
        self.process_early = lambda x: addcomponents(x, comp)
        self.bfs(0)  # start search from arbitrary node
        components = [comp]

        for i in range(1, self.nv):  # already did 0
            if not self._discovered[i]:
                comp = []
                self.process_early = lambda x: addcomponents(x, comp)
                self.bfs(i)
                components += [comp]

        self.reset()
        return components

    def dfs(self, s):
        self._discovered = getattr(self, "_discovered", [False for i in range(self.nv)])
        self._processed = [False for i in range(self.nv)]
        self._parent = [None for i in range(self.nv)]
        self._entry = [None for i in range(self.nv)]
        self._exit = [None for i in range(self.nv)]
        self._finished = False
        self._time = 0

        def search(v):
            if self._finished:
                return
            self._discovered[v] = True
            self._time += 1
            self._entry[v] = self._time
            self.process_early(v)
            p = self.edges[v]
            while p:
                y = p.y
                if not self._discovered[y]:
                    self._parent[y] = v
                    self.process_edge(v, y)

                    search(y)

                elif (not self._processed[y]) or self.directed:
                    self.process_edge(v, y)
                    if self._finished:
                        return
                p = p.nextedge

            self.process_late(v)
            self._time += 1
            self._exit[v] = self._time
            self._processed[v] = True

        search(s)
        return self._parent, self._entry, self._exit

    def dfs_graph(self):
        self._discovered = [False for i in range(self.nv)]

        def process_early(v, comp):
            comp.append(v)

        components = []
        for i in range(self.nv):
            if not self._discovered[i]:
                comp = []
                self.process_early = lambda x: process_early(x, comp)
                self.dfs(i)
                components.append(comp)
        self.reset()
        del self._discovered
        return components

    def findcycle(self, x):
        cycle = []

        def process_edge(x, y, path=cycle):
            """ Note the subtle conditional for finding a back edge. """
            if self._parent[x] != y:
                p = self.findpath(y, x, self._parent)
                if p:  # only quit if theres a path completing this back edge
                    self._finished = True  # stop looking!
                    [path.append(i) for i in p]

        self.process_edge = process_edge
        self.dfs(x)
        self.reset()
        return cycle

    def edge_classification(self, x, y):
        if self._parent[y] == x:
            return "TREE"
        if self._discovered[y] and not self._processed[y]:
            return "BACK"
        if self._processed[y] and self._entry[y] > self._entry[x]:
            return "FORWARD"
        if self._processed[y] and self._entry[y] < self._entry[x]:
            return "CROSS"
        print("Warning: unclassified edge ({}, {})".format(x, y))

    def findarticulations(self, x=1):
        self._reachable_ancestor = [None for i in range(self.nv)]
        self._tree_out_degree = [0 for i in range(self.nv)]

        def process_early(v):
            self._reachable_ancestor[v] = v

        def process_edge(x, y):
            cls = self.edge_classification(x, y)
            if cls == "TREE":
                self._tree_out_degree[x] += 1
            if cls == "BACK" and self._parent[x] != y:
                # If y is older than current ancestor of x, update
                if self._entry[y] < self._entry[self._reachable_ancestor[x]]:
                    self._reachable_ancestor[x] = y

        articulations = []

        def process_late(v, lst=articulations):
            # Root cut
            if self._parent[v] is None and self._tree_out_degree[v] > 1:
                articulations.append(v)
                return
            # Parent cut
            if self._reachable_ancestor[v] == self._parent[v]:
                if not self._parent[self._parent[v]] is None:  # if parent not root
                    articulations.append(v)
            # Bridge cut
            elif self._reachable_ancestor[v] == v:
                if self._tree_out_degree[v] > 0:  # if v is not a leaf
                    articulations.append(v)
            entry_v = self._entry[self._reachable_ancestor[v]]
            entry_vparent = self._entry[self._reachable_ancestor[self._parent[v]]]
            if entry_v < entry_vparent:
                p = self._parent[v]  # if anc[v] is younger than anc[parent[v]
                self._reachable_ancestor[p] = self._reachable_ancestor[v]

        self.process_early = process_early
        self.process_edge = process_edge
        self.process_late = process_late
        self.dfs(x)
        self.reset()
        return articulations

    def topsort(self):
        assert self.directed, "Graph not directed, topological sort undefined"
        topsorted = []

        def process_late(v, lst=topsorted):
            lst.insert(0, v)  # REVERSE processed order

        self.process_late = process_late

        def process_edge(x, y):
            cls = self.edge_classification(x, y)
            if cls == "BACK":
                print("Warning: directed cycle found, graph not a DAG")

        self._discovered = [False for i in range(self.nv)]
        for i in range(self.nv):
            if not self._discovered[i]:
                self.dfs(i)
        del self._discovered
        self.reset()
        return topsorted

    def prim_mst(self, start=1):
        intree = [False for i in range(self.nv)]
        distance = [self.maxint for i in range(self.nv)]
        parent = [-1 for i in range(self.nv)]

        v = start
        distance[v] = 0
        while not intree[v]:
            intree[v] = True
            p = self.edges[v]
            while p:
                y = p.y
                w = p.w
                if distance[y] > w and not intree[y]:
                    distance[y] = w
                    parent[y] = v
                p = p.nextedge

            v = 1
            d = self.maxint
            for i in range(self.nv):
                if not intree[i] and distance[i] < d:
                    d = distance[i]
                    v = i

        return parent
