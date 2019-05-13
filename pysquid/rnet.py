"""
rnet.py

author: Colin Clement date: 2015-11-14

This computes currents in a resistor network provided by a mask
in order to approximate current entering and leaving an image.
"""

from __future__ import print_function

from pysquid.component import ModelComponent
from pysquid.util.graph import Graph
from pysquid.util.helpers import curl

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from collections import defaultdict
from scipy.ndimage import label

def meshcornerindices(i, Lx):
    """ upper-left, lower-left, lower-right, upper-right """
    ul = i + i // Lx
    return [ul, ul + Lx + 1, ul + Lx + 2, ul + 1]

def meshedges(i, Ly, Lx):
    """ left, bottom, right, top """
    Nhor = Lx*(Ly + 1)
    return [i + i//Lx + Nhor, i + Lx, i + i//Lx + Nhor+1, i]  # CCW

def edgemeshes(e, Ly, Lx):
    Nhor = Lx*(Ly + 1)
    if e < Nhor:
        y, x = e // Lx, e % Lx
        if y == 0:  # top edge
            return [e]
        elif y == Ly:  # bottom edge
            return [e - Lx]
        else:
            return [e, e - Lx]
    else:  #vertical
        e -= Nhor
        y, x = e // (Lx+1), e % (Lx+1)
        if x == 0:  # left edge
            return [e - y]
        elif x == Lx:  # right edge
            return [e - y - 1]
        else:
            return [e - y, e - y - 1]

def sharededge(m, n, Ly, Lx):
    medges = set(meshedges(m, Ly, Lx))
    nedges = set(meshedges(n, Ly, Lx))
    return medges.intersection(nedges).pop()

def meshholefromedge(mask, edge):
    flatmask = mask.flat
    for m in edgemeshes(edge, *mask.shape):
        if not flatmask[m]:  # hole in mask
            return m

def nodetoedgepath(nodepath, G):
    epath = []
    for x, y in zip(nodepath[:-1], nodepath[1:]):
        v = G.edges[x]
        while v:
            if v.y == y:
                epath.append(v.w)
                break
            v = v.nextedge
    return epath

def fillones(mask):
    comps, num = label(1 - mask)
    cflat = comps.ravel()
    maskflat = mask.ravel()
    indices = np.array(range(len(cflat)))
    for comp in range(1, num+1):
        i = indices[cflat == comp]
        if len(i)==1:
            maskflat[i] = 1
    return mask

def findtopology(mask, G):
    """ 
    This function returns closed paths in resistor corner indices which surround
    topologically distinct holes in the mask. Returns path in counter-clockwise
    direction in accordance with the right hand rule and positive current
    creating a magnetic field which points out of the page.
    """
    comps, num = label(1 - mask)
    cflat = comps.ravel()
    indices = np.array(range(len(cflat)))
    Lx = mask.shape[1]
    loops = []
    loopindices = []
    for comp in range(1, num+1):
        corners = set()
        for i in indices[cflat == comp]:
            for c in meshcornerindices(i, Lx):
                if G.edges[c] is not None:
                    corners.add(c)
        path = [corners.pop()]
        while corners:
            v = G.edges[path[-1]]
            N = len(corners)
            while v:
                if v.y in corners:
                    path.append(v.y)
                    corners.remove(v.y)
                    v = None
                else:
                    v = v.nextedge
            if len(corners) == N:
                break
        if not len(corners):  # if path consumed corners
            v, p0 = G.edges[path[-1]], path[0]
            while v:
                if v.y == p0:  # path closes on itself
                    path.append(v.y)  # make path closed for next step
                    # Correct the orientation
                    medges = meshedges(meshholefromedge(mask, v.w),
                                       *mask.shape)
                    orient = 2*(medges.index(v.w) < 2) - 1
                    orient = (2*(path[-2] < path[-1]) - 1) * orient
                    loops.append(path if orient > 0 else path[::-1])
                    loopindices.append(indices[cflat==comp])
                    break
                else:
                    v = v.nextedge
    return loops, loopindices


class ResistorNetworkModel(ModelComponent):
    """
    An object for modeling arbitrary linear surface currents

    """
    def __init__(self, mask, kernel=None, phi_offset=None, gshape=None,
                 padding=None, **kwargs):
        """
        Initialize a resistor network.
        mask : Boolean or int array of shape (Ly, Lx) for determining
                which elements of mesh array should have loop currents
        kernel : instance of Kernel class for computing magnetic flux

        phi_offset : position of g-field corner in mask for aligning
                    the mask and the sample data

        gshape : (Ly, Lx) tuple of ints, shape of sample g-field

        padding : (py, px) tuple of ints, padding of sample g for model

        kwargs:
        resistors : array_like of shape ((Lx+1)*(Ly+1)) 
            resistances of mesh surrounding mask pixels
        deltaV :    (float) is the applied voltage
                    (to the top and bottom unless otherwise specified)
        electrodes : list of two integers [cathode, anode]. Default is
                     [0, Lx], i.e. the top two corners
                     Specify in original mask units.
        """
        self.mask = fillones(mask)  # Since single holes do not remove resistors
        super(ResistorNetworkModel, self).__init__(mask.shape, **kwargs)

        self.deltaV = kwargs.get('deltaV', 1.)
        self._setupGridsNumbers()

        self.kernel = kernel
        self.phi_offset = phi_offset if phi_offset is not None else [0, 0]
        self.gshape = gshape if gshape is not None else mask.shape
        self.padding = padding if padding is not None else [0, 0]
        self.resistors = kwargs.get('resistors', np.ones(self.N_hor+self.N_ver))
        self.electrodes = kwargs.get('electrodes', np.array([0, self.Lx]))

        # Build matrix coupling loop voltages
        row, col, data, self.G = self._make_meshandgraph()

        cathode, anode = self.electrodes
        self.vpath = self.G.findpath(cathode, anode, self.G.bfs(cathode))
        if not self.vpath:
            raise RuntimeError("No path exists between cathode and anode")
            
        self._topology = findtopology(self.mask, self.G)

        for loop in self._topology[0]:
            row, col, data = self._addloop(row, col, data, self.G, loop, 0.)
        row, col, data = self._addloop(row, col, data, self.G, self.vpath, 
                                       self.deltaV)
        self.R = coo_matrix(
            (data, (row, col)), (self.N_loops, self.N_loops)
        ).tocsc()

        # solve for loop currents
        self.solve()
        self.updateParams('J_ext', [1.])

    def __setstate__(self, d):
        self.__dict__ = d
        self._setupGridsNumbers()

    # ==============================
    #   square grid of resistors
    # ==============================

    def _setupGridsNumbers(self):
        #  Re-evaluate whether we need this complex interpolation stuff in here
        Ly, Lx = self.Ly, self.Lx
        self.N_nodes = (Lx+1)*(Ly+1)
        self.N_hor = self.N + Lx
        self.N_ver = self.N + Ly
        self._globalLoopField = np.zeros((Ly, Lx))

        self.maskflat = self.mask.flatten()

        # Only make if don't exist
        try:
            self.gfield
        except AttributeError:
            self.gfield = np.zeros(self.N)
            self.gfieldflat = self.gfield.flatten()
            self.gfieldpadded = np.zeros((2*Ly, 2*Lx))

        # Mesh coordinates
        self.meshIndex = (np.cumsum(self.maskflat)-1).astype('int')
        self.N_meshes = np.sum(self.mask>0)
        self.N_loops = self.N_meshes #  will add loops!
        self.v = np.zeros(self.N_loops)

    def _make_meshandgraph(self):
        """
        This function computes self.meshMatrix as a sparse matrix
        by loops over the meshes of a square grid. It also computes
        a dictionary of meshes whose values are the edges contained
        and a dictionary of edges whose values are the meshes which
        contain them.
        """
        row, col, data = [], [], []
        neigh = [[-1, 0], [0, 1], [1, 0], [0, -1]]  # leftdownrightup
        Lx, Ly, N_hor = self.Lx, self.Ly, self.N_hor
        self.meshEdges = defaultdict(list)   # {'mesh': [list of edges]}
        self.edgeMeshes = defaultdict(list)  # {'edge': [list of meshes]}
        G = Graph((Lx + 1) * (Ly + 1))

        meshes = np.arange(self.N)[self.maskflat > 0]
        for i in meshes:
            mesh = self.meshIndex[i]  # meshes don't cound empty spots
            mesh_edges = meshedges(i, *self.mask.shape)

            self.meshEdges[mesh] = mesh_edges
            row.append(mesh)
            col.append(mesh)
            data.append(self.resistors[mesh_edges].sum())

            ul, ll, lr, ur = meshcornerindices(i, self.Lx)
            G.insert(ul, ur, i)  # w = resistor label
            G.insert(ul, ll, i + N_hor + i//Lx)

            [self.edgeMeshes[edge].append(mesh) for edge in mesh_edges]

            ix, iy = i % Lx, i // Lx
            if (ix == Lx - 1) or not (i+1 in meshes):  # right edge
                G.insert(ur, lr, i + N_hor + i//Lx + 1)
                
            if (iy == Ly - 1) or not (i+Lx in meshes):  # bottom edge
                G.insert(ll, lr, i + Lx)

            if ix == Lx - 1 and iy < Ly - 1:  # right edge
                nextmeshes, nextedges = [i + Lx], [mesh_edges[1]]
            elif ix < Lx - 1 and iy == Ly - 1:  # bottom edge
                nextmeshes, nextedges = [i + 1], [mesh_edges[2]]
            elif ix < Lx - 1 and iy < Ly - 1:
                nextmeshes, nextedges = [i + 1, i + Lx], mesh_edges[1:3]
            else:
                nextmeshes, nextedges = [], []
                
            for j, edge in zip(nextmeshes, nextedges):
                jx, jy = j % Lx, j // Lx
                if j in meshes:
                    row.append(mesh)
                    col.append(self.meshIndex[j])
                    data.append(-1 * self.resistors[edge])
                    # symmetric!
                    row.append(self.meshIndex[j])
                    col.append(mesh)
                    data.append(-1 * self.resistors[edge])
            
        self.N_currents = G.ne
        return row, col, data, G

    def _addloop(self, row, col, data, G, path, voltage):
        edgepath = nodetoedgepath(path, G)

        # add the loop voltage and edges
        self.v = np.concatenate([self.v, [voltage]])
        self.N_loops += 1
        self.meshEdges[self.N_loops-1] = edgepath

        for edge, n1, n2 in zip(edgepath, path[:-1], path[1:]):
            for m in self.edgeMeshes[edge]:
                if m >= self.N_meshes:  # find grid mesh in mask hole
                    medges = meshedges(meshholefromedge(self.mask, edge),
                                       *self.mask.shape)
                else:
                    medges = self.meshEdges[m]

                # right-hand-rule and path direction
                orient = 2*(medges.index(edge) < 2) - 1
                orient = (2 * (n1 < n2) - 1) * orient

                row.append(m)
                col.append(self.N_loops - 1)
                data.append(orient * self.resistors[edge])
                # symmetric!
                row.append(self.N_loops - 1)
                col.append(m)
                data.append(orient * self.resistors[edge])
            
            self.edgeMeshes[edge].append(self.N_loops-1)

        row.append(self.N_loops - 1)
        col.append(self.N_loops - 1)
        data.append(self.resistors[edgepath].sum())
        return row, col, data

    def _loop_patches(self):
        """ 
        Convention: right-hand-rule, CCW currents are positive so that B-field
        is positive out of the page
        """
        self.gpatch = np.zeros(self.mask.shape, dtype='float')
        gpatchflat = self.gpatch.ravel()
        comps, num = label(1 - self.mask)
        cflat = comps.ravel()
        loopcurrents = self.i[self.N_meshes:]
        for inds, i in zip(self._topology[1], loopcurrents[:-1]):
            gpatchflat[inds] = i

        self.vloop = np.zeros_like(self.gpatch)
        vloopflat = self.vloop.ravel()
        epath = nodetoedgepath(self.vpath, self.G)

        p0, p1 = self.vpath[:2] 
        v = self.G.edges[p0]
        while v:
            if v.y == p1:
                break
            v = v.nextedge
        m = edgemeshes(v.w, *self.mask.shape)[0]
        orient = 2*(meshedges(m, *self.mask.shape).index(v.w) < 2) - 1
        orient = (2 * (p0 < p1) - 1) * orient

        queue = set([m])
        done = set([])
        while queue:
            n = queue.pop()
            vloopflat[n] = orient
            done.add(n)

            ny, nx = n//self.Lx, n%self.Lx
            for y, x in zip([1, -1, 0, 0], [0, 0, 1, -1]):
                my, mx = ny + y, nx + x
                if my < 0 or my == self.Ly or mx < 0 or mx == self.Lx:
                    continue  # outside domain
        
                m = my * self.Lx + mx
                if m in done:
                    continue  # don't reprocess points
                if not sharededge(m, n, *self.mask.shape) in epath:
                    queue.add(m)  # cannot cross loop path

        return self.gpatch + self.i[-1] * self.vloop
    
    # ==============================
    #   Calculate currents and fields
    # ==============================

    def solve(self):
        self.i = spsolve(self.R, self.v)
        self.gfieldflat[self.maskflat == 1] = self.i[:self.N_meshes].copy()
        self.gfield = self.gfieldflat.reshape(self.Ly, self.Lx)
        self._gnoloop = self.gfield.copy()
        self._gpatch = self._loop_patches()
        self.gfield += self._gpatch
        return False

    @property
    def _cutp(self):
        oy, ox = self.phi_offset
        Ly, Lx = self.gshape  # This is NOT self.Lx/self.Ly!
        py, px = self.padding
        return np.s_[oy-py:oy+Ly+py, ox-px:ox+Lx+px]

    @property
    def _cut(self):
        oy, ox = self.phi_offset
        Ly, Lx = self.gshape  # This is NOT self.Lx/self.Ly!
        return np.s_[oy:oy+Ly, ox:ox+Lx]

    @property
    def _nop(self):
        Ly, Lx = self.gshape  # This is NOT self.Lx/self.Ly!
        py, px = self.padding
        return np.s_[py:py+Ly, px:px+Lx]

    def updateParams(self, name, values):
        """
        This object formally relies only upon J_ext,
        all psf_params dependence is handled  by the kernel
        """
        if name == 'J_ext':
            self.J_ext = values[0]
        self._updateModel_g()
        self._updateField()
        self.ext_flux = self.J_ext * self._unitJ_flux[self._cut]

    def _updateField(self):
        if self.kernel:
            self._unitJ_flux = self.kernel.applyM(self.gfield).real
        else:
            self._unitJ_flux = np.zeros_like(self.mask, dtype='float')
            print('No kernel attribute\n')

    def _updateModel_g(self):
        """
        Compute gfield and its curvatures for fitting
        """
        J = self.J_ext
        cutp = self._cutp
        nop, cut = self._nop, self._cut
        self.ext_g = J * self.gfield[cutp]

    def computeGradients(self):
        if self.kernel:
            self._d_unitJ_flux = self.kernel.computeGradients(self.gfield)
            self.d_ext_flux = self.J_ext*self._d_unitJ_flux[self._cut]
        else:
            self.d_ext_flux = np.zeros_like(self.ext_flux, dtype='float')
        return self.d_ext_flux

    @property
    def currents(self):
        return curl(self.gfield)
