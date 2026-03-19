import json
from pathlib import Path
from time import time, sleep
from types import SimpleNamespace

import imgui
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import make_moons, make_circles, make_blobs, make_swiss_roll
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import glfw

from OpenGL import GL
from OpenGL.GL.shaders import compileShader, compileProgram
from imgui.integrations.glfw import GlfwRenderer

from opengl.learnopengl.fps import FPS
from opengl.puzzle.cdh_3D import create_ssbo, create_ssbo_full, create_ssbo_empty
from opengl.puzzle.lin_alg import view_matrix_orbit, projection_matrix_perspective, orthonormalize
from opengl.puzzle.quaternion import quaternion_rotate
from opengl.puzzle.cdh_jit import get_scan_length, parents


def compile_glsl(filename, shaders=None):
    """
    Args:
        filename: str
            filename of glsl source
        shaders: SimpleNamespace
            output of commpile_glsl
    Notes:
        HEADER will be overwritten if shaders is provided
        code sections are split by prefix '// SOURCE '
            followed by included code sections, last of which is name of current code section
            code sections must be defined before use in other code sections
        bindings will be extracted from shaders['MACROS']
    """
    with open(f"{filename}{'' if filename.endswith('.glsl') else '.glsl'}", 'r') as f:
        lines = [line.rstrip() for line in f]

    shaders = {'HEADER': ''} if shaders is None else vars(shaders)  # lines before first code section, e.g. copyright
    current_shader = 'HEADER'
    code_sections = []
    for line in lines:
        if line.startswith('// SOURCE '):
            shaders[current_shader] = '\n'.join(code_sections).rstrip()
            *code_sections, current_shader = line.split(' ')[2:]  # ignore prefix
            code_sections = [shaders[i] for i in code_sections]
        else:
            code_sections.append(line)

    # save last shader
    shaders[current_shader] = '\n'.join(code_sections).rstrip()

    def _int(string):
        if string.endswith('u'):
            string = string[:-1]
        if string.startswith('0x'):
            return int(string, 16)
        else:
            return int(string)

    bindings = {key: _int(value) for key, value in [line.split(' ')[1:3] for line in shaders['MACROS'].splitlines()
                                                    if line and not line.startswith('//')]}

    return SimpleNamespace(**shaders) , SimpleNamespace(**bindings)


def create_ubo(data, binding=None):
    ubo = np.zeros(1, dtype=np.uint32)
    GL.glCreateBuffers(1, ubo)
    ubo = ubo[0]
    GL.glNamedBufferData(ubo, data.nbytes, data, GL.GL_DYNAMIC_DRAW)
    if binding is not None:
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, binding, ubo)

    return ubo


class Visualization:
    def __init__(self, x, y,
                 max_points_per_node, min_prominence=0.1, n_clusters=None,
                 save_path=None
                 ):
        s_total = time()

        # Data ---------------------------------------------------------------------------------------------------------
        n_points, d = x.shape

        if d == 2:
            xyzw = np.hstack([x, np.zeros([n_points, 1], dtype=np.float32), np.ones([n_points, 1], dtype=np.float32)])
        elif d == 3:
            xyzw = np.hstack([x, np.ones([n_points, 1], dtype=np.float32)])

        # Setup --------------------------------------------------------------------------------------------------------
        self.width, self.height = 1920, 1080
        self.aspect = self.height / self.width

        # GLFW
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        glfw.window_hint(glfw.SAMPLES, 4)
        self.window = glfw.create_window(self.width, self.height, 'CDH', None, None)
        # glfw.set_window_pos(self.window, 0, 32)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        SHADERS, _ = compile_glsl('blelloch.glsl')
        SHADERS, _ = compile_glsl('cdh.glsl', SHADERS)
        SHADERS, MACROS = compile_glsl(f'cdh_{d}D.glsl', SHADERS)

        # CDH ----------------------------------------------------------------------------------------------------------
        n_nodes = n_points
        scan_length = get_scan_length(n_nodes)

        blelloch_parameters = np.array([(1, 1, 0)], dtype=[('scan_length', 'u4'), ('offset', 'u4'), ('sum', 'u4')])
        tree_parameters = np.array([(n_points, max_points_per_node, 0, 1, 0)], dtype=[('n_points', 'u4'), ('max_points_per_node', 'u4'), ('level_a', 'u4'), ('level_b', 'u4'), ('depth', 'u4')])

        # nodes = np.zeros(n_nodes, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('w', 'f4'), ('radius', 'f4'),
        #                                  ('parent', 'u4'), ('childs', 'u4'), ('count', 'u4')]
        # neighbors = np.zeros([n_nodes, 6], dtype=np.uint32)  # north, east, south, west, up, down
        ssbo_blelloch =    create_ssbo(blelloch_parameters, binding=MACROS.SSBO_BLELLOCH)
        ssbo_tree =        create_ssbo(tree_parameters,     binding=MACROS.SSBO_TREE)
        ssbo_points =      create_ssbo(xyzw,                binding=MACROS.SSBO_POINTS)
        ssbo_nodes =       create_ssbo_full(0, n_nodes * 8 * 4, binding=MACROS.SSBO_NODES)
        ssbo_scan =        create_ssbo_full(0, scan_length * 4, binding=MACROS.SSBO_SCAN)
        ssbo_points_node = create_ssbo_empty(n_nodes * 4,            binding=MACROS.SSBO_POINTS_NODE)
        ssbo_neighbors =   create_ssbo_empty(n_nodes * 6 * 4, binding=MACROS.SSBO_NEIGHBORS)

        blelloch_up_step_program =     compileProgram(compileShader(SHADERS.BLELLOCH_UP_STEP,     GL.GL_COMPUTE_SHADER))
        blelloch_up_offset_program =   compileProgram(compileShader(SHADERS.BLELLOCH_UP_OFFSET,   GL.GL_COMPUTE_SHADER))
        blelloch_mid_step_program =    compileProgram(compileShader(SHADERS.BLELLOCH_MID_STEP,    GL.GL_COMPUTE_SHADER))
        blelloch_down_step_program =   compileProgram(compileShader(SHADERS.BLELLOCH_DOWN_STEP,   GL.GL_COMPUTE_SHADER))
        blelloch_down_offset_program = compileProgram(compileShader(SHADERS.BLELLOCH_DOWN_OFFSET, GL.GL_COMPUTE_SHADER))

        setup_tree_program =  compileProgram(compileShader(SHADERS.SETUP_TREE,  GL.GL_COMPUTE_SHADER))
        check_split_program = compileProgram(compileShader(SHADERS.CHECK_SPLIT, GL.GL_COMPUTE_SHADER))
        split_program =       compileProgram(compileShader(SHADERS.SPLIT,       GL.GL_COMPUTE_SHADER))
        neighbors_program =   compileProgram(compileShader(SHADERS.NEIGHBORS,   GL.GL_COMPUTE_SHADER))
        orthant_program =     compileProgram(compileShader(SHADERS.ORTHANT,     GL.GL_COMPUTE_SHADER))
        parameters_program =  compileProgram(compileShader(SHADERS.PARAMETERS,  GL.GL_COMPUTE_SHADER))

        # init root node
        GL.glUseProgram(setup_tree_program)
        GL.glDispatchCompute(1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        def blelloch(_scan_length):
            i = _scan_length
            # up sweep
            while i > 1:
                i //= 2
                # up step
                GL.glUseProgram(blelloch_up_step_program)
                GL.glDispatchCompute((i - 1) // 1024 + 1, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
                # up offset
                GL.glUseProgram(blelloch_up_offset_program)
                GL.glDispatchCompute(1, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

            # mid step
            GL.glUseProgram(blelloch_mid_step_program)
            GL.glDispatchCompute(1, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

            # down sweep
            while i < _scan_length:
                # down offset
                GL.glUseProgram(blelloch_down_offset_program)
                GL.glDispatchCompute(1, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
                # down step
                GL.glUseProgram(blelloch_down_step_program)
                GL.glDispatchCompute((i - 1) // 1024 + 1, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
                i *= 2

        while True:
            # calculate range in split to compute (has to be power of 2 for Blelloch)
            # smallest power of 2 larger than number of nodes in level
            i = 1  # scan_length
            while i < (tree_parameters['level_b'][0] - tree_parameters['level_a'][0]):
                i *= 2

            # check split -------------------------------------------------------------------
            GL.glUseProgram(check_split_program)
            GL.glDispatchCompute((i - 1) // 1024 + 1, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

            # prefix sum --------------------------------------------------------------------
            blelloch(i)

            # break condition
            GL.glGetNamedBufferSubData(ssbo_blelloch, 0, blelloch_parameters.nbytes, blelloch_parameters)
            if blelloch_parameters['sum'][0] == 0:
                break

            # split -------------------------------------------------------------------------
            GL.glUseProgram(split_program)
            GL.glDispatchCompute((i - 1) // 1024 + 1, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

            # neighbors ---------------------------------------------------------------------
            GL.glUseProgram(neighbors_program)
            GL.glDispatchCompute((i - 1) // 1024 + 1, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

            # sort points into child octant -------------------------------------------------
            GL.glUseProgram(orthant_program)
            GL.glDispatchCompute((n_points - 1) // 1024 + 1, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

            # set new level range -----------------------------------------------------------
            # gpu
            GL.glUseProgram(parameters_program)
            GL.glDispatchCompute(1, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            # cpu
            tree_parameters['level_a'] = tree_parameters['level_b']
            tree_parameters['level_b'] += blelloch_parameters['sum'] * 2**d

        # fix nodes length
        n_nodes = tree_parameters['level_b'][0]

        ssbo_nodes_copy = create_ssbo_empty(n_nodes * 8 * 4, binding=MACROS.SSBO_SORTED)
        GL.glCopyNamedBufferSubData(ssbo_nodes, ssbo_nodes_copy, 0, 0, n_nodes * 8 * 4)
        GL.glDeleteBuffers(1, ssbo_nodes)
        ssbo_nodes = ssbo_nodes_copy
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, MACROS.SSBO_NODES, ssbo_nodes)

        # Cell Density Hiking #########################################################################
        is_cell_program =             compileProgram(compileShader(SHADERS.IS_CELL,             GL.GL_COMPUTE_SHADER))
        compact_cell_program =        compileProgram(compileShader(SHADERS.COMPACT_CELL,        GL.GL_COMPUTE_SHADER))
        count_neighbors_program =     compileProgram(compileShader(SHADERS.COUNT_NEIGHBORS,     GL.GL_COMPUTE_SHADER))
        compact_neighbors_program =   compileProgram(compileShader(SHADERS.COMPACT_NEIGHBORS,   GL.GL_COMPUTE_SHADER))
        density_program =             compileProgram(compileShader(SHADERS.DENSITY,             GL.GL_COMPUTE_SHADER))
        max_neighbor_program =        compileProgram(compileShader(SHADERS.MAX_NEIGHBOR,        GL.GL_COMPUTE_SHADER))
        compact_peak_program =        compileProgram(compileShader(SHADERS.COMPACT_PEAK,        GL.GL_COMPUTE_SHADER))
        hike_program =                compileProgram(compileShader(SHADERS.HIKE,                GL.GL_COMPUTE_SHADER))
        count_borders_program =       compileProgram(compileShader(SHADERS.COUNT_BORDERS,       GL.GL_COMPUTE_SHADER))
        borders_program =             compileProgram(compileShader(SHADERS.BORDERS,             GL.GL_COMPUTE_SHADER))
        edges_program =               compileProgram(compileShader(SHADERS.EDGES,               GL.GL_COMPUTE_SHADER))
        compact_edges_program =       compileProgram(compileShader(SHADERS.COMPACT_EDGES,       GL.GL_COMPUTE_SHADER))
        count_sort_edges_program =    compileProgram(compileShader(SHADERS.COUNT_SORT_EDGES,    GL.GL_COMPUTE_SHADER))
        sort_edges_program =          compileProgram(compileShader(SHADERS.SORT_EDGES,          GL.GL_COMPUTE_SHADER))
        cluster_peaks_program =       compileProgram(compileShader(SHADERS.CLUSTER_PEAKS,       GL.GL_COMPUTE_SHADER))
        cluster_cells_program =       compileProgram(compileShader(SHADERS.CLUSTER_CELLS,       GL.GL_COMPUTE_SHADER))
        cluster_points_program =      compileProgram(compileShader(SHADERS.CLUSTER_POINTS,      GL.GL_COMPUTE_SHADER))
        reset_cluster_peaks_program = compileProgram(compileShader(SHADERS.RESET_CLUSTER_PEAKS, GL.GL_COMPUTE_SHADER))

        # check leaf/cell -----------------------------------------------
        # reuse scan array from tree
        scan_length = get_scan_length(n_nodes)  # new size scan_length = 2**? > n_nodes
        GL.glNamedBufferData(ssbo_scan, scan_length * 4, None, GL.GL_DYNAMIC_DRAW)  # set size (orphan buffer)
        GL.glClearNamedBufferData(ssbo_scan, GL.GL_R32UI, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT,
                                  np.array([0], dtype=np.uint32))  # set value
        blelloch_parameters['scan_length'] = scan_length  # to get sum at correct location
        GL.glNamedBufferData(ssbo_blelloch, blelloch_parameters.nbytes, blelloch_parameters, GL.GL_DYNAMIC_DRAW)

        # scan nodes for leafs/cells
        GL.glUseProgram(is_cell_program)
        GL.glDispatchCompute((n_nodes - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        blelloch(scan_length)

        # create cells ssbo
        GL.glGetNamedBufferSubData(ssbo_blelloch, 0, blelloch_parameters.nbytes, blelloch_parameters)
        n_cells = blelloch_parameters['sum'][0]

        # cells = np.empty(n_cells, dtype=[('node', 'u4'), ('density', 'u4'), ('neighbors', 'u4'), ('n_neighbors', 'u4'),
        #                                  ('max_neighbor', 'u4'), ('peak', 'u4'), ('cluster', 'u4')])
        ssbo_cells = create_ssbo_full(0, n_cells*7*4, binding=MACROS.SSBO_CELLS)

        # compact scan (set cell.node)
        GL.glUseProgram(compact_cell_program)
        GL.glDispatchCompute(scan_length // 1024, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        # store node->cell association
        ssbo_nodes_cells = create_ssbo_empty(n_nodes * 4, binding=MACROS.SSBO_NODES_CELLS)
        GL.glCopyNamedBufferSubData(ssbo_scan, ssbo_nodes_cells, 0, 0, n_nodes * 4)

        # neighbors -----------------------------------------------------
        scan_length = get_scan_length(n_cells)  # new size scan_length = 2**? > n_nodes
        GL.glNamedBufferData(ssbo_scan, scan_length * 4, None, GL.GL_DYNAMIC_DRAW)  # orphan buffer
        GL.glClearNamedBufferData(ssbo_scan, GL.GL_R32UI, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, np.array([0], dtype=np.uint32))
        blelloch_parameters['scan_length'] = scan_length  # to get sum at correct location
        GL.glNamedBufferData(ssbo_blelloch, blelloch_parameters.nbytes, blelloch_parameters, GL.GL_DYNAMIC_DRAW)

        GL.glUseProgram(count_neighbors_program)
        GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        blelloch(scan_length)

        # create neighbors array
        GL.glGetNamedBufferSubData(ssbo_blelloch, 0, blelloch_parameters.nbytes, blelloch_parameters)
        n_neighbors = blelloch_parameters['sum'][0]

        ssbo_cells_neighbors = create_ssbo_empty(n_neighbors * 4, binding=MACROS.SSBO_CELLS_NEIGHBORS)

        GL.glUseProgram(compact_neighbors_program)
        GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        # density -------------------------------------------------------
        GL.glUseProgram(density_program)
        GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        # max_neighbors and check peak --------------------------------------------
        # setup scan array and parameters for peaks
        scan_length = get_scan_length(n_cells)
        GL.glNamedBufferData(ssbo_scan, scan_length * 4, None, GL.GL_DYNAMIC_DRAW)  # orphan buffer
        GL.glClearNamedBufferData(ssbo_scan, GL.GL_R32UI, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT,
                                  np.array([0], dtype=np.uint32))
        blelloch_parameters['scan_length'] = scan_length  # to get sum at correct location
        GL.glNamedBufferData(ssbo_blelloch, blelloch_parameters.nbytes, blelloch_parameters, GL.GL_DYNAMIC_DRAW)

        GL.glUseProgram(max_neighbor_program)
        GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        # peaks
        blelloch(scan_length)

        # create peaks array (set cell.peak and peak.cell)
        GL.glGetNamedBufferSubData(ssbo_blelloch, 0, blelloch_parameters.nbytes, blelloch_parameters)
        n_peaks = blelloch_parameters['sum'][0]

        ssbo_peaks = create_ssbo_full(0, n_peaks * 5 * 4, binding=MACROS.SSBO_PEAKS)

        GL.glUseProgram(compact_peak_program)
        GL.glDispatchCompute(scan_length // 1024, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        # hike ----------------------------------------------------------
        GL.glUseProgram(hike_program)
        GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        # build peak graph
        # borders -------------------------------------------------------
        # | peak_0                                         | peak_1                       |  # scan[peak]
        # | cell_0                     | cell_1            | cell_0   | cell_1            |  # cell.cluster
        # | border_0 border_1 border_2 | border_0 border_1 | border_0 | border_0 border_1 |  # i_border
        # count borders per cell and per peak
        scan_length = get_scan_length(n_peaks)
        GL.glNamedBufferData(ssbo_scan, scan_length * 4, None, GL.GL_DYNAMIC_DRAW)  # orphan buffer
        GL.glClearNamedBufferData(ssbo_scan, GL.GL_R32UI, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT,
                                  np.array([0], dtype=np.uint32))
        blelloch_parameters['scan_length'] = scan_length
        GL.glNamedBufferData(ssbo_blelloch, blelloch_parameters.nbytes, blelloch_parameters, GL.GL_DYNAMIC_DRAW)

        GL.glUseProgram(count_borders_program)
        GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        blelloch(scan_length)

        # create array for border information
        GL.glGetNamedBufferSubData(ssbo_blelloch, 0, blelloch_parameters.nbytes, blelloch_parameters)
        n_borders = blelloch_parameters['sum'][0]

        # borders = np.empty(n_borders, dtype=[('from', 'u4'), ('col', 'u4'), ('col_height', 'f4'), ('to', 'u4')])  # from=peak, col=cell, to=peak
        ssbo_borders = create_ssbo_full(0, n_borders * 4 * 4, binding=MACROS.SSBO_BORDERS)

        GL.glUseProgram(borders_program)
        GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        # edges ---------------------------------------------------------
        # find key col (highest border to each neighbor peak)
        GL.glUseProgram(edges_program)
        GL.glDispatchCompute((n_peaks - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        # ! set padded scan indexes to 0 (edges_program uses scan as input and output so cannot just orphan or set everything to 0)
        GL.glNamedBufferSubData(ssbo_scan, n_peaks * 4, (scan_length - n_peaks) * 4,
                                np.zeros((scan_length - n_peaks), dtype=np.uint32), GL.GL_DYNAMIC_DRAW)

        blelloch(scan_length)

        GL.glGetNamedBufferSubData(ssbo_blelloch, 0, blelloch_parameters.nbytes, blelloch_parameters)
        n_edges = blelloch_parameters['sum'][0]

        # edges = np.empty(n_edges, dtype=[('from', 'u4'), ('col', 'u4'), ('col_height', 'f4'), ('to', 'u4')])
        ssbo_edges = create_ssbo_full(0, n_edges * 4 * 4, binding=MACROS.SSBO_EDGES)

        GL.glUseProgram(compact_edges_program)
        GL.glDispatchCompute((n_peaks - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        # Sort edges
        ssbo_tmp = create_ssbo_full(0, n_edges * 4, binding=MACROS.SSBO_TMP)

        GL.glGetNamedBufferSubData(ssbo_tree, 0, tree_parameters.nbytes, tree_parameters)
        max_density = (tree_parameters[0]['depth'] + 1) * (tree_parameters[0]['max_points_per_node'] + 1)
        scan_length = get_scan_length(max_density)
        GL.glNamedBufferData(ssbo_scan, scan_length * 4, None, GL.GL_DYNAMIC_DRAW)  # orphan buffer
        GL.glClearNamedBufferData(ssbo_scan, GL.GL_R32UI, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT,
                                  np.array([0], dtype=np.uint32))
        blelloch_parameters['scan_length'] = scan_length
        GL.glNamedBufferData(ssbo_blelloch, blelloch_parameters.nbytes, blelloch_parameters, GL.GL_DYNAMIC_DRAW)
        ssbo_sorted = create_ssbo_empty(n_edges * 4, MACROS.SSBO_SORTED)

        GL.glUseProgram(count_sort_edges_program)
        GL.glDispatchCompute((n_edges - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        blelloch(scan_length)

        GL.glUseProgram(sort_edges_program)
        GL.glDispatchCompute((n_edges - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        # Parents -------------------------------------------------------
        # CPU -------------------------------------------------
        edges = np.empty(n_edges, dtype=[('from', 'u4'), ('col', 'u4'), ('col_height', 'u4'), ('to', 'u4')])
        GL.glGetNamedBufferSubData(ssbo_edges, 0, edges.nbytes, edges)
        sorted = np.empty(n_edges, dtype=np.uint32)
        GL.glGetNamedBufferSubData(ssbo_sorted, 0, sorted.nbytes, sorted)
        peaks = np.empty(n_peaks, dtype=[('cell', 'u4'), ('height', 'u4'), ('prominence', 'u4'), ('parent', 'u4'), ('cluster', 'u4')])
        GL.glGetNamedBufferSubData(ssbo_peaks, 0, peaks.nbytes, peaks)

        cols = np.empty(n_peaks, dtype=np.uint32)  # for visualization
        peaks, cols, highest_peak = parents(edges[sorted], peaks, cols)
        GL.glNamedBufferData(ssbo_peaks, peaks.nbytes, peaks, GL.GL_DYNAMIC_DRAW)

        # Cluster -------------------------------------------------------
        ssbo_points_clusters = create_ssbo_empty(n_points * 4, binding=MACROS.SSBO_POINTS_CLUSTERS)
        max_prominence = peaks[highest_peak]['prominence']
        # set min_density (avoid normalization of prominence)
        min_density = np.array((max_prominence * min_prominence,), dtype=np.uint32)
        GL.glNamedBufferData(ssbo_tmp, 1 * 4, min_density, GL.GL_DYNAMIC_DRAW)

        # cluster peaks
        GL.glUseProgram(cluster_peaks_program)
        GL.glDispatchCompute((n_peaks - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # cluster cells
        GL.glUseProgram(cluster_cells_program)
        GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # cluster points
        GL.glUseProgram(cluster_points_program)
        GL.glDispatchCompute((n_points - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        points_clusters = np.empty(n_points, dtype=np.uint32)
        GL.glGetNamedBufferSubData(ssbo_points_clusters, 0, points_clusters.nbytes, points_clusters)

        GL.glFinish()
        t_total = time()
        time_total = t_total - s_total
        print('Time total:', time_total)

        print('n_points', n_points)
        print('n_nodes', n_nodes)
        print('n_cells', n_cells)
        print('n_borders', n_borders)
        print('n_edges', n_edges)
        print('n_peaks', n_peaks)

        # Save ----------------------------------------------------------------
        # if save_path is not None:
            # print('n_points', n_points)
            # print('n_nodes', n_nodes)
            # print('n_cells', n_cells)
            # print('n_borders', n_borders)
            # print('n_edges', n_edges)
            # print('n_peaks', n_peaks)
            #
            # z = LabelEncoder().fit_transform(points_clusters)
            # path = save_path / f'max_points_per_node_{max_points_per_node}'
            # path.mkdir(parents=True, exist_ok=True)
            # with open(path / 'parameters.json', 'w') as f:
            #     json.dump({'rng_seed': rng_seed,
            #                'n_points': n_points,
            #                'n_clusters': n_clusters,
            #                'cluster_std': 0.8,
            #                'max_points_per_node': max_points_per_node,
            #                'min_prominence': min_prominence,
            #                }, f, indent=4)
            # # np.save(path / 'x.npy', xyz)
            # # np.save(path / 'y.npy', y)
            # np.save(path / 'z_cdh.npy', z)
            # np.save(path / 't_cdh.npy', np.array(time_total))
            #
            # print('Saved', path)
            #
            # glfw.terminate()
            #
            # return

            # max_points_per_node -----------------------------------
            # z = LabelEncoder().fit_transform(points_clusters)
            # path = save_path / f'max_points_per_node_{max_points_per_node}'
            # path.mkdir(parents=True, exist_ok=True)
            # with open(path / 'parameters.json', 'w') as f:
            #     json.dump({
            #         'n_points': n_points,
            #         'n_clusters': n_clusters,
            #         'cluster_std': 0.8,
            #         'max_points_per_node': max_points_per_node,
            #         'min_prominence': min_prominence,
            #     }, f, indent=4)
            # # np.save(path / 'x.npy', xyz)
            # # np.save(path / 'y.npy', y)
            # np.save(path / 'z_cdh.npy', z)
            # np.save(path / 't_cdh.npy', np.array(time_total))
            #
            # print('Saved', path)
            #
            # return
        # ---------------------------------------------------------------------

        # Visualization ------------------------------------------------------------------------------------------------
        print('d', d)
        print('n_points', n_points)
        print('n_nodes', n_nodes)
        print('n_cells', n_cells)
        print('n_borders', n_borders)
        print('n_edges', n_edges)
        print('n_peaks', n_peaks)

        SHADERS, MACROS = compile_glsl(f'cdh_{d}D_visualization.glsl', SHADERS)

        # points
        visualize_points_program = compileProgram(compileShader(SHADERS.VISUALIZE_POINTS, GL.GL_COMPUTE_SHADER))
        ssbo_points_colors = create_ssbo(np.repeat(np.array([[0, 0, 1, 1]], dtype=np.float32), n_points * 4, axis=0))
        ssbo_points_sizes = create_ssbo(np.ones((n_points,), dtype=np.float32))
        GL.glUseProgram(visualize_points_program)
        GL.glDispatchCompute((n_points - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # cells
        visualize_cells_program = compileProgram(compileShader(SHADERS.VISUALIZE_CELLS, GL.GL_COMPUTE_SHADER))
        ssbo_cells_xyzw = create_ssbo_empty(n_cells * 8 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_XYZW)
        ssbo_cells_uv = create_ssbo_empty(n_cells * 12 * 2 * 4, binding=MACROS.SSBO_VISUALIZATION_UV)
        GL.glUseProgram(visualize_cells_program)
        GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # neighbors
        visualize_neighbors_program = compileProgram(compileShader(SHADERS.VISUALIZE_NEIGHBORS, GL.GL_COMPUTE_SHADER))
        ssbo_neighbors_xyzw = create_ssbo_empty((n_cells + n_neighbors) * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_XYZW)
        ssbo_neighbors_uv = create_ssbo_empty(n_neighbors * 2 * 4, binding=MACROS.SSBO_VISUALIZATION_UV)
        GL.glUseProgram(visualize_neighbors_program)
        GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # max_neighbor
        visualize_max_neighbor_program = compileProgram(compileShader(SHADERS.VISUALIZE_MAX_NEIGHBOR, GL.GL_COMPUTE_SHADER))
        ssbo_max_neighbor_xyzw = create_ssbo_empty(n_cells * 2 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_XYZW)
        ssbo_max_neighbor_uv = create_ssbo_empty(n_cells * 2 * 4, binding=MACROS.SSBO_VISUALIZATION_UV)
        GL.glUseProgram(visualize_max_neighbor_program)
        GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # hike (cell colors based on cell.peak)
        # visualize_hike_program = compileProgram(compileShader(SHADERS.VISUALIZE_HIKE, GL.GL_COMPUTE_SHADER))
        peaks_colors = np.concatenate([np.random.rand(n_peaks * 3).reshape(-1, 3).astype(np.float32), np.ones((n_peaks, 1), dtype=np.float32)], axis=1)
        peak_ids = np.argsort(peaks['prominence'])[::-1]
        color_i = peak_ids[0]
        peaks_colors[peak_ids[0]] = [0, 0.6, 0, 1]
        peaks_colors[peak_ids[1]] = [0, 0, 0.6, 1]
        ssbo_peaks_colors = create_ssbo(peaks_colors, binding=MACROS.SSBO_PEAKS_COLORS)
        # ssbo_cells_colors = create_ssbo_empty(n_cells * 12 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_COLORS)
        # GL.glUseProgram(visualize_hike_program)
        # GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
        # GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # borders
        # visualize_borders_program = compileProgram(compileShader(SHADERS.VISUALIZE_BORDERS, GL.GL_COMPUTE_SHADER))
        # ssbo_borders_xyzw = create_ssbo_empty(n_borders * 3 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_XYZW)
        # ssbo_borders_uv = create_ssbo_empty(n_borders * 2 * 2 * 4, binding=MACROS.SSBO_VISUALIZATION_UV)
        # ssbo_borders_colors = create_ssbo_empty(n_borders * 3 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_COLORS)
        # GL.glUseProgram(visualize_borders_program)
        # GL.glDispatchCompute((n_borders - 1) // 1024 + 1, 1, 1)
        # GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # edges
        visualize_edges_program = compileProgram(compileShader(SHADERS.VISUALIZE_EDGES, GL.GL_COMPUTE_SHADER))
        ssbo_edges_xyzw = create_ssbo_empty(n_edges * 3 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_XYZW)
        ssbo_edges_uv = create_ssbo_empty(n_edges * 2 * 2 * 4, binding=MACROS.SSBO_VISUALIZATION_UV)
        ssbo_edges_colors = create_ssbo_empty(n_edges * 3 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_COLORS)
        GL.glUseProgram(visualize_edges_program)
        GL.glDispatchCompute((n_edges - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # edges graph
        visualize_edges_graph_program = compileProgram(compileShader(SHADERS.VISUALIZE_EDGES_GRAPH, GL.GL_COMPUTE_SHADER))
        ssbo_edges_graph_xyzw = create_ssbo_empty(n_edges * 2 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_XYZW)
        ssbo_edges_graph_uv = create_ssbo_empty(n_edges * 1 * 2 * 4, binding=MACROS.SSBO_VISUALIZATION_UV)
        ssbo_edges_graph_colors = create_ssbo_empty(n_edges * 1 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_COLORS)
        GL.glUseProgram(visualize_edges_graph_program)
        GL.glDispatchCompute((n_edges - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # parents
        visualize_parents_program = compileProgram(compileShader(SHADERS.VISUALIZE_PARENTS, GL.GL_COMPUTE_SHADER))
        ssbo_parents_xyzw = create_ssbo_empty(n_peaks * 2 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_XYZW)
        ssbo_parents_uv = create_ssbo_empty(n_peaks * 2 * 4, binding=MACROS.SSBO_VISUALIZATION_UV)
        ssbo_parents_colors = create_ssbo_empty(n_peaks * 2 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_COLORS)
        GL.glUseProgram(visualize_parents_program)
        GL.glDispatchCompute((n_peaks - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # prominence
        visualize_prominence_program = compileProgram(compileShader(SHADERS.VISUALIZE_PROMINENCE, GL.GL_COMPUTE_SHADER))
        ssbo_prominence_xyzw = create_ssbo_empty(n_peaks * 2 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_XYZW)
        ssbo_prominence_uv = create_ssbo_empty(n_peaks * 2 * 4, binding=MACROS.SSBO_VISUALIZATION_UV)
        ssbo_prominence_colors = create_ssbo_empty(n_peaks * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_COLORS)
        GL.glUseProgram(visualize_prominence_program)
        GL.glDispatchCompute((n_peaks - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # parent cols
        # visualize_parent_cols_program = compileProgram(compileShader(SHADERS.VISUALIZE_PARENT_COLS, GL.GL_COMPUTE_SHADER))
        # ssbo_parent_cols = create_ssbo(cols, binding=SSBO_parent_cols)
        # ssbo_parent_cols_xyzw = create_ssbo_empty(n_peaks * 3 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_XYZW)
        # ssbo_parent_cols_uv = create_ssbo_empty(n_peaks * 2 * 2 * 4, binding=MACROS.SSBO_VISUALIZATION_UV)
        # ssbo_parent_cols_colors = create_ssbo_empty(n_peaks * 3 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_COLORS)
        # GL.glUseProgram(visualize_parent_cols_program)
        # GL.glDispatchCompute((n_peaks - 1) // 1024 + 1, 1, 1)
        # GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # clusters
        visualize_clusters_cells_program = compileProgram(compileShader(SHADERS.VISUALIZE_CLUSTERS_CELLS, GL.GL_COMPUTE_SHADER))
        ssbo_cluster_cells_colors = create_ssbo_empty(n_cells * 12 * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_COLORS)
        GL.glUseProgram(visualize_clusters_cells_program)
        GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        visualize_clusters_points_program = compileProgram(compileShader(SHADERS.VISUALIZE_CLUSTERS_POINTS, GL.GL_COMPUTE_SHADER))
        ssbo_cluster_points_colors = create_ssbo_empty(n_points * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_COLORS)
        GL.glUseProgram(visualize_clusters_points_program)
        GL.glDispatchCompute((n_points - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # labels
        visualize_labels_points_program = compileProgram(compileShader(SHADERS.VISUALIZE_LABELS_POINTS, GL.GL_COMPUTE_SHADER))
        ssbo_points_labels = create_ssbo(y, binding=MACROS.SSBO_POINTS_LABELS)
        ssbo_labels_points_colors = create_ssbo_empty(n_points * 4 * 4, binding=MACROS.SSBO_VISUALIZATION_COLORS)
        GL.glUseProgram(visualize_labels_points_program)
        GL.glDispatchCompute((n_points - 1) // 1024 + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # prominence histogram
        histogram = np.arcsinh(np.histogram(peaks['prominence'], bins=256)[0].astype(np.float32))
        # colors
        colors = np.empty(1, dtype=[('neighbors', 'f4', (4,)), ('max_neighbor', 'f4', (4,))])
        colors['neighbors'] = [1, 0, 0, 1]
        colors['max_neighbor'] = [0, 1, 0, 1]
        ssbo_colors = create_ssbo(colors, binding=MACROS.SSBO_COLORS)
        ssbo_neighbors_colors = create_ssbo(np.repeat([colors['neighbors'][0]], n_neighbors, axis=0))
        ssbo_max_neighbor_colors = create_ssbo(np.repeat([colors['max_neighbor'][0]], n_cells, axis=0))
        visualize_neighbors_colors_program = compileProgram(compileShader(SHADERS.VISUALIZE_NEIGHBORS_COLORS, GL.GL_COMPUTE_SHADER))
        visualize_max_neighbor_colors_program = compileProgram(compileShader(SHADERS.VISUALIZE_MAX_NEIGHBOR_COLORS, GL.GL_COMPUTE_SHADER))

        DRAW_POINTS, DRAW_LINES, DRAW_GRADIENT = 0, 1, 2
        self.drawcalls = [
            # (DRAW_POINTS, 'points', ssbo_points, None, ssbo_points_colors, ssbo_points_sizes, n_points),
            # (DRAW_POINTS, 'points labels', ssbo_points, None, ssbo_labels_points_colors, ssbo_points_sizes, n_points),
            (DRAW_POINTS, 'points clusters', ssbo_points, None, ssbo_cluster_points_colors, ssbo_points_sizes, n_points),
            # (DRAW_POINTS, 'points borders', ssbo_borders_xyzw, None, ssbo_borders_colors, ssbo_points_sizes, n_borders*3),
            (DRAW_POINTS, 'points edges', ssbo_edges_xyzw, None, ssbo_edges_colors, ssbo_points_sizes, n_edges*3),

            # (DRAW_LINES, 'cells', ssbo_cells_xyzw, ssbo_cells_uv, ssbo_cells_colors, n_cells*4),
            (DRAW_LINES, 'cells clusters', ssbo_cells_xyzw, ssbo_cells_uv, ssbo_cluster_cells_colors, None, n_cells * (12 if d == 3 else 4)),
            (DRAW_LINES, 'neighbors', ssbo_neighbors_xyzw, ssbo_neighbors_uv, ssbo_neighbors_colors, None, n_neighbors),
            (DRAW_LINES, 'max_neighbor', ssbo_max_neighbor_xyzw, ssbo_max_neighbor_uv, ssbo_max_neighbor_colors, None, n_cells),
            (DRAW_LINES, 'prominence', ssbo_prominence_xyzw, ssbo_prominence_uv, ssbo_prominence_colors, None, n_peaks),
            (DRAW_LINES, 'edges_graph', ssbo_edges_graph_xyzw, ssbo_edges_graph_uv, ssbo_edges_graph_colors, None, n_edges),

            (DRAW_GRADIENT, 'edges', ssbo_edges_xyzw, ssbo_edges_uv, ssbo_edges_colors, None, n_edges*2),
            # (DRAW_GRADIENT, 'borders', ssbo_borders_xyzw, ssbo_borders_uv, ssbo_borders_colors, None, n_borders*2),
            (DRAW_GRADIENT, 'parents', ssbo_parents_xyzw, ssbo_parents_uv, ssbo_parents_colors, None, n_peaks),
            # (DRAW_GRADIENT, 'parents cols', ssbo_parent_cols_xyzw, ssbo_parent_cols_uv, ssbo_parent_cols_colors, None, n_peaks*2),
        ]

        n_drawcalls = len(self.drawcalls)
        parameters = [np.array([(1, 1, 0, 0, 0.01)], dtype=[('scale_x', 'f4'), ('scale_y', 'f4'), ('scale_z', 'f4'), ('color_alpha', 'f4'), ('point_size', 'f4')]) for _ in range(n_drawcalls)]
        ubos_parameters = [create_ubo(parameters[i]) for i in range(n_drawcalls)]

        VIS_SHADERS, VIS_MACROS = compile_glsl('visualization.glsl')
        # [print(f"{k}\n{v}\n\n") for k, v in vars(VIS_SHADERS).items()]
        # [print(k, v) for k, v in  vars(VIS_MACROS).items()]
        # exit()

        # Camera ---------------------------------------------------------------------------------------------------------------
        self.projection_fov = 90 * np.pi / 180
        self.projection_near = 0.01
        self.projection_far = 64
        # self.projection_d = np.tan(self.projection_fov/2)

        self.view_target = np.array([0, 0, 0], dtype=np.float32)
        self.view_radius = np.array(1.5, dtype=np.float32)
        self.view_basis = np.eye(3, dtype=np.float32)


        self.speed_scroll = 16
        self.speed_drag = 0.5
        self.speed_rotation = 1
        self.speed_translation = 1

        self.time_last = 0
        self.time_delta = 0

        self.drag_start = (0, 0)


        # init camera position --------------------------------------
        # rotate Q
        # self.view_basis = orthonormalize(*quaternion_rotate(-self.speed_rotation * 3.5, self.view_basis[2], self.view_basis))
        # rotate S
        # self.view_basis = orthonormalize(*quaternion_rotate(self.speed_rotation * 0.9, self.view_basis[0], self.view_basis))


        camera_matrixes = np.zeros((4+4+4, 4), dtype=np.float32)  # model, view, projection
        ubo_camera = create_ubo(camera_matrixes, binding=VIS_MACROS.UBO_CAMERA)

        # Shaders --------------------------------------------------------------------------------------------------------------
        shader_program_points = compileProgram(compileShader(VIS_SHADERS.VERTEX_POINTS, GL.GL_VERTEX_SHADER),
                                               compileShader(VIS_SHADERS.FRAGMENT, GL.GL_FRAGMENT_SHADER))
        shader_program_lines = compileProgram(compileShader(VIS_SHADERS.VERTEX_LINES, GL.GL_VERTEX_SHADER),
                                              compileShader(VIS_SHADERS.FRAGMENT, GL.GL_FRAGMENT_SHADER))
        shader_program_gradient = compileProgram(compileShader(VIS_SHADERS.VERTEX_GRADIENT, GL.GL_VERTEX_SHADER),
                                                 compileShader(VIS_SHADERS.FRAGMENT_GRADIENT, GL.GL_FRAGMENT_SHADER))

        GL.glClearColor(1, 1, 1, 1)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glLineWidth(3)

        # ImGUI ----------------------------------------------------------------------------------------------------------------
        imgui.create_context()
        gui = GlfwRenderer(self.window)

        # Callbacks ------------------------------------------------------------------------------------------------------------
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)  # window resize
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)

        # Mainloop -------------------------------------------------------------------------------------------------------------
        fps = FPS()
        while not (glfw.window_should_close(self.window) or glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS):
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Input ------------------------------------------------------------------------------------------------------------
            time_current = glfw.get_time()
            self.time_delta = time_current - self.time_last
            self.time_last = time_current
            glfw.poll_events()
            self.process_keyboard_inputs()
            gui.process_inputs()

            # Camera -----------------------------------------------------------------------------------------------------------
            # TODO pre-compute projection * view
            camera_matrixes[:3, :3] = self.view_basis
            camera_matrixes[4:8] = view_matrix_orbit(self.view_target, self.view_radius, self.view_basis).T
            camera_matrixes[8:12] = projection_matrix_perspective(self.projection_fov, self.aspect, self.projection_near,
                                                                  self.projection_far)
            GL.glNamedBufferSubData(ubo_camera, 0, camera_matrixes.nbytes, camera_matrixes)

            # ImGui ------------------------------------------------------------------------------------------------------------
            imgui.new_frame()
            imgui.set_next_window_size(0, 0)
            imgui.set_next_window_position(4, 4, imgui.ONCE)
            imgui.begin('Visualization')

            imgui.text('Cluster color')
            imgui.text_wrapped(f'{peak_ids}')
            changed, value = imgui.input_int('i', color_i)
            if changed:
                color_i = value

            changed, values = imgui.color_edit4('color##clusters', *peaks_colors[color_i])
            if changed:
                peaks_colors[color_i] = values
                GL.glNamedBufferData(ssbo_peaks_colors, peaks_colors.nbytes, peaks_colors, GL.GL_DYNAMIC_DRAW)

                # set colors
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, MACROS.SSBO_VISUALIZATION_COLORS, ssbo_cluster_cells_colors)
                GL.glUseProgram(visualize_clusters_cells_program)
                GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, MACROS.SSBO_VISUALIZATION_COLORS, ssbo_cluster_points_colors)
                GL.glUseProgram(visualize_clusters_points_program)
                GL.glDispatchCompute((n_points - 1) // 1024 + 1, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

            imgui.text('Neighbors Color')
            changed, colors['neighbors'][0] = imgui.color_edit4('color##neighbors', *colors['neighbors'][0])
            if changed:
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, MACROS.SSBO_COLORS, ssbo_colors)
                GL.glNamedBufferSubData(ssbo_colors, 0, 4*4, colors['neighbors'][0])

                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, MACROS.SSBO_VISUALIZATION_COLORS, ssbo_neighbors_colors)
                GL.glUseProgram(visualize_neighbors_colors_program)
                GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)


            imgui.text('Max_Neighbors Color')
            changed, colors['max_neighbor'][0] = imgui.color_edit4('color##max_neighbor', *colors['max_neighbor'][0])
            if changed:
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, MACROS.SSBO_COLORS, ssbo_colors)
                GL.glNamedBufferSubData(ssbo_colors, 4*4, 4*4, colors['max_neighbor'][0])

                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, MACROS.SSBO_VISUALIZATION_COLORS, ssbo_max_neighbor_colors)
                GL.glUseProgram(visualize_max_neighbor_colors_program)
                GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

            for i, (_, name, *_) in enumerate(self.drawcalls):
                imgui.text(name)
                _, parameters[i]['scale_x']     = imgui.slider_float(f"Scale X##{name}",    parameters[i]['scale_x'][0], 0, 1)
                _, parameters[i]['scale_y']     = imgui.slider_float(f"Scale Y##{name}",    parameters[i]['scale_y'][0], 0, 1)
                _, parameters[i]['scale_z']     = imgui.slider_float(f"Scale Z##{name}",    parameters[i]['scale_z'][0], 0, 1)
                _, parameters[i]['color_alpha'] = imgui.slider_float(f"Alpha##{name}",      parameters[i]['color_alpha'][0], 0, 1)
                _, parameters[i]['point_size']  = imgui.slider_float(f"Point Size##{name}", parameters[i]['point_size'][0], 0.01, 16)

                GL.glNamedBufferSubData(ubos_parameters[i], 0, parameters[i].nbytes, parameters[i])

            # Evaluation --------------------------------------------------------------------------
            if imgui.button('Evaluate'):
                points_clusters = np.empty(n_points, dtype=np.uint32)
                GL.glGetNamedBufferSubData(ssbo_points_clusters, 0, points_clusters.nbytes, points_clusters)

                # rename predictions
                z = LabelEncoder().fit_transform(points_clusters)

                # stratified subsampling
                n_test = int(2**13)
                if n_points > n_test:
                    _, x_sample, _, y_sample, _, z_sample = train_test_split(x, y, z, test_size=n_test, stratify=z, random_state=42)
                else:
                    x_sample = x
                    y_sample = y
                    z_sample = z

                print('z')
                print('ARI', adjusted_rand_score(y_sample, z_sample))
                print('Silhouette', silhouette_score(x_sample, z_sample))
                print('Davis-Bouldin', davies_bouldin_score(x_sample, z_sample))
                print('Calinski-Harabasz', calinski_harabasz_score(x_sample, z_sample))
                # print('k-DBCV', DBCV_score(x_sample, z_sample))

                print('y')
                print('ARI', adjusted_rand_score(y_sample, y_sample))
                print('Silhouette', silhouette_score(x_sample, y_sample))
                print('Davis-Bouldin', davies_bouldin_score(x_sample, y_sample))
                print('Calinski-Harabasz', calinski_harabasz_score(x_sample, y_sample))
                # print('k-DBCV', DBCV_score(x_sample, y_sample))

            if imgui.button('Save'):
                points_clusters = np.empty(n_points, dtype=np.uint32)
                GL.glGetNamedBufferSubData(ssbo_points_clusters, 0, points_clusters.nbytes, points_clusters)

                z = LabelEncoder().fit_transform(points_clusters)
                path = save_path / f'{dataset_name}'
                path.mkdir(parents=True, exist_ok=True)
                with open(path / 'parameters.json', 'w') as f:
                    json.dump({
                        'n_points': n_points,
                        'n_clusters': n_clusters,
                        'cluster_std': 0.8,
                        'max_points_per_node': max_points_per_node,
                        'min_prominence': min_prominence,
                    }, f, indent=4)
                np.save(path / 'x.npy', x)
                np.save(path / 'y.npy', y)
                np.save(path / 'z_cdh.npy', z)
                np.save(path / 't_cdh.npy', np.array(time_total))

                print('Saved', path)

            # -------------------------------------------------------------------------------------
            imgui.set_next_window_size(0, 0)
            imgui.set_next_window_position(self.width-1024-16-4, 4, imgui.ONCE)

            imgui.begin('Hyperparameters')
            # min_prominence
            imgui.text('min_prominence')
            imgui.push_item_width(1024)
            changed, min_prominence = imgui.slider_float('', min_prominence, 0, 1)
            # if imgui.button('Cluster'):
            if changed:
                min_density = np.array((max_prominence * min_prominence,), dtype=np.uint32)
                GL.glNamedBufferData(ssbo_tmp, 1 * 4, min_density, GL.GL_DYNAMIC_DRAW)

                # reset peak.cluster
                GL.glUseProgram(reset_cluster_peaks_program)
                GL.glDispatchCompute((n_peaks - 1) // 1024 + 1, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

                # cluster peaks
                GL.glUseProgram(cluster_peaks_program)
                GL.glDispatchCompute((n_peaks - 1) // 1024 + 1, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
                # cluster cells
                GL.glUseProgram(cluster_cells_program)
                GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
                # cluster points
                GL.glUseProgram(cluster_points_program)
                GL.glDispatchCompute((n_points - 1) // 1024 + 1, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

                # set colors
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, MACROS.SSBO_VISUALIZATION_COLORS, ssbo_cluster_cells_colors)
                GL.glUseProgram(visualize_clusters_cells_program)
                GL.glDispatchCompute((n_cells - 1) // 1024 + 1, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, MACROS.SSBO_VISUALIZATION_COLORS, ssbo_cluster_points_colors)
                GL.glUseProgram(visualize_clusters_points_program)
                GL.glDispatchCompute((n_points - 1) // 1024 + 1, 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

            imgui.plot_histogram('', histogram, graph_size=(1024, 64))
            # add line at min_prominence
            _min, _max = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            _scale = _max.x - _min.x
            _x = _min.x + 1 + min_prominence * _scale
            drawlist = imgui.get_window_draw_list()
            drawlist.add_line(_x, _min.y-1, _x, _max.y-1, imgui.get_color_u32_rgba(1, 0, 0, 1), 1)

            imgui.end()

            imgui.end()
            imgui.end_frame()
            imgui.render()

            # Render -----------------------------------------------------------------------------------------------------------
            for (ubo_parameters), (p), (draw, name, ssbo_xyzw, ssbo_uv, ssbo_colors, ssbo_sizes, count) in zip(ubos_parameters, parameters, self.drawcalls):
                if p['color_alpha'][0] > 0:
                    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, VIS_MACROS.SSBO_VISUALIZATION_XYZW, ssbo_xyzw)
                    if ssbo_uv is not None: GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, VIS_MACROS.SSBO_VISUALIZATION_UV, ssbo_uv)
                    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, VIS_MACROS.SSBO_VISUALIZATION_COLORS, ssbo_colors)
                    if ssbo_sizes is not None: GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, VIS_MACROS.SSBO_VISUALIZATION_SIZES, ssbo_sizes)
                    GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, VIS_MACROS.UBO_PARAMETERS, ubo_parameters)

                    if draw == DRAW_POINTS:
                        GL.glUseProgram(shader_program_points)
                        GL.glDrawArrays(GL.GL_POINTS, 0, count)
                    if draw == DRAW_LINES:
                        GL.glUseProgram(shader_program_lines)
                        GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, count)
                    elif draw == DRAW_GRADIENT:
                        GL.glUseProgram(shader_program_gradient)
                        GL.glDrawArraysInstanced(GL.GL_LINES, 0, 2, count)

            gui.render(imgui.get_draw_data())

            glfw.set_window_title(self.window, f'FPS {fps.update():.0f}')
            glfw.swap_buffers(self.window)

        glfw.terminate()

    def framebuffer_size_callback(self, window, width, height):
        self.width = width
        self.height = height
        self.aspect = self.height / self.width
        GL.glViewport(0, 0, width, height)

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_1 and action == glfw.PRESS:
            self.drag_start = glfw.get_cursor_pos(window)
        if button == glfw.MOUSE_BUTTON_2 and action == glfw.PRESS:
            self.drag_start = glfw.get_cursor_pos(window)

    def cursor_pos_callback(self, window, x_pos, y_pos):
        if not imgui.get_io().want_capture_mouse:  # only process if drag not started in imgui window
            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_1) == glfw.PRESS:
                dx, dy = self.drag_start[0] - x_pos, self.drag_start[1] - y_pos
                self.view_basis = orthonormalize(*quaternion_rotate(
                    dy * self.speed_drag * self.time_delta, self.view_basis[0], self.view_basis))
                self.view_basis = orthonormalize(*quaternion_rotate(
                    dx * self.speed_drag * self.time_delta, self.view_basis[1], self.view_basis))
                self.drag_start = [x_pos, y_pos]
            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_2) == glfw.PRESS:
                dx, dy = self.drag_start[0] - x_pos, self.drag_start[1] - y_pos
                self.view_target -= self.view_basis[1] * dy * self.speed_translation * self.time_delta
                self.view_target += self.view_basis[0] * dx * self.speed_translation * self.time_delta
                self.drag_start = [x_pos, y_pos]

    def scroll_callback(self, window, x_offset, y_offset):
        """
        Args:
            window: GLFW window instance
            x_offset: 1.0 left, -1.0 right
            y_offset: 1.0 up, -1.0 down
        """
        if y_offset > 0:
            if self.speed_scroll * self.time_delta < self.view_radius + 1e-16:
                self.view_radius -= self.speed_scroll * self.time_delta
        if y_offset < 0:
            self.view_radius += self.speed_scroll * self.time_delta

    def process_keyboard_inputs(self):
        if not imgui.get_io().want_text_input:
            # view_basis
            if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
                self.view_basis = orthonormalize(*quaternion_rotate(
                    -self.speed_rotation * self.time_delta, self.view_basis[0], self.view_basis))
            if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
                self.view_basis = orthonormalize(*quaternion_rotate(
                    self.speed_rotation * self.time_delta, self.view_basis[0], self.view_basis))
            if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
                self.view_basis = orthonormalize(*quaternion_rotate(
                    -self.speed_rotation * self.time_delta, self.view_basis[1], self.view_basis))
            if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
                self.view_basis = orthonormalize(*quaternion_rotate(
                    self.speed_rotation * self.time_delta, self.view_basis[1], self.view_basis))
            if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
                self.view_basis = orthonormalize(*quaternion_rotate(
                    -self.speed_rotation * self.time_delta, self.view_basis[2], self.view_basis))
            if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
                self.view_basis = orthonormalize(*quaternion_rotate(
                    self.speed_rotation * self.time_delta, self.view_basis[2], self.view_basis))
            # view_target
            if glfw.get_key(self.window, glfw.KEY_J) == glfw.PRESS:
                self.view_target -= self.view_basis[0] * self.speed_translation * self.time_delta
            if glfw.get_key(self.window, glfw.KEY_L) == glfw.PRESS:
                self.view_target += self.view_basis[0] * self.speed_translation * self.time_delta
            if glfw.get_key(self.window, glfw.KEY_U) == glfw.PRESS:
                self.view_target += self.view_basis[1] * self.speed_translation * self.time_delta
            if glfw.get_key(self.window, glfw.KEY_O) == glfw.PRESS:
                self.view_target -= self.view_basis[1] * self.speed_translation * self.time_delta
            if glfw.get_key(self.window, glfw.KEY_K) == glfw.PRESS:
                self.view_target += self.view_basis[2] * self.speed_translation * self.time_delta
            if glfw.get_key(self.window, glfw.KEY_I) == glfw.PRESS:
                self.view_target -= self.view_basis[2] * self.speed_translation * self.time_delta
            # view_radius
            if glfw.get_key(self.window, glfw.KEY_Y) == glfw.PRESS:
                self.view_radius -= self.speed_translation * self.time_delta
            if glfw.get_key(self.window, glfw.KEY_H) == glfw.PRESS:
                self.view_radius += self.speed_translation * self.time_delta
            # reset
            if glfw.get_key(self.window, glfw.KEY_BACKSPACE) == glfw.PRESS:
                self.view_target = np.zeros([3], dtype=np.float32)
                self.view_radius = np.array(1, dtype=np.float32)
                self.view_basis = np.eye(3, dtype=np.float32)
            # shuffle
            if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
                self.shuffle = True
            elif glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.RELEASE:
                self.shuffle = False


def main():
    rng = np.random.default_rng()
    rng_seed = int(rng.integers(0, 2 ** 32 - 1))
    np.random.seed(rng_seed)
    rng = np.random.default_rng(rng_seed)

    from data import get_data
    n_clusters = 8
    x, y, n_points = get_data('blobs', n_features=3, centers=n_clusters, cluster_std=np.random.rand(n_clusters) * 0.8, random_state=rng_seed)
    # x, y, n_points = get_data('two_moons', noise=0.1)
    # x, y, n_points = get_data('circles', factor=0.5, noise=0.1)
    # x, y, n_points = get_data('chainlink', noise=(0.2, 0.2))
    # x, y, n_points = get_data('crown', n_blobs=16, noise_floor=0.01)


    # compile
    # Visualization(n_points=int(2**10), n_clusters=8, max_points_per_node=64, save_path=Path('data', 'experiments', 'init'))

    # quality
    Visualization(n_points=int(2**16), n_clusters=8, max_points_per_node=256+128, save_path=Path('data', 'experiments', 'quality'))

    # runtime
    # for iteration in [0, 1, 2, 3]:
    #     for n_clusters in [8, 256, 1024]:
    #         for i in [10]:
    #             sleep(1)
    #             Visualization(n_points=int(2**i),
    #                           n_clusters=n_clusters,
    #                           save_path=Path('data', 'experiments', f'iteration_{iteration}'))

    # max_points_per_node
    # for i in [
    #     0,
    #     1,
    #     2,
    #     3
    # ]:
    #     path = Path('data', 'experiments', f'iteration_{i}', 'n_clusters_256', 'n_points_16777216')
    #     xyz = np.load(path / 'x.npy')
    #     y = np.load(path / 'y.npy')
    #
    #     for max_points_per_node in [int(m) for m in 2**np.arange(4, 15)]:
    #         print(max_points_per_node)
    #         Visualization(n_points=int(2**24), n_clusters=256,
    #                       max_points_per_node=max_points_per_node, xyz=xyz, y=y,
    #                       save_path=Path('data', 'experiments', 'max_points_per_node', f'iteration_{i}'))




if __name__ == '__main__':
    main()
