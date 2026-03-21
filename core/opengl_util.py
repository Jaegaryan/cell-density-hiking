from types import SimpleNamespace

import numpy as np
from OpenGL import GL


def create_ssbo(data, binding=None):
    ssbo = np.zeros(1, dtype=np.uint32)
    GL.glCreateBuffers(1, ssbo)
    ssbo = ssbo[0]
    GL.glNamedBufferData(ssbo, data.nbytes, data, GL.GL_DYNAMIC_DRAW)
    if binding is not None:
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, binding, ssbo)

    return ssbo


def create_ssbo_empty(nbytes, binding=None):
    ssbo = np.array((1,), dtype=np.uint32)
    GL.glCreateBuffers(1, ssbo)
    ssbo = ssbo[0]
    GL.glNamedBufferData(ssbo, nbytes, None, GL.GL_DYNAMIC_DRAW)
    if binding is not None:
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, binding, ssbo)

    return ssbo


def create_ssbo_full(value, nbytes, binding=None, dtype=np.uint32):
    ssbo = np.array((1,), dtype=np.uint32)
    GL.glCreateBuffers(1, ssbo)
    ssbo = ssbo[0]
    GL.glNamedBufferData(ssbo, nbytes, None, GL.GL_DYNAMIC_DRAW)
    if dtype == np.uint32:
        GL.glClearNamedBufferData(ssbo, GL.GL_R32UI, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, np.array([value], dtype=np.uint32))
    elif dtype == np.float32:
        GL.glClearNamedBufferData(ssbo, GL.GL_R32F, GL.GL_RED, GL.GL_FLOAT, np.array([value], dtype=np.float32))
    elif dtype == np.uint64:
        GL.glClearNamedBufferData(ssbo, GL.GL_R64UI, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT64_ARB, np.array([value], dtype=np.float64))
    else:
        raise ValueError(f"Unsupported dtype {dtype}")
        return None

    if binding is not None:
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, binding, ssbo)

    return ssbo


def create_ubo(data, binding=None):
    ubo = np.zeros(1, dtype=np.uint32)
    GL.glCreateBuffers(1, ubo)
    ubo = ubo[0]
    GL.glNamedBufferData(ubo, data.nbytes, data, GL.GL_DYNAMIC_DRAW)
    if binding is not None:
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, binding, ubo)

    return ubo


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
    with open(filename.with_suffix('.glsl'), 'r') as f:
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
