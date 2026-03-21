# Cell Density Hiking (CDH)
This repository holds the implementation of the Cell Density Hiking (CDH) clustering algorithm.

# Citation
For details on how CDH works please refer to, or if you use this code please cite, our ARCS 2026 paper:  
"CDH: Accelerating Topological Mode-Seeking via Parallel Cell-Based Density Hiking", Proc. ARCS 2026.

# How To
### Hyperparameters
- max_points_per_node
- min_prominence or (TODO n_clusters)

## Interactive min_prominence
TODO [cdh_interactive.py](cdh_interactive.py)  
Visualize the point cloud and clustering, and set min_prominence interactively.

### Orbit Camera
- WASD or left-click to rotate
- JKLI or right-click to translate
- mousewheel to zoom
- Return to reset
- ESC to exit

## sklearn conform
TODO [cdh.py](cdh.py)
- .fit()
- .predict()
- .set_min_prominence() to update the clustering (only re-runs the threshold on the fitted merge tree)

## ARCS2026 Experiments
[cdh_experiments.py](cdh_experiments.py)  
This script can be used to reproduce the results of the ARCS2026 paper.  
It is similar to `cdh_interactive.py` but with additional visualizations of the algorithm steps, 
as well as buttons:
- Evaluate button to calculate scores
- Save button to save the data, predictions, and hyperparameters

# Requirements
This version uses:  
python 3.11.14  
numpy 2.2.6  
numba 0.61.2  
PyOpenGL 3.1.9  
PyOpenGL-accelerate 3.1.9  
glfw 2.9.0  
imgui 2.0.0  
scikit-learn 1.7.1  
scipy 1.15.3  

# License
GNU Lesser General Public License v2.1
