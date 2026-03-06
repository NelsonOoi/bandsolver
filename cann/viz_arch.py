"""Visualize CANN v3 architectures with torchview."""

import torch
from torchview import draw_graph

from cann_v3 import DualGridCANN, DualGridCANN_cross

GRID = 32
COMMON = dict(
    expand_nested=True,
    hide_module_functions=False,
    hide_inner_tensors=False,
    depth=4,
)

# --- DualGridCANN ---
model = DualGridCANN(grid_size=GRID)
graph = draw_graph(
    model,
    input_data=torch.randn(1, 1, GRID, GRID),
    graph_name="DualGridCANN",
    **COMMON,
)
graph.visual_graph.render("model_diagrams/cann_v3_dual", format="png", cleanup=True)
print("Saved model_diagrams/cann_v3_dual.png")

# --- DualGridCANN_cross ---
model_x = DualGridCANN_cross(grid_size=GRID)
graph_x = draw_graph(
    model_x,
    input_data=torch.randn(1, 1, GRID, GRID),
    graph_name="DualGridCANN_cross",
    **COMMON,
)
graph_x.visual_graph.render("model_diagrams/cann_v3_cross", format="png", cleanup=True)
print("Saved model_diagrams/cann_v3_cross.png")
