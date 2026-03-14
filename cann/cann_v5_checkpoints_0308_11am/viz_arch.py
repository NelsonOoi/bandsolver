"""Visualize CANN v5 architectures with torchview."""

import torch
from torchview import draw_graph

from cann_v5 import DualGridCANN_v5, DualGridCANN_v5_cross

GRID = 32
COMMON = dict(
    expand_nested=True,
    hide_module_functions=False,
    hide_inner_tensors=False,
    depth=4,
)

# --- DualGridCANN_v5 ---
model = DualGridCANN_v5(grid_size=GRID)
graph = draw_graph(
    model,
    input_data=torch.randn(1, 1, GRID, GRID),
    graph_name="DualGridCANN_v5",
    graph_dir="LR",
    **COMMON,
)
graph.visual_graph.render("model_diagrams/cann_v5_dual_lr", format="png", cleanup=True)
print("Saved model_diagrams/cann_v5_dual_lr.png")

# --- DualGridCANN_v5_cross ---
model_x = DualGridCANN_v5_cross(grid_size=GRID)
graph_x = draw_graph(
    model_x,
    input_data=torch.randn(1, 1, GRID, GRID),
    graph_name="DualGridCANN_v5_cross",
    graph_dir="LR",
    **COMMON,
)
graph_x.visual_graph.render("model_diagrams/cann_v5_cross_lr", format="png", cleanup=True)
print("Saved model_diagrams/cann_v5_cross_lr.png")
