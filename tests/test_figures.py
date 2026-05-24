import pytest
import os
from src.generate_publication_figures import (
    generate_fig1_top_k,
    generate_fig2_mrr,
    generate_fig3_exam,
    generate_fig4_runtime,
    generate_fig5_memory,
    generate_fig6_weight_evolution,
    generate_fig7_dependency_graph,
    generate_fig8_ablation_heatmap
)

def test_figures_methods_exist():
    """
    Verify that all 8 figure generation callables import properly.
    """
    assert callable(generate_fig1_top_k)
    assert callable(generate_fig2_mrr)
    assert callable(generate_fig3_exam)
    assert callable(generate_fig4_runtime)
    assert callable(generate_fig5_memory)
    assert callable(generate_fig6_weight_evolution)
    assert callable(generate_fig7_dependency_graph)
    assert callable(generate_fig8_ablation_heatmap)
