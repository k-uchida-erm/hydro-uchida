# =============================================================================
# analysis package
# =============================================================================
# このパッケージは、モデルの分析と可視化のためのツールを含みます：
# - check_model: モデルの構造とパラメータの確認
# - visualize: 学習結果の可視化
# ============================================================================= 

from .check_model import (
    check_pde_residual,
    check_boundary_conditions,
    check_initial_conditions,
    check_observation_data
)

from .visualize import (
    plot_1d_results,
    plot_2d_results,
    plot_loss_history,
    plot_boundary_conditions,
    plot_soil_parameters
)

__all__ = [
    'check_pde_residual',
    'check_boundary_conditions',
    'check_initial_conditions',
    'check_observation_data',
    'plot_1d_results',
    'plot_2d_results',
    'plot_loss_history',
    'plot_boundary_conditions',
    'plot_soil_parameters'
] 