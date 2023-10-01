""" Several plots resulting from the hyperparameter optimization."""

import optuna
from src.utils.constants import FIGURES_DIR
import os


study_label = "sqlite:///" + os.path.join(FIGURES_DIR, "fc_hyperparam_opt.db")
study = optuna.load_study(study_name="fc_study", storage=study_label)

fig = optuna.visualization.plot_optimization_history(study)

#fig = optuna.visualization.plot_param_importances(study)

#fig = optuna.visualization.plot_slice(study)

#fig = optuna.visualization.plot_contour(study, params=['lr', 'hidden_dim'])

fig = optuna.visualization.plot_intermediate_values(study)

fig.show()