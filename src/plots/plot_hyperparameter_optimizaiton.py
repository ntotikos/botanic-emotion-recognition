""" Several plots resulting from the hyperparameter optimization."""

import optuna
from src.utils.constants import FIGURES_DIR
import os


study_label = "sqlite:///" + os.path.join(FIGURES_DIR, "fc_hyperparam_opt_96_trials.db")
study = optuna.load_study(study_name="fc_study", storage=study_label)

#print(study.trials)
fig = optuna.visualization.plot_optimization_history(study)

#fig = optuna.visualization.plot_param_importances(study)

#fig = optuna.visualization.plot_slice(study)

#fig = optuna.visualization.plot_contour(study, params=['lr', 'hidden_dim'])

#fig = optuna.visualization.plot_intermediate_values(study)
#fig.write_html("optimization_history_history.html")

#fig.show()

best_trial = study.best_trial

print(f"Best Trial: {best_trial.number}")
print(f"Value: {best_trial.value}")
print("Parameters:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")