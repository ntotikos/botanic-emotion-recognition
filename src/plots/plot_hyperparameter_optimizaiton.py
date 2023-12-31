""" Several plots resulting from the hyperparameter optimization."""

import optuna
from src.utils.constants import FIGURES_DIR
import os


study_label = "sqlite:///" + os.path.join(FIGURES_DIR, "fc_hpo_baseline_6_normalized.db")
#study = optuna.load_study(study_name="fc_study", storage=study_label)
study = optuna.load_study(study_name="fc_baseline_6_normalized", storage=study_label)


#print(study.trials)
#fig = optuna.visualization.plot_optimization_history(study)

#fig = optuna.visualization.plot_param_importances(study)

#fig = optuna.visualization.plot_slice(study)

#fig = optuna.visualization.plot_contour(study, params=['lr', 'hidden_dim'])

#fig = optuna.visualization.plot_intermediate_values(study)
#fig.write_html("optimization_history_history.html")

#fig.show()

#f1_scores = [trial.user_attrs["f1_class"] for trial in study.trials]
#print(f1_scores)
#for trial in study.trials[:10]:  # Just inspecting the first 10 trials
#    print(trial.user_attrs["classification_report"])

best_trial = study.best_trial

print(f"Best Trial: {best_trial.number}")
print(f"Value: {best_trial.value}")
print("Parameters:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")


trials = study.trials

for trial in trials:
    print(f"Trial {trial.number}: Balanced Accuracy = {trial.value}")

print(f"Total number of trials: {len(trials)}")

stopped_trials = [trial for trial in trials if trial.state in [optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL]]

# Print information about stopped trials
for trial in stopped_trials:
    print(f"Trial {trial.number}: State = {trial.state}")