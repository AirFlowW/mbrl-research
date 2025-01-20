import matplotlib.pyplot as plt
import mbrl.diagnostics.test_data.viz.viz_utils as viz_utils

def viz_ensemble(ensemble_model, dataloader, stdevs = 1., title = None, save_path=None):
    plt.figure(figsize=(10, 6))
    ensemble_model.set_thompson_sampling_active()
    ensemble_model.propagation_method = "all_thompson_heads"
    Xp, Y_mean, Y_stdev = viz_utils.get_viz_pred(ensemble_model, dataloader, stdevs, title, save_path)
    viz_utils.plot_thompson_ensemble_pred(ensemble_model, Y_mean, Y_stdev, Xp, stdevs)
    viz_utils.plot_axis_and_save(dataloader, title, save_path, legend=True)