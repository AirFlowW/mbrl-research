import matplotlib.pyplot as plt
import os

def visualize_data(
    data,
    save_dir='.',
    combined_filename='combined_visualization.png',
    obs_prefix='obs_dim_',
    rewards_filename='rewards.png'
):
    """
    Visualize real vs. model observations and rewards.

    Parameters:
    -----------
    data : tuple
        A tuple of (real_obses, real_rewards, model_obses, model_rewards, actions).
        - real_obses.shape  = (num_sequences, sequence_length, obs_dim)
        - real_rewards.shape = (num_sequences, sequence_length)
        - model_obses.shape = (num_sequences, sequence_length, obs_dim)
        - model_rewards.shape = (num_sequences, sequence_length)
        - actions is not used in this visualization but is kept in the data tuple
    save_dir : str
        Directory where plots will be saved.
    combined_filename : str
        Filename for the combined (multi-subplot) figure.
    obs_prefix : str
        Prefix for filenames of the individual observation dimension plots.
    rewards_filename : str
        Filename for the individual reward plot.
    """

    def plot_obs_dimension(ax, dim_idx):
        # Plot each observation dimension
        
            for seq_idx in range(num_sequences):
                # Plot the real observation for this dimension
                ax.plot(
                    real_obses[seq_idx, :, dim_idx],
                    color='black',
                    label='Real Obs' if seq_idx == 0 else None
                )
                # Plot the model observation for this dimension
                for num_model_idx in range(num_model_samples):
                    ax.plot(
                        model_obses[seq_idx, num_model_idx, :, dim_idx],
                        color='green',
                        label='Model Obs' if seq_idx == 0 and num_model_idx == 0 else None
                    )
            ax.set_title(f'Observation Dimension {dim_idx}')
            ax.legend(loc='best')

    def plot_rewards(ax):
        """
        Plot real_rewards vs. model_rewards on a given axis (ax).
        """
        for seq_idx in range(num_sequences):
            # Real reward (one black line per sequence)
            ax.plot(
                real_rewards[seq_idx, :],
                color='black',
                label='Real Reward' if seq_idx == 0 else None
            )
            # Multiple model samples (red lines)
            for m_idx in range(num_model_samples):
                ax.plot(
                    model_rewards[seq_idx, m_idx, :],
                    color='red',
                    label='Model Reward' if seq_idx == 0 and m_idx == 0 else None
                )
        ax.set_title('Rewards')
        ax.legend(loc='best')

    # Unpack the data tuple
    real_obses, real_rewards, model_obses, model_rewards, actions = data
    # Shapes:
    # real_obses:   (num_sequences, sequence_length, obs_dim)
    # real_rewards: (num_sequences, sequence_length)
    # model_obses:  (num_sequences, num_model_samples, sequence_length, obs_dim)
    # model_rewards:(num_sequences, num_models, sequence_length)
    
    num_sequences, sequence_length, obs_dim = real_obses.shape
    num_sequences_m, num_model_samples, sequence_length_m, obs_dim_m = model_obses.shape

    # 1) Combined figure:
    fig, axes = plt.subplots(obs_dim + 1, 1, figsize=(10, 4*(obs_dim + 1)))
    # If we have only 1 dimension, axes will not be a list by default, so ensure it is:
    if obs_dim == 1:
        axes = [axes]  # for the single dimension
    else:
        axes = axes.flatten()
    for dim_idx in range(obs_dim):
        plot_obs_dimension(axes[dim_idx], dim_idx)

    # Plot rewards on the last axis
    plot_rewards(axes[-1])

    plt.tight_layout()
    combined_path = os.path.join(save_dir, combined_filename)
    plt.savefig(combined_path)
    plt.close(fig)
    print(f"Saved combined figure to: {combined_path}")

    # 2) Individual plots
    for dim_idx in range(obs_dim):
        fig_dim, ax_dim = plt.subplots(figsize=(8, 4))
        plot_obs_dimension(ax_dim, dim_idx)
        plt.tight_layout()
        dim_path = os.path.join(save_dir, f"{obs_prefix}{dim_idx}.png")
        plt.savefig(dim_path)
        plt.close(fig_dim)
        print(f"Saved individual dimension {dim_idx} figure to: {dim_path}")

    # Individual reward plot
    fig_r, ax_r = plt.subplots(figsize=(8, 4))
    plot_rewards(ax_r)
    plt.tight_layout()
    rewards_path = os.path.join(save_dir, rewards_filename)
    plt.savefig(rewards_path)
    plt.close(fig_r)
    print(f"Saved rewards figure to: {rewards_path}")