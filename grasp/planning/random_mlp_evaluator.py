"""
RandomMlp-specific evaluator that works with state vectors directly.
"""

from .state_vector_evaluator import StateVectorEvaluator


class RandomMlpEvaluator(StateVectorEvaluator):
    """Evaluator specifically designed for random_mlp environment with state vectors."""

    def __init__(self, obs_0, obs_g, state_0, state_g, env, wm, frameskip,
                 seed, preprocessor, n_plot_samples, save_images=True):
        super().__init__(
            obs_0=obs_0, obs_g=obs_g, state_0=state_0, state_g=state_g,
            env=env, wm=wm, frameskip=frameskip, seed=seed,
            preprocessor=preprocessor, n_plot_samples=n_plot_samples,
            save_images=save_images,
            goal_threshold=wm.goal_threshold,
            env_name="RandomMlp",
        )

    def _extract_viz_frames(self, img_tensors):
        """RandomMlp world model returns an extra batch dimension."""
        return img_tensors[0]
