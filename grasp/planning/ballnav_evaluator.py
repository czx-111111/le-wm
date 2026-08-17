"""
BallNav-specific evaluator that works with state vectors directly.
"""

from .state_vector_evaluator import StateVectorEvaluator


class BallNavEvaluator(StateVectorEvaluator):
    """Evaluator specifically designed for ballnav environment with state vectors."""

    def __init__(self, obs_0, obs_g, state_0, state_g, env, wm, frameskip,
                 seed, preprocessor, n_plot_samples, save_images=True):
        super().__init__(
            obs_0=obs_0, obs_g=obs_g, state_0=state_0, state_g=state_g,
            env=env, wm=wm, frameskip=frameskip, seed=seed,
            preprocessor=preprocessor, n_plot_samples=n_plot_samples,
            save_images=save_images, goal_threshold=0.5, env_name="BallNav",
        )
