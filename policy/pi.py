class StateDepthPolicy(MultiInputActorCriticPolicy):

    def __init__(
            self,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Union[List[int], Dict[str, List[int]], List[Dict[str, List[int]]], None] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        with open("config/config.json", "r") as json_file:
            config = json.load(json_file)
        
        self.action_dist = make_beta_distribution(action_space, dist_kwargs=self.dist_kwargs)

        self.build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = state_depth_actor_critic(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            action_space_dim = 2
        )
    
    def build(self, lr_schedule: Schedule) -> None:

        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        self.alpha_net, self.beta_net = self.action_dist.proba_distribution_net(latent_dim_pi)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.alpha_net: 0.01,
                self.beta_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        alpha = self.alpha_net(latent_pi) + 1e-2
        beta = self.beta_net(latent_pi) + 1e-2

        return self.action_dist.proba_distribution(alpha, beta)
        
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        # actions_mean = distribution.mean()
        actions_mean = ((self.alpha_net(latent_pi) + 1e-2) / 
                        (self.alpha_net(latent_pi) + 1e-2 + self.beta_net(latent_pi) + 1e-2))
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob, actions_mean
    
    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        deterministic=True
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

def make_beta_distribution(
    action_space: spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    
    if dist_kwargs is None:
        dist_kwargs = {}
    return BetaDistributionAction(get_action_dim(action_space), **dist_kwargs)