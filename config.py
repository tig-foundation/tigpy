from tigpy.data import *


V1 = Config(
    erc20=ERC20Config(
        rpc_url="https://mainnet.base.org",
        chain_id="0x2105",
        precision=int(1e18),
        token_address="base-gor:0x0C03Ce270B4826Ec62e7DD007f0B716068639F7B",
        minter_address="base-gor:0x30FeC1f3690F3207d1A239dB392f62C9CD1deF3F"
    ),
    benchmark_submissions=BenchmarkSubmissionsConfig(
        submission_delay_multiplier=3,
        max_samples=3,
        max_confirmations_per_block=200,
        max_mempool_size=500,
        lifespan_period=120
    ),
    solution_signature_threshold=SolutionSignatureThresholdConfig(
        min_value=0,
        max_value=int(1e9),
        min_delta=0,
        max_delta=int(2.5e6),
        rolling_average_lag_period=15,
        rolling_average_window=10,
        target_solutions_rate=20,
        target_error_multiplier=int(1e5)
    ),
    cutoff=CutoffConfig(
        avg_solutions_multiplier=2.5
    ),
    difficulty_frontiers=DifficultyFrontiersConfig(
        num_qualifiers_threshold=1000,
        max_difficulty_multiplier=2.0
    ),
    optimisable_proof_of_work=OptimisableProofOfWorkConfig(
        imbalance_multiplier=10
    ),
    verification=VerificationConfig(
        max_seconds=5,
        max_memory=1000,
        stderr_max_bytes=100000
    ),
    blocks=BlocksConfig(
        blocks_per_round=10080,
        seconds_between_blocks=60
    ),
    rewards=RewardsConfig(
      distribution=DistributionConfig(
          benchmarkers=0.85,
          innovators=0.15,
          implementations=1.0,
          breakthroughs=0.0
      ),
      schedule=[
            EmissionsConfig(
                block_reward=100,
                round_start=1,
            ),
            EmissionsConfig(
                block_reward=50,
                round_start=27,
            ),
            EmissionsConfig(
                block_reward=25,
                round_start=79,
            ),
            EmissionsConfig(
                block_reward=12.5,
                round_start=183,
            ),
            EmissionsConfig(
                block_reward=6.25,
                round_start=391,
            ),
            EmissionsConfig(
                block_reward=3.125,
                round_start=807,
            ),
            EmissionsConfig(
                block_reward=0,
                round_start=1639,
            )
      ]
    ),
    algorithm_submissions=AlgorithmSubmissionsConfig(
        submission_fee=1000000000000000,
        burn_address="0x0000000000000000000000000000000000000000",
        adoption_threshold=0.25,
        merge_points_threshold=5040,
        push_delay=3,
        git_branch="main",
        git_repo="https://github.com/tig-foundation/challenges.git"
    ),
    challenges=[
        ChallengeConfig(
            id="c001_satisfiability",
            difficulty_parameters=[
                DifficultyParameterConfig(
                    name="num_variables",
                    min_value=50,
                    max_value=2147483647
                ),
                DifficultyParameterConfig(
                    name="clauses_to_variables_percent",
                    min_value=300,
                    max_value=2147483647
                )
            ],
            solution_signature_threshold=int(1e9)
        ),
        ChallengeConfig(
            id="c002_vehicle_routing",
            difficulty_parameters=[
                DifficultyParameterConfig(
                    name="num_nodes",
                    min_value=40,
                    max_value=2147483647
                ),
                DifficultyParameterConfig(
                    name="better_than_baseline",
                    min_value=250,
                    max_value=999
                )
            ],
            solution_signature_threshold=int(1e9)
        ),
        ChallengeConfig(
            id="c003_knapsack",
            difficulty_parameters=[
                DifficultyParameterConfig(
                    name="num_items",
                    min_value=50,
                    max_value=2147483647
                ),
                DifficultyParameterConfig(
                    name="better_than_baseline",
                    min_value=10,
                    max_value=999
                )
            ],
            solution_signature_threshold=int(1e9),
        ),
    ],
)


TEST = Config(
    erc20=ERC20Config(
        rpc_url="https://goerli.base.org",
        chain_id="0x14a33",
        precision=int(1e18),
        token_address="base-gor:0xC8BBeAE27F1AE30908E60e69D6eF0B89929fe9AA",
        minter_address="base-gor:0xe0217560FEFb31ac7E6599cF1d844CE8732e0AB0"
    ),
    benchmark_submissions=BenchmarkSubmissionsConfig(
        submission_delay_multiplier=3,
        max_samples=3,
        max_confirmations_per_block=200,
        max_mempool_size=500,
        lifespan_period=120
    ),
    solution_signature_threshold=SolutionSignatureThresholdConfig(
        min_value=0,
        max_value=int(1e9),
        min_delta=0,
        max_delta=int(2.5e6),
        rolling_average_lag_period=15,
        rolling_average_window=10,
        target_solutions_rate=20,
        target_error_multiplier=int(1e5)
    ),
    cutoff=CutoffConfig(
        avg_solutions_multiplier=2.5
    ),
    difficulty_frontiers=DifficultyFrontiersConfig(
        num_qualifiers_threshold=1000,
        max_difficulty_multiplier=2.0,
    ),
    optimisable_proof_of_work=OptimisableProofOfWorkConfig(
        imbalance_multiplier=10
    ),
    verification=VerificationConfig(
        max_seconds=5,
        max_memory=1000,
        stderr_max_bytes=100000
    ),
    blocks=BlocksConfig(
        blocks_per_round=5,
        seconds_between_blocks=60
    ),
    rewards=RewardsConfig(
      distribution=DistributionConfig(
          benchmarkers=0.85,
          innovators=0.15,
          implementations=1.0,
          breakthroughs=0.0
      ),
      schedule=[
            EmissionsConfig(
                block_reward=100,
                round_start=1,
            ),
            EmissionsConfig(
                block_reward=50,
                round_start=27,
            ),
            EmissionsConfig(
                block_reward=25,
                round_start=79,
            ),
            EmissionsConfig(
                block_reward=12.5,
                round_start=183,
            ),
            EmissionsConfig(
                block_reward=6.25,
                round_start=391,
            ),
            EmissionsConfig(
                block_reward=3.125,
                round_start=807,
            ),
            EmissionsConfig(
                block_reward=0,
                round_start=1639,
            )
      ]
    ),
    algorithm_submissions=AlgorithmSubmissionsConfig(
        submission_fee=5000000000000000,
        burn_address="0x0000000000000000000000000000000000000000",
        adoption_threshold=0.25,
        merge_points_threshold=2,
        push_delay=3,
        git_branch="dev",
        git_repo="https://github.com/tig-foundation/challenges.git"
    ),
    challenges=[
        ChallengeConfig(
            id="c001_satisfiability",
            difficulty_parameters=[
                DifficultyParameterConfig(
                    name="num_variables",
                    min_value=50,
                    max_value=2147483647
                ),
                DifficultyParameterConfig(
                    name="clauses_to_variables_percent",
                    min_value=300,
                    max_value=2147483647
                )
            ],
            solution_signature_threshold=int(1e9)
        ),
        ChallengeConfig(
            id="c002_vehicle_routing",
            difficulty_parameters=[
                DifficultyParameterConfig(
                    name="num_nodes",
                    min_value=40,
                    max_value=2147483647
                ),
                DifficultyParameterConfig(
                    name="better_than_baseline",
                    min_value=250,
                    max_value=999
                )
            ],
            solution_signature_threshold=int(1e9)
        ),
        ChallengeConfig(
            id="c003_knapsack",
            difficulty_parameters=[
                DifficultyParameterConfig(
                    name="num_items",
                    min_value=50,
                    max_value=2147483647
                ),
                DifficultyParameterConfig(
                    name="better_than_baseline",
                    min_value=10,
                    max_value=999
                )
            ],
            solution_signature_threshold=int(1e9),
        ),
    ],
)