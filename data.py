from typing import List, Dict, Any, get_args, get_origin
from dataclasses import dataclass, fields, is_dataclass, asdict
from datetime import datetime

def asDict(d):
    if isinstance(d, list):
        return [asDict(_d) for _d in d]
    elif is_dataclass(d):
        return asdict(d)
    else:
        return d    

def fromDict(klass, d):
    if d is None:
        return None
    elif get_origin(klass) is list:
        t = get_args(klass)[0]
        return [fromDict(t, _d) for _d in d]
    elif get_origin(klass) is dict:
        t1, t2 = get_args(klass)
        return {fromDict(t1, k): fromDict(t2, v) for k, v in d.items()}
    elif is_dataclass(klass):
        fieldtypes = {f.name: f.type for f in fields(klass)}
        return klass(**{f: fromDict(fieldtypes[f], d[f]) for f in d})
    elif not isinstance(d, klass):
        return datetime.fromisoformat(d) if klass == datetime else klass(d)
    else:
        return d
    
@dataclass
class ERC20Config:
    rpc_url: str
    chain_id: str
    precision: int
    token_address: str
    minter_address: str

@dataclass
class BenchmarkSubmissionsConfig:
    submission_delay_multiplier: int
    max_samples: int
    max_confirmations_per_block: int
    max_mempool_size: int
    lifespan_period: int

@dataclass
class SolutionSignatureThresholdConfig:
    min_value: int
    max_value: int
    min_delta: int
    max_delta: int
    rolling_average_lag_period: int
    rolling_average_window: int
    target_solutions_rate: int
    target_error_multiplier: int

@dataclass
class CutoffConfig:
    avg_solutions_multiplier: float

@dataclass
class DifficultyFrontiersConfig:
    num_qualifiers_threshold: int
    max_difficulty_multiplier: float

@dataclass
class OptimisableProofOfWorkConfig:
    imbalance_multiplier: float

@dataclass
class BlocksConfig:
    blocks_per_round: int
    seconds_between_blocks: float

@dataclass
class AlgorithmSubmissionsConfig:
    submission_fee: int
    burn_address: str
    adoption_threshold: float
    merge_points_threshold: int
    push_delay: int
    git_branch: str
    git_repo: str

@dataclass
class VerificationConfig:
    max_seconds: int
    max_memory: int
    stderr_max_bytes: int

@dataclass
class EmissionsConfig:
    block_reward: float
    round_start: int

@dataclass
class DistributionConfig:
    benchmarkers: float
    innovators: float
    implementations: float
    breakthroughs: float

@dataclass
class RewardsConfig:
    distribution: DistributionConfig
    schedule: List[EmissionsConfig]

@dataclass
class DifficultyParameterConfig:
    name: str
    min_value: int
    max_value: int

@dataclass
class ChallengeConfig:
    id: str
    difficulty_parameters: List[DifficultyParameterConfig]
    solution_signature_threshold: int
    round_start: int = 1 # new
    weight: int = 1 # new
    solutions_rate: float = 0

@dataclass
class Config:
    erc20: ERC20Config
    benchmark_submissions: BenchmarkSubmissionsConfig
    solution_signature_threshold: SolutionSignatureThresholdConfig
    cutoff: CutoffConfig
    difficulty_frontiers: DifficultyFrontiersConfig
    optimisable_proof_of_work: OptimisableProofOfWorkConfig
    verification: VerificationConfig
    blocks: BlocksConfig
    rewards: RewardsConfig
    algorithm_submissions: AlgorithmSubmissionsConfig
    challenges: List[ChallengeConfig]

@dataclass
class Proof:
    nonce: int
    solution_base64: str
    runtime_signature: int
    solution_signature: int

@dataclass
class Benchmark:
    id: str
    player_id: str
    algorithm_id: str
    challenge_id: str
    difficulty: List[int]
    block_started: int
    block_submitted: int
    block_proofs_submitted: int
    block_active: int
    sampled_nonces: List[int]
    num_solutions: int
    num_qualifiers: int
    rejected: bool
    innovator_latest_reward: int
    benchmarker_latest_reward: int
    benchmarker_total_reward: int
    innovator_total_reward: int
    frontier_idx: int
    percent_qualifiers: float
    weight: float

@dataclass
class BenchmarkData:
    nonces: List[int]
    compute_times: List[int]
    solution_signatures: List[int]
    proofs: List[Proof]
    rejection_reason: str

@dataclass
class FrontierPoint:
    challenge_id: str
    frontier_idx: int
    difficulty: List[int]
    num_solutions: int
    num_qualifiers: int

@dataclass
class Algorithm:
    id: str
    challenge_id: str
    player_id: str
    block_submitted: int
    block_pushed: int
    block_merged: int
    banned: bool
    innovator_latest_reward: int
    innovator_round_reward: int
    percent_qualifiers: float
    weight: float
    adoption: float
    merge_points: int

@dataclass
class Player:
    id: str
    datetime_joined: datetime
    address: str
    banned: bool
    benchmarker_round_reward: int
    innovator_round_reward: int
    benchmarker_latest_reward: int
    innovator_latest_reward: int
    num_solutions: Dict[str, float]
    qualifiers_cutoff: int
    num_qualifiers: Dict[str, float]
    percent_qualifiers: Dict[str, float]
    avg_percent_qualifiers: float
    std_percent_qualifiers: float
    imbalance: float
    weight: int

@dataclass
class BlockData:
    benchmarks: List[Benchmark]
    algorithms: List[Algorithm]
    players: List[Player]

@dataclass
class Block:
    height: int
    round: int
    datetime_added: datetime
    random_hash: str
    config: Config
    data: BlockData = None

@dataclass
class TestResult:
    id: str = None
    player_id: str = None
    challenge_id: str = None
    algorithm_code: str = None
    stderr: str = None

@dataclass
class TestAlgorithmReq:
    algorithm_code: str

@dataclass
class TestAlgorithmResp:
    test_result_id: str


@dataclass
class SubmitProofsReq:
    benchmark_id: str
    proofs: List[Proof]

@dataclass
class SubmitProofsResp:
    ok: bool

@dataclass
class SubmitBenchmarkReq:
    player_id: str
    block_started: int
    algorithm_id: str
    challenge_id: str
    difficulty: List[int]
    nonces: List[int]
    solution_signatures: List[int]
    compute_times: List[int]

@dataclass
class SubmitBenchmarkResp:
    benchmark_id: str

@dataclass
class SubmitAlgorithmReq:
    algorithm_id: str
    test_result_id: str
    tx_hash: str

@dataclass
class SubmitAlgorithmResp:
    ok: bool

@dataclass
class RequestAPIKeyReq:
    signature: str
    address: str
    is_gnosis_safe: bool

@dataclass
class RequestAPIKeyResp:
    player_id: str
    api_key: str

@dataclass
class VerifyAlgorithmReq:
    challenge_id: str
    challenge_code: str
    algorithm_code: str
    difficulty: List[int]
    max_seconds_to_verify: int
    stderr_max_bytes: int

@dataclass
class VerifyAlgorithmResp:
    ok: bool

@dataclass
class VerifyProofsReq:
    player_id: str
    random_hash: str
    algorithm_id: str
    challenge_id: str
    difficulty: List[int]
    proofs: List[Proof]
    benchmark_id: str
    challenge_code: str
    algorithm_code: str
    max_seconds_to_verify: int
    stderr_max_bytes: int

@dataclass
class VerifyProofsResp:
    benchmark_id: str
    stderr: str

@dataclass
class ProcessVerificationResultReq:
    benchmark_id: str
    stderr: str

@dataclass
class ProcessVerificationResultResp:
    ok: bool

@dataclass
class Table:
    header: List[str]
    rows: List[list]        

@dataclass
class LastUpdated:
    datetime: datetime
    block: int
    round: int

@dataclass
class GetAlgorithmsReq:
    block: int = None
    round: int = None
    player_id: str = None
    challenge_id: str = None

@dataclass
class GetAlgorithmsResp:
    last_updated: LastUpdated
    table: Table

    @property
    def algorithms(self) -> List[Algorithm]:
        return [
            fromDict(Algorithm, dict(zip(self.table.header, row)))
            for row in self.table.rows
        ]

@dataclass
class GetFrontierPointsReq:
    block: int = None
    round: int = None
    challenge_id: str = None

@dataclass
class GetFrontierPointsResp:
    last_updated: LastUpdated
    table: Table

    @property
    def frontier_points(self) -> List[FrontierPoint]:
        return [
            fromDict(FrontierPoint, dict(zip(self.table.header, row)))
            for row in self.table.rows
        ]

@dataclass
class GetBlockReq:
    block: int = None
    round: int = None

@dataclass
class GetBlockResp:
    last_updated: LastUpdated
    block: Block

@dataclass
class GetPlayersReq:
    block: int = None
    round: int = None

@dataclass
class GetPlayersResp:
    last_updated: LastUpdated
    table: Table

    @property
    def players(self):
        return [
            fromDict(Player, dict(zip(self.table.header, row)))
            for row in self.table.rows
        ]

@dataclass
class GetBenchmarksReq:
    block: int = None
    player_id: str = None
    challenge_id: str = None
    algorithm_id: str = None

@dataclass
class GetBenchmarksResp:
    last_updated: LastUpdated
    table: Table

    @property
    def benchmarks(self):
        return [
            fromDict(Benchmark, dict(zip(self.table.header, row)))
            for row in self.table.rows
        ]

@dataclass
class GetBenchmarkDataReq:
    benchmark_id: str

@dataclass
class GetBenchmarkDataResp:
    last_updated: LastUpdated
    benchmark_data: BenchmarkData

@dataclass
class Result:
    status_code: int
    data: Any