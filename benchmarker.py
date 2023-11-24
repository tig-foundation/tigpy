from tigpy.utils import base64Encode
from tigpy.data import Algorithm, Benchmark, Block, FrontierPoint, Player, Proof
from tigpy.verification import *
from tigpy.api import API, request
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, List
import abc
import asyncio

@dataclass
class QueryData:
    benchmarks: List[Benchmark]
    block: Block
    algorithms: List[Algorithm]
    frontier_points: List[FrontierPoint]
    players: List[Player]
    
@dataclass
class BenchmarkParams:
    challenge_id: str
    algorithm_id: str
    difficulty: List[int]
    duration: int

@dataclass
class BenchmarkModules:
    challenge_module: Any
    algorithm_module: Any

class BaseBenchmarker:
    def __init__(self, api_url: str, api_key: str, player_id: str, num_workers: int = 1):
        self._api = API(api_url, api_key)
        self._player_id = player_id
        self._loop_task = None
        self._running = False
        self._modules = {}
        self._num_workers = num_workers
        self._pending_proofs = {}
        self._next_proof_submission = datetime.now()

    async def _worker(self, worker_id: int, data: QueryData, params: BenchmarkParams, modules: BenchmarkModules, proofs: List[Proof]):
        Challenge = modules.challenge_module.Challenge
        Difficulty = modules.challenge_module.Difficulty
        solveChallenge = modules.algorithm_module.solveChallenge
        divisor = data.block.config.preimage_threshold.max
        preimage_threshold = next(
            c.preimage_threshold
            for c in data.block.config.challenges
            if c.id == params.challenge_id
        )
        nonce = md5Seed(f"{datetime.now()},{worker_id}")
        while True:
            try:
                nonce += 1
                await self._onWorkerStartAttempt(worker_id, nonce)
                start = datetime.now()
                seed = generateSeed(
                    player_id=self._player_id, 
                    random_hash=data.block.random_hash, 
                    algorithm_id=params.algorithm_id, 
                    challenge_id=params.challenge_id, 
                    difficulty=params.difficulty, 
                    nonce=nonce
                )
                await asyncio.sleep(0)
                challenge = Challenge.generateInstance(seed, Difficulty(*params.difficulty))
                await asyncio.sleep(0)
                solution, runtime_signature = generateRuntimeSignature(seed, solveChallenge, challenge)
                await asyncio.sleep(0)
                await self._onWorkerFinishAttempt(worker_id, nonce)
                if not challenge.verifySolution(solution):
                    continue
                await self._onWorkerSolution(worker_id, nonce)
                await asyncio.sleep(0)
                solution_base64 = base64Encode(minJsonDump(solution))
                solution_id = generateSolutionId(
                    nonce, 
                    solution_base64=solution_base64, 
                    runtime_signature=runtime_signature
                )
                if solution_id % divisor > preimage_threshold:
                    continue
                await self._onWorkerSolutionId(worker_id, nonce)
                p = Proof(
                    nonce=nonce,
                    runtime_signature=runtime_signature,
                    solution_base64=solution_base64,
                    solution_id=solution_id,
                )
                p.compute_time = int((datetime.now() - start).total_seconds() * 1000)
                proofs.append(p)
                await asyncio.sleep(0)
                if len(proofs) > 250: # rough estimate of request size reaching api limit..
                    break
            except Exception as e:
                await self._handleWorkerError(worker_id, e)
                await asyncio.sleep(1)

    async def _queryData(self) -> QueryData:
        results = await asyncio.gather(
            self._api.getBenchmarks(player_id=self._player_id),
            self._api.getBlock(),
            self._api.getAlgorithms(),
            self._api.getFrontierPoints(),
            self._api.getPlayers()
        )
        if results[0].status_code != 200:
            raise Exception(f"Failed to getBenchmarks: {results[0].data}")
        if results[1].status_code != 200:
            raise Exception(f"Failed to getBlock: {results[1].data}")
        if results[2].status_code != 200:
            raise Exception(f"Failed to getAlgorithms: {results[2].data}")
        if results[3].status_code != 200:
            raise Exception(f"Failed to getFrontierPoints: {results[3].data}")
        if results[4].status_code != 200:
            raise Exception(f"Failed to getPlayers: {results[4].data}")
        return QueryData(
            benchmarks=results[0].data.benchmarks,
            block=results[1].data.block,
            algorithms=results[2].data.algorithms,
            frontier_points=results[3].data.frontier_points,
            players=results[4].data.players
        )

    @abc.abstractmethod
    async def _onWorkerStartAttempt(self, worker_id: int, nonce: int):
        raise NotImplementedError

    @abc.abstractmethod
    async def _onWorkerFinishAttempt(self, worker_id: int, nonce: int):
        raise NotImplementedError

    @abc.abstractmethod
    async def _onWorkerSolution(self, worker_id: int, nonce: int):
        raise NotImplementedError

    @abc.abstractmethod
    async def _onWorkerSolutionId(self, worker_id: int, nonce: int):
        raise NotImplementedError

    @abc.abstractmethod
    async def _handleWorkerError(self, worker_id: int, error: Exception):
        raise NotImplementedError
    
    @abc.abstractmethod
    async def _handleBenchmarkerError(self, error: Exception):
        raise NotImplementedError
    
    @abc.abstractmethod
    async def _pickBenchmarkParams(self, data: QueryData) -> BenchmarkParams:
        raise NotImplementedError

    async def _setupModules(self, data: QueryData, params: BenchmarkParams) -> BenchmarkModules:
        if params.challenge_id not in self._modules:
            status_code, text = await request(challengeCodeURL(
                repo_url=data.block.config.algorithm_submissions.git_repo,
                branch=data.block.config.algorithm_submissions.git_branch, 
                challenge_id=params.challenge_id
            ))
            if status_code != 200:
                raise Exception(f"Failed to get challenge code: {text}")
            setupChallengeModule(params.challenge_id, text)
            self._modules[params.challenge_id] = importChallengeModule(params.challenge_id)
        
        key = (params.challenge_id, params.algorithm_id)
        if key not in self._modules:
            status_code, text = await request(algorithmCodeURL(
                repo_url=data.block.config.algorithm_submissions.git_repo,
                branch=data.block.config.algorithm_submissions.git_branch, 
                challenge_id=params.challenge_id,
                algorithm_id=params.algorithm_id
            ))
            if status_code != 200:
                raise Exception(f"Failed to get algorithm code: {text}")
            setupAlgorithmModule(*key, text)
            self._modules[key] = importAlgorithmModule(*key)

        return BenchmarkModules(
            challenge_module=self._modules[params.challenge_id],
            algorithm_module=self._modules[key],
        )

    async def _doBenchmark(
        self, 
        data: QueryData, 
        params: BenchmarkParams, 
        modules: BenchmarkModules, 
        datetime_end: datetime,
        proofs: List[Proof]
    ):
        workers = [
            asyncio.create_task(self._worker(i, data, params, modules, proofs)) 
            for i in range(self._num_workers)
        ]
        while datetime.now() < datetime_end:
            await asyncio.sleep(1)
        for w in workers:
            w.cancel()

    async def _doSubmitBenchmark(self, data: QueryData, params: BenchmarkParams, proofs: List[Proof]) -> Result:
        return await self._api.submitBenchmark(
            player_id=self._player_id,
            block_started=data.block.height,
            algorithm_id=params.algorithm_id,
            challenge_id=params.challenge_id,
            difficulty=params.difficulty,
            nonces=[p.nonce for p in proofs],
            solution_ids=[p.solution_id for p in proofs],
            compute_times=[p.compute_time for p in proofs],
        )
    
    async def _doSubmitProofs(self, benchmark: Benchmark) -> Result:
        proofs = self._pending_proofs.pop(benchmark.id)
        sampled_nonces = set(benchmark.sampled_nonces)
        return await self._api.submitProofs(
            benchmark_id=benchmark.id, 
            proofs=[
                p for p in proofs
                if p.nonce in sampled_nonces
            ]
        )

    async def _loop(self):
        last_height = -1
        while self._running:
            try:
                data = await self._queryData()
                if data.block.height > last_height:
                    last_height = data.block.height
                    for b in data.benchmarks:
                        if (
                            b.sampled_nonces is None or 
                            b.id not in self._pending_proofs
                        ):
                            continue
                        if (now := datetime.now()) < self._next_proof_submission:
                            await asyncio.sleep((self._next_proof_submission - now).total_seconds())
                        if (result := await self._doSubmitProofs(b)).status_code != 200:
                            raise Exception(f"Failed to submit proofs: {result.data}")
                        self._next_proof_submission = datetime.now() + timedelta(seconds=5.5)
                params = await self._pickBenchmarkParams(data)
                modules = await self._setupModules(data, params)
                proofs = []
                await self._doBenchmark(data, params, modules, datetime.now() + timedelta(seconds=params.duration), proofs=proofs)
                if len(proofs) == 0:
                    continue
                if (result := await self._doSubmitBenchmark(data, params, proofs)).status_code != 200:
                    raise Exception(f"Failed to submit benchmark: {result.data}")
                self._pending_proofs[result.data.benchmark_id] = proofs
            except Exception as e:
                await self._handleBenchmarkerError(e)
                await asyncio.sleep(30)

    async def start(self):
        if self._loop_task or self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        await self._loop_task
        self._loop_task = None