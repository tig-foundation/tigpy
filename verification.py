from tigpy.utils import AttrDict, md5Seed, minJsonDump, md5Hex, timeit
from tigpy.data import Result, Proof
from tigpy.errors import *
from datetime import datetime
import sys
import json
import subprocess
import time
import os
import random


@timeit
def challengeCodeURL(repo_url, branch, challenge_id):
    return repo_url.replace("github.com", "raw.githubusercontent.com").replace("challenges.git", f"challenges/{branch}/{challenge_id}/challenge.py")    
    
@timeit
def algorithmCodeURL(repo_url, branch, challenge_id, algorithm_id):
    algo_branch = algorithmBranch(branch, challenge_id, algorithm_id)
    return repo_url.replace("github.com", "raw.githubusercontent.com").replace("challenges.git", f"challenges/{algo_branch}/{challenge_id}/algorithms/{algorithm_id}.py")

def algorithmBranch(branch, challenge_id, algorithm_id):
    if branch == "main":
        return f"{challenge_id}_{algorithm_id}"
    else:
        return f"{branch}_{challenge_id}_{algorithm_id}"

def setupChallengeModule(challenge_id, challenge_code):
    os.makedirs(challenge_id, exist_ok=True)
    with open(f"{challenge_id}/__init__.py", "w") as f:
        pass
    with open(f"{challenge_id}/challenge.py", "w") as f:
        f.write(challenge_code)

def setupAlgorithmModule(challenge_id, algorithm_id, algorithm_code):
    os.makedirs(f"{challenge_id}/algorithms", exist_ok=True)
    with open(f"{challenge_id}/algorithms/__init__.py", "w") as f:
        pass
    with open(f"{challenge_id}/algorithms/{algorithm_id}.py", "w") as f:
        f.write(algorithm_code)

def importChallengeModule(challenge_id):
    return __import__(f"{challenge_id}.challenge").challenge

def importAlgorithmModule(challenge_id, algorithm_id):
    return getattr(
        __import__(f"{challenge_id}.algorithms.{algorithm_id}").algorithms,
        algorithm_id
    )

def generateSeed(player_id, random_hash, algorithm_id, challenge_id, difficulty, nonce):
    return md5Seed(minJsonDump(dict(
        player_id=player_id,
        random_hash=random_hash,
        algorithm_id=algorithm_id,
        challenge_id=challenge_id,
        difficulty=difficulty,
        nonce=nonce
    )))

def generateSolutionSignature(nonce, solution_base64, runtime_signature):
    return md5Seed(f"{nonce},{solution_base64},{runtime_signature}")

def generateBenchmarkId(player_id, random_hash, algorithm_id, challenge_id, difficulty, nonces, solution_signatures) -> str:
    return md5Hex(minJsonDump(dict(
        player_id=player_id,
        random_hash=random_hash,
        algorithm_id=algorithm_id,
        challenge_id=challenge_id,
        difficulty=difficulty,
        nonces=nonces,
        solution_signatures=solution_signatures
    )))

@timeit
def generateRuntimeSignature(seed: int, func, *args, **kwargs):
    random.seed(seed)
    import numpy as np
    d = AttrDict(
        signature=seed,
        next_update=0,
        event_count=0
    )

    def traceCalls(frame, event, args):
        if frame.f_code.co_filename == func.__code__.co_filename:
            frame.f_trace = traceOpcodes
            frame.f_trace_lines = False
            frame.f_trace_opcodes = True

    def traceOpcodes(frame, event, args):
        if d.next_update == d.event_count:
            k = random.choice(list(frame.f_locals))
            d.signature *= (randInt(frame.f_locals[k]) or 1) * (randInt(k) or 1)
            d.signature >>= max(0, d.signature.bit_length() - 32)              
            d.next_update += 500 + d.signature % 1024
        d.event_count += 1

    def randInt(o):
        if isinstance(o, bytes):
            return random.choice(o) if len(o) else 0
        elif isinstance(o, (str, int, float, bool)) or np.isscalar(o):
            return randInt(str(o).encode())
        elif isinstance(o, (list, tuple, np.ndarray)):
            return len(o) and randInt(random.choice(o))
        else:
            return 0

    sys.settrace(traceCalls)
    output = func(*args, **kwargs)
    sys.settrace(None)
    return output, d.signature

@timeit
def runVerificationWithTimeout(verification_code, max_seconds_to_verify, max_stderr_bytes) -> Result:
    file = f"verify.py"
    with open(file, "w") as f:
        f.write(verification_code)
    proc = subprocess.Popen(
        ["python", file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    start = datetime.now()
    timeout = False
    while proc.poll() is None:
        elapsed = (datetime.now() - start).total_seconds()
        if (timeout := elapsed > max_seconds_to_verify):
            proc.terminate()
        time.sleep(0.1)
    if proc.returncode != 0:
        raw = proc.stderr.read(max_stderr_bytes)
        parts = [
            "-----verify.py-----",
            verification_code,
            "-----STDERR-----",
            raw.decode()
        ]
        if len(raw) == max_stderr_bytes:
            parts.append("-----STDERR TRUNCATED-----")
        if timeout:
            parts.append(f"-----Code took longer than {max_seconds_to_verify}s to run.-----")
        return Result(400, "\n\n".join(parts))
    else:
        return Result(200, None)

@timeit
def verifyProof(challenge, solution, runtime_signature, solveChallenge) -> Result:
    try:
        if not challenge.verifySolution(solution):
            return Result(400, E049)
        actual_solution, actual_runtime_signature = generateRuntimeSignature(challenge.seed, solveChallenge, challenge)
        if runtime_signature != actual_runtime_signature:
            return Result(400, E050)
        if json.dumps(solution) != json.dumps(actual_solution):
            return Result(400, E051)
    except Exception as e:
        return Result(400, str(e))
    return Result(200, None)

@timeit
def verifyAlgorithm(challenge, solveChallenge) -> Result:
    solution, runtime_signature = generateRuntimeSignature(challenge.seed, solveChallenge, challenge)
    try:
        solution_dump = json.dumps(solution)
    except:
        return Result(400, "Failed to JSON serialize solution")
    solution2, runtime_signature2 = generateRuntimeSignature(challenge.seed, solveChallenge, challenge)
    try:
        solution_dump2 = json.dumps(solution2)
    except:
        return Result(400, "Failed to JSON serialize solution")
    if solution_dump != solution_dump2 or runtime_signature != runtime_signature2:
        return Result(400, "Mismatched output from re-running algorithm on the same challenge instance. Did you seed your random number generator with 'challenge.seed'?")
    return Result(200, solution)

@timeit
def sampleNonces(block, benchmark):
    import numpy as np
    np.random.seed(md5Seed(f"{block.random_hash},{benchmark.id}"))
    n = len(benchmark.nonces)
    return [
        benchmark.nonces[idx]
        for idx in np.random.choice(
            n, 
            replace=False,
            size=min(n, block.config.benchmark_submissions.max_samples)
        )
    ]