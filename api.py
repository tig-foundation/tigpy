from tigpy.data import *
from tigpy.utils import loadJson, minJsonDump
from urllib.parse import urlencode
try:
    from pyodide.http import pyfetch
    async def request(url, method='GET', headers=None, data=None):
        resp = await pyfetch(url, method=method, headers=headers, body=data)
        return resp.status, await resp.string()

except ImportError:
    import requests
    async def request(url, method='GET', headers=None, data=None):
        resp = requests.request(method, url, headers=headers, data=data)
        return resp.status_code, resp.text

class API:
    def __init__(self, api_url: str, api_key: str = None):
        self.api_url = api_url
        self.api_key = api_key

    async def _call(self, query, respType, data=None, params=None):
        url = f"{self.api_url}/{query}"
        if params is not None:
            url += "?" + urlencode({k: v for k, v in asDict(params).items() if v is not None})
        status_code, text = await request(
            url=url,
            method='POST' if data else 'GET',
            headers={
                'X-Api-Key': self.api_key,
                'Content-Type': 'application/json'
            },
            data=minJsonDump(asDict(data)) if data else None
        )
        if status_code == 200:
            return Result(200, fromDict(respType, loadJson(text)))
        else:
            return Result(status_code, text)

    async def getAlgorithms(
        self,
        block: int = None,
        round: int = None,
        player_id: str = None,
        challenge_id: str = None,
    ) -> GetAlgorithmsResp:
        return await self._call(
            query=f"getAlgorithms",
            respType=GetAlgorithmsResp,
            params=GetAlgorithmsReq(
                block=block,
                round=round,
                player_id=player_id,
                challenge_id=challenge_id,
            )
        )

    async def getBenchmarks(
        self,
        block: int = None,
        player_id: str = None,
        challenge_id: str = None,
        algorithm_id: str = None,
    ) -> GetBenchmarksResp:
        return await self._call(
            query=f"getBenchmarks",
            respType=GetBenchmarksResp,
            params=GetBenchmarksReq(
                block=block,
                player_id=player_id,
                challenge_id=challenge_id,
                algorithm_id=algorithm_id,
            )
        )
    
    async def getBenchmarkData(self, benchmark_id: str) -> GetBenchmarkDataResp:
        return await self._call(
            query=f"getBenchmarkData",
            respType=GetBenchmarkDataResp,
            params=GetBenchmarkDataReq(benchmark_id=benchmark_id)
        )

    async def getBlock(
        self, 
        block: int = None, 
        round: int = None
    ) -> GetBlockResp:
        return await self._call(
            query=f"getBlock",
            respType=GetBlockResp,
            params=GetBlockReq(block=block, round=round)
        )

    async def getFrontierPoints(
        self,
        block: int = None,
        round: int = None,
        challenge_id: str = None,
    ) -> GetFrontierPointsResp:
        return await self._call(
            query=f"getFrontierPoints",
            respType=GetFrontierPointsResp,
            params=GetFrontierPointsReq(
                block=block,
                round=round,
                challenge_id=challenge_id,
            )
        )

    async def getPlayers(
        self, 
        block: int = None, 
        round: int = None
    ) -> GetPlayersResp:
        return await self._call(
            query=f"getPlayers",
            respType=GetPlayersResp,
            params=GetPlayersReq(
                block=block,
                round=round
            )
        )

    async def submitBenchmark(
        self, 
        player_id: str,
        block_started: int,
        algorithm_id: str,
        challenge_id: str,
        difficulty: List[int],
        nonces: List[int],
        solution_ids: List[int],
        compute_times: List[int],
    ) -> SubmitBenchmarkResp:
        return await self._call(
            query="submitBenchmark", 
            respType=SubmitBenchmarkResp,
            data=SubmitBenchmarkReq(
                player_id=player_id,
                block_started=block_started,
                algorithm_id=algorithm_id,
                challenge_id=challenge_id,
                difficulty=difficulty,
                nonces=nonces,
                solution_ids=solution_ids,
                compute_times=compute_times,
            )
        )

    async def submitProofs(
        self,
        benchmark_id: str,
        proofs: List[Proof],
    ) -> SubmitProofsResp:
        return await self._call(
            query="submitProofs", 
            respType=SubmitProofsResp,
            data=SubmitProofsReq(
                benchmark_id=benchmark_id,
                proofs=proofs,
            )
        )

    async def submitAlgorithm(
        self, 
        algorithm_id: str,
        test_result_id: str,
        tx_hash: str,
    ) -> SubmitAlgorithmResp:
        return await self._call(
            query="submitAlgorithm", 
            respType=SubmitAlgorithmResp,
            data=SubmitAlgorithmReq(
                algorithm_id=algorithm_id,
                test_result_id=test_result_id,
                tx_hash=tx_hash
            )
        )
    
    async def testAlgorithm(self, algorithm_code: str) -> TestAlgorithmResp:
        return await self._call(
            query=f"testAlgorithm", 
            respType=TestAlgorithmResp,
            data=TestAlgorithmReq(algorithm_code=algorithm_code)
        )