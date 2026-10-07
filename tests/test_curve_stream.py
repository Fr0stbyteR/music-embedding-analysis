import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf
from music_annotation_backend.config import Settings
from music_annotation_backend.providers import ProviderRegistry
from music_annotation_backend.schemas import Asset, SemanticCurveRequest
from music_annotation_backend.interactive import relevance_curve
from music_annotation_backend.curve_stream import stream_curve
from test_api import client, wait_for_job


def fixture(tmp_path):
    path = tmp_path / 'audio.wav'
    sf.write(path, np.zeros(22 * 16000), 16000)
    asset = Asset(id=uuid4(), project_id=uuid4(), path=str(path), name='audio.wav', sample_rate=16000, channels=1, duration_seconds=22, samples=22*16000, content_hash='test', created_at=datetime.now(timezone.utc))
    providers = ProviderRegistry(Settings(data_root=tmp_path, model_root=tmp_path, vendor_root=tmp_path, auto_load_provider=''))
    asyncio.run(providers.get('mock').load('cpu', None, False))
    request = SemanticCurveRequest(keyword='piano', provider_id='mock', window_seconds=5, hop_seconds=1, timeline_duration_seconds=24)
    return asset, providers, request


def test_stream_chunks_precede_complete_and_match_legacy_result(tmp_path):
    asset, providers, request = fixture(tmp_path)
    async def collect():
        return [json.loads(event) async for event in stream_curve(asset, request, providers, tmp_path / 'cache')]
    events = asyncio.run(collect())
    assert [event['type'] for event in events] == ['started', 'chunk', 'chunk', 'chunk', 'complete']
    chunks = events[1:-1]
    assert [event['offset'] for event in chunks] == [0, 8, 16]
    assert all(event['total'] == 18 for event in chunks)
    points = [point for event in chunks for point in event['result']['points']]
    assert points == events[-1]['result']['points']
    assert all(not event['result']['metadata'] for event in chunks), 'partial chunks are not whole-song summaries'
    metadata = events[-1]['result']['metadata']
    assert metadata['statistics.0.count'] == len(points)
    assert metadata['statistics.0.mean'] == pytest.approx(np.mean([point['cosineSimilarity'] for point in points]))
    assert points[-1]['timeSeconds'] <= 24
    cached = relevance_curve(asset, request, providers, tmp_path / 'cache')
    assert cached.cached and cached.metadata == metadata


def test_cancelled_partial_is_not_cached(tmp_path):
    asset, providers, request = fixture(tmp_path)
    chunks = []
    with pytest.raises(RuntimeError, match='cancelled'):
        relevance_curve(asset, request, providers, tmp_path / 'cache', lambda *chunk: chunks.append(chunk), lambda: bool(chunks))
    assert len(chunks) == 1
    assert not list((tmp_path / 'cache').glob('*.json'))


def test_embedding_cache_is_reused_for_another_keyword_with_progress(tmp_path, monkeypatch):
    asset, providers, request = fixture(tmp_path)
    provider = providers.get('mock')
    calls = []
    def embed(self, windows):
        calls.append(len(windows))
        return np.ones((len(windows), 4), dtype=np.float32)
    monkeypatch.setattr(type(provider), 'embed_audio_for_text', embed)
    monkeypatch.setattr(type(provider), 'score_audio_embeddings_text', lambda self, embeddings, texts: np.full((len(embeddings), len(texts)), .6, dtype=np.float32))
    first = relevance_curve(asset, request, providers, tmp_path / 'cache')
    assert calls == [8, 8, 2]
    chunks = []
    second = relevance_curve(asset, request.model_copy(update={'keyword': 'violin'}), providers, tmp_path / 'cache', lambda offset, result, total: chunks.append((offset, len(result.points), total)))
    assert calls == [8, 8, 2], 'new text must not re-embed audio'
    assert chunks == [(0, 8, 18), (8, 8, 18), (16, 2, 18)]
    assert second.points == first.points and not second.cached


def test_disconnected_stream_stops_worker_and_does_not_cache(tmp_path, monkeypatch):
    import time
    asset, providers, request = fixture(tmp_path)
    request = request.model_copy(update={'hop_seconds': .1})
    provider = providers.get('mock')
    score = provider.score_audio_text
    monkeypatch.setattr(provider, 'score_audio_text', lambda *args: (time.sleep(.02), score(*args))[1])
    async def disconnect():
        stream = stream_curve(asset, request, providers, tmp_path / 'cache')
        assert json.loads(await anext(stream))['type'] == 'started'
        assert json.loads(await anext(stream))['type'] == 'chunk'
        await stream.aclose()
        await asyncio.sleep(.1)
    asyncio.run(disconnect())
    assert not list((tmp_path / 'cache').glob('*.json'))


def test_authenticated_stream_endpoint_cache_and_provider_errors(tmp_path):
    asset, _, _ = fixture(tmp_path)
    with client(tmp_path) as api:
        wait_for_job(api, api.post('/v1/providers/mock:load', json={'device': 'cpu'}).json()['jobId'])
        uploaded = api.post('/v1/interactive-assets', content=Path(asset.path).read_bytes(), headers={'X-File-Name': 'audio.wav'}).json()
        url = f"/v1/interactive-assets/{uploaded['id']}:relevance-curve-stream"
        request = {'keyword': 'piano', 'providerId': 'mock', 'windowSeconds': 5, 'hopSeconds': 1}
        assert api.post(url, json=request, headers={'Authorization': ''}).status_code == 401
        response = api.post(url, json=request)
        assert response.status_code == 200 and 'ndjson' in response.headers['content-type']
        assert json.loads(response.text.splitlines()[-1])['type'] == 'complete'
        cached = api.post(url, json=request)
        assert json.loads(cached.text.splitlines()[-1])['result']['cached']
        error = api.post(url, json={**request, 'providerId': 'not-a-provider'})
        assert json.loads(error.text.splitlines()[-1])['type'] == 'error'
