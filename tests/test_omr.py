import asyncio
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient
from music_annotation_backend.app import create_app
from music_annotation_backend.config import Settings
from music_annotation_backend.omr_worker import merge_pages
from test_api import wait_for_job


def page(path, parts=1, notes=True):
    root = ET.Element('score-partwise')
    for index in range(parts):
        part = ET.SubElement(root, 'part', id=f'P{index}')
        measure = ET.SubElement(part, 'measure', number='1')
        if notes: ET.SubElement(measure, 'note')
    ET.ElementTree(root).write(path)
    return path


def test_pdf_pages_merge_by_part_and_renumber(tmp_path):
    output = tmp_path / 'merged.musicxml'
    merge_pages([page(tmp_path / 'one.xml', 2), page(tmp_path / 'two.xml', 2)], output)
    root = ET.parse(output).getroot()
    assert len(root.findall('.//note')) == 4
    assert [[m.get('number') for m in part.findall('measure')] for part in root.findall('part')] == [['1', '2'], ['1', '2']]


def test_missing_staves_or_empty_page_is_not_silently_imported(tmp_path):
    output = tmp_path / 'merged.musicxml'
    with pytest.raises(ValueError, match='Different part counts'):
        merge_pages([page(tmp_path / 'one.xml', 2), page(tmp_path / 'two.xml', 1)], output)
    with pytest.raises(ValueError, match='No score notes'):
        merge_pages([page(tmp_path / 'empty.xml', notes=False)], output)
    assert not output.exists()


def test_omr_job_auth_upload_result_failure_and_cancel(tmp_path, monkeypatch):
    app = create_app(Settings(data_root=tmp_path / 'data', model_root=tmp_path / 'models', vendor_root=tmp_path / 'vendor', session_token='test', auto_load_provider=''))
    async def recognize(source, name, progress):
        assert source.parent.parent == tmp_path / 'data/omr-uploads'
        assert source.name == 'score.png'
        assert name == 'score.png'
        if source.read_bytes() == b'bad': raise RuntimeError('No score notes')
        if source.read_bytes() == b'wait': await asyncio.sleep(30)
        progress('omr', .5, 'Recognized page 1/2')
        return {'musicxml': '<score-partwise/>', 'name': 'score.musicxml', 'pageCount': 2, 'warnings': ['Review'], 'cached': False}
    monkeypatch.setattr(app.state.omr, 'recognize', recognize)
    with TestClient(app, headers={'Authorization': 'Bearer test'}) as api:
        assert api.get('/v1/omr/capabilities', headers={'Authorization': ''}).status_code == 401
        assert api.post('/v1/omr', content=b'ok', headers={'X-File-Name': 'score.exe'}).status_code == 422
        assert api.post('/v1/omr', content=b'', headers={'X-File-Name': 'score.png'}).status_code == 422
        submit = lambda content: api.post('/v1/omr', content=content, headers={'X-File-Name': '..%2F..%2Fscore.png'})
        accepted = submit(b'ok')
        assert accepted.status_code == 202
        job = wait_for_job(api, accepted.json()['jobId'])
        assert job['state'] == 'succeeded' and job['result']['pageCount'] == 2
        job = wait_for_job(api, submit(b'bad').json()['jobId'])
        assert job['state'] == 'failed' and job['error']['detail'] == 'No score notes'
        job_id = submit(b'wait').json()['jobId']
        api.post(f'/v1/jobs/{job_id}:cancel')
        assert wait_for_job(api, job_id)['state'] == 'cancelled'

