import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_all():
    print("Testing GET / ...")
    res = client.get("/")
    assert res.status_code == 200, f"Status {res.status_code}"
    assert "<title>UNIVERSAL AI — Precision Studio Core</title>" in res.text, "Title mismatch"
    assert "Phase 1: Core Engine" in res.text, "Phase 1 badge mismatch"
    print("✓ GET / returned 200 with precision studio title and branding")

    print("Testing GET /api/status ...")
    res = client.get("/api/status")
    assert res.status_code == 200
    st = res.json()
    print(f"✓ System: {st['system']}, Mode: {st['mode']}")

    print("Testing GET /api/models ...")
    res = client.get("/api/models")
    assert res.status_code == 200
    models = res.json()["models"]
    print(f"✓ Loaded {len(models)} models: {[m['id'] for m in models]}")

    print("Testing GET /api/tts/voices ...")
    res = client.get("/api/tts/voices")
    assert res.status_code == 200
    voices = res.json()["voices"]
    print(f"✓ Loaded {len(voices)} TTS voices: {[v['id'] for v in voices[:3]]}")

    print("Testing POST /api/tts/file ...")
    res = client.post("/api/tts/file", json={
        "text": "Universal AI Neural Voice Online.",
        "voice": "auto"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success" and data["audio_url"].startswith("/audio/")
    audio_res = client.get(data["audio_url"])
    assert audio_res.status_code == 200 and len(audio_res.content) > 0
    print(f"✓ Synthesized audio file ({data['voice']}): {len(audio_res.content)} bytes")

    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_all()
