import json
import socket
import threading
import time
import urllib.request
from unittest.mock import patch

import pytest
from static_ghost.detector import Region
from static_ghost.picker import pick_region


@pytest.fixture
def sample_frame(tmp_path):
    """Create a minimal PNG file for picker."""
    import cv2
    import numpy as np
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    path = str(tmp_path / "frame.png")
    cv2.imwrite(path, frame)
    return path


def _post_region(port, region_data, delay=0.3):
    """Post a region to the picker server after a short delay."""
    time.sleep(delay)
    data = json.dumps(region_data).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/submit",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req)


def test_pick_region_returns_region_on_submit(sample_frame):
    """Simulate browser submitting a region via POST."""
    port = 19876

    def fake_open(url):
        # Schedule POST in background after server is ready
        threading.Thread(
            target=_post_region,
            args=(port, {"x": 10, "y": 20, "w": 30, "h": 40}),
            daemon=True,
        ).start()

    with patch("static_ghost.picker.webbrowser.open", side_effect=fake_open):
        region = pick_region(sample_frame, port=port)

    assert region == Region(x=10, y=20, w=30, h=40, confidence=1.0)


def test_pick_region_port_fallback(sample_frame):
    """If first port is taken, picker should try the next one."""
    base_port = 19877
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", base_port))
    blocker.listen(1)

    actual_port = base_port + 1  # should fall back here

    def fake_open(url):
        assert str(base_port) not in url
        threading.Thread(
            target=_post_region,
            args=(actual_port, {"x": 5, "y": 5, "w": 10, "h": 10}),
            daemon=True,
        ).start()

    try:
        with patch("static_ghost.picker.webbrowser.open", side_effect=fake_open):
            region = pick_region(sample_frame, port=base_port)
        assert region is not None
    finally:
        blocker.close()
