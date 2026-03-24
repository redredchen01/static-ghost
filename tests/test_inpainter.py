from unittest.mock import patch, MagicMock
from static_ghost.inpainter import run, check_iopaint


def test_run_calls_iopaint_cli():
    with patch("static_ghost.inpainter.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        run("/tmp/input", "/tmp/mask.png", "/tmp/output", device="cpu")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "iopaint" in cmd[0]
        assert "--model" in cmd
        assert "lama" in cmd
        assert "--image" in cmd
        assert "--device" in cmd
        assert "cpu" in cmd


def test_check_iopaint_available():
    with patch("static_ghost.inpainter.shutil.which", return_value="/usr/local/bin/iopaint"):
        check_iopaint()  # should not raise


def test_check_iopaint_missing():
    import pytest
    with patch("static_ghost.inpainter.shutil.which", return_value=None):
        with pytest.raises(SystemExit):
            check_iopaint()
