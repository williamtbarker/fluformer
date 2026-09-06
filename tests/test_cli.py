import json
import runpy
import sys

import pytest

from fluformer import __version__
from fluformer.cli import main


def test_cli_without_command_prints_help(capsys):
    result = main([])

    captured = capsys.readouterr()

    assert result == 0
    assert "Reusable Fluformer model components" in captured.out
    assert "doctor" in captured.out


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    captured = capsys.readouterr()

    assert exc.value.code == 0
    assert f"fluformer {__version__}" in captured.out


def test_cli_doctor_returns_machine_readable_json(capsys):
    result = main(["doctor"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert result == 0
    assert payload["fluformer"] == __version__
    assert "torch" in payload
    assert payload["default_device"] in {
        "cpu",
        "cuda",
        "mps",
    }
    assert isinstance(payload["cuda_available"], bool)
    assert isinstance(payload["mps_available"], bool)


def test_python_module_entrypoint(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["fluformer", "doctor"],
    )

    with pytest.raises(SystemExit) as exc:
        runpy.run_module(
            "fluformer.__main__",
            run_name="__main__",
        )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exc.value.code == 0
    assert payload["fluformer"] == __version__
