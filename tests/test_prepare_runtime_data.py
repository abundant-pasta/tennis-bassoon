import importlib.util
from pathlib import Path


def _load_prepare_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_runtime_data.py"
    spec = importlib.util.spec_from_file_location("prepare_runtime_data", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_runtime_data_refreshes_changed_model_bundle(tmp_path, monkeypatch):
    prepare_runtime_data = _load_prepare_module()

    seed_root = tmp_path / "seed"
    target_root = tmp_path / "target"
    (seed_root / "model" / "promoted" / "current").mkdir(parents=True)
    (target_root / "model" / "promoted" / "current").mkdir(parents=True)
    (seed_root / "model" / "promoted" / "current" / "model.pkl").write_text("new")
    (target_root / "model" / "promoted" / "current" / "model.pkl").write_text("old")

    monkeypatch.setenv("TENNIS_SEED_DATA_DIR", str(seed_root))
    monkeypatch.setenv("TENNIS_DATA_DIR", str(target_root))

    prepare_runtime_data.main()

    assert (target_root / "model" / "promoted" / "current" / "model.pkl").read_text() == "new"
