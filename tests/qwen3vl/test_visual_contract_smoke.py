from scripts.qwen3vl_visual_contract_smoke import parse_args


def test_visual_contract_smoke_requires_model_and_rgb(monkeypatch):
    monkeypatch.setattr("sys.argv", ["smoke", "--rgb-npy", "/tmp/rgb.npy", "--model", "model"])

    args = parse_args()

    assert args.rgb_npy == "/tmp/rgb.npy"
    assert args.model == "model"
