import torch
import torch.utils.model_zoo

from scripts.benchmark_nyu import NYUInputs, ROOT, build_model, model_config, summary, DepthOnly


def test_random_initialization_never_loads_weights(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Random inference must not load or download weights')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)
    monkeypatch.setattr(torch.utils.model_zoo, 'load_url', forbidden)
    torch.set_num_threads(1)
    args = model_config(2023, ROOT / 'data/nyudepthv2_h5')
    model = build_model(args)
    assert args.variant == 'SPNet-Large' and args.norm_type == 'CNX'
    assert len(model.encoder.stages[4]) == 27
    assert model.encoder.downsample_layers[0].in_channels == 5


def test_nyu_sampling_and_shape():
    dataset = NYUInputs(ROOT / 'data/nyudepthv2_h5')
    first, repeat = dataset[0], dataset[0]
    assert len(dataset) == 654
    assert first['rgb'].shape == (1, 3, 228, 304)
    assert first['dep'].shape == (1, 1, 228, 304)
    assert (first['dep'] > 0).sum() == 500
    assert torch.equal(first['dep'], repeat['dep'])


def test_released_default_configuration():
    args = model_config(2023, ROOT / 'data/nyudepthv2_h5')
    assert args.dims == [192, 384, 768, 1536]
    assert args.depths == [3, 3, 27, 3] and args.dp_rate == 0.2
    assert args.internal_hw == [256, 320] and args.depth_scale_m == 20


def test_adapter_channels_mask_scale_and_crop():
    class Dummy(torch.nn.Module):
        def forward(self, rgb, raw, hole_raw):
            assert rgb.shape == (1, 3, 256, 320)
            assert raw.shape == hole_raw.shape == (1, 1, 256, 320)
            assert torch.equal(hole_raw, (raw > 0).to(raw.dtype))
            assert raw[0, 0, 0, 0] == 0.1
            return raw
    depth = torch.zeros(1, 1, 228, 304)
    depth[0, 0, 0, 0] = 2
    result = DepthOnly(Dummy())(torch.zeros(1, 3, 228, 304), depth)
    torch.testing.assert_close(result, depth)


def test_summary():
    result = summary([10, 20, 30])
    assert result['mean_ms'] == 20 and result['p95_ms'] == 29 and result['fps'] == 50
