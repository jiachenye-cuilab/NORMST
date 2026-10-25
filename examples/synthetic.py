"""Small CPU workflow example with generated counts and image features."""

import argparse
from pathlib import Path

import numpy as np

from normst.data import write_json
from normst.features import validate_external_features
from normst.predict import predict
from normst.train import fit


def make_inputs(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    generator = np.random.default_rng(2027)
    bounds = [0, 0, 320, 320]
    centers = np.stack(np.meshgrid([60., 160., 260.], [60., 160., 260.], indexing="ij"), -1).reshape(-1, 2)
    counts = generator.poisson(10., size=(9, 12)).astype(np.float32)
    features = generator.standard_normal((7, 40, 40)).astype(np.float32)
    np.savez_compressed(folder / "observations.npz", counts=counts, centers_yx_um=centers,
        radius_um=np.full(9, 27.5), gene_ids=np.asarray([f"gene_{i:02d}" for i in range(12)]))
    np.save(folder / "features.npy", features, allow_pickle=False)
    np.save(folder / "feature_valid.npy", np.ones((40, 40), dtype=bool), allow_pickle=False)
    write_json(folder / "dataset.json", dict(observations="observations.npz", features="features.npy",
        feature_valid="feature_valid.npy", bounds_yxyx_um=bounds))
    validate_external_features(folder / "dataset.json", confirm_complete=True)
    return folder / "dataset.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/demo")
    args = parser.parse_args()
    folder = Path(args.output)
    dataset = make_inputs(folder / "data")
    write_json(folder / "smoke_config.json", dict(epochs=4))
    checkpoint = fit(dataset, folder / "model", device="cpu", config=folder / "smoke_config.json")
    prediction = predict(checkpoint, folder / "prediction", device="cpu")
    values = np.load(prediction)
    assert values.shape == (1600, 12) and np.isfinite(values).all() and (values >= 0).all()
    print(f"Example complete: {prediction} ({values.shape[0]} regions, {values.shape[1]} genes)")


if __name__ == "__main__":
    main()
