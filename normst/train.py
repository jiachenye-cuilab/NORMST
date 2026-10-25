"""Fixed-endpoint fitting from raw observed spot counts and frozen image features."""

import math
from pathlib import Path
import time

import numpy as np
import torch

from .conditioning import condition_with_prior, coarsened_counts, group_batch, spatial_groups
from .data import context_to, load_image_features, load_observations, load_settings, manifest, runtime, write_json
from .geometry import BasisIntegrals
from .losses import allocation_deviance, coarse_deviance, count_deviance
from .model import NORMST


def learning_rate(settings, progress):
    peak, floor = settings["learning_rate"], settings["minimum_learning_rate"]
    warmup, horizon = settings["warmup_epochs"], settings["schedule_horizon_epochs"]
    if progress < warmup:
        return peak * progress / warmup
    fraction = min(1., (progress - warmup) / (horizon - warmup))
    return floor + (peak - floor) * (1 + math.cos(math.pi * fraction)) / 2


def make_contexts(data, settings, device):
    def encode(counts):
        return context_to(condition_with_prior(counts, data["centers"], data["areas"],
            np.ones(len(counts), dtype=bool), data["bounds"], settings["conditioning"]), device)

    full = encode(data["counts"])
    owners = [spatial_groups(data["centers"], data["bounds"], settings["block_um"], offset)
              for offset in settings["offsets_um"]]
    contexts = [encode(coarsened_counts(data["counts"], data["areas"], ids)) for ids in owners]
    return full, contexts, owners


def backward_views(field, full, coarse, quadrature, rows, observed, parents, chunk):
    """Accumulate the two input-view gradients with one image encoder graph."""
    image = field.he_encoder(field.histology_features)
    shared = image.detach().requires_grad_(True)
    points, weights, owners = quadrature.batch(rows)
    values = {}
    for name, context in (("coarsened", coarse), ("observed", full)):
        field.set_context(context)
        latent = field.encode(image_features=shared)
        predicted = field.regional_counts(latent, points, weights, owners, len(rows), chunk)
        if name == "coarsened":
            allocation = allocation_deviance(predicted, observed, parents)
            total = coarse_deviance(predicted, observed, parents)
            loss = allocation + total
            values.update(allocation=float(allocation.detach()), coarse_total=float(total.detach()))
        else:
            loss = count_deviance(predicted.double(), observed.double())
            values["observed_count"] = float(loss.detach())
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        del loss, predicted, latent
    if shared.grad is None or not torch.isfinite(shared.grad).all():
        raise ValueError("missing or nonfinite image gradient")
    image.backward(shared.grad)
    return values


def fit(data_path, output, *, device="cuda:0", config=None):
    settings = load_settings(config)
    device = runtime(device, settings["cpu_threads"])
    spec = manifest(data_path)
    data = load_observations(spec)
    features, valid = load_image_features(spec)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    full, contexts, owners = make_contexts(data, settings, device)
    # Construct on CPU so seeded initialization is independent of GPU placement.
    torch.manual_seed(settings["seed"])
    field = NORMST(context_to(contexts[0], "cpu"), data["bounds"], features, valid,
        tile_offset_yx=spec.get("feature_tile_offset_yx", (0, 0))).to(device)
    quadrature = BasisIntegrals(data, device, 8.)
    optimizer = torch.optim.AdamW([
        {"params": [p for name, p in field.named_parameters() if not name.startswith("adapter.")]},
        {"params": list(field.adapter.parameters())},
    ], lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    target = torch.as_tensor(data["counts"], device=device)
    generator = np.random.default_rng(settings["seed"])
    history, updates = [], 0
    write_json(output / "config.json", settings)
    for epoch in range(1, settings["epochs"] + 1):
        field.train()
        grouping = (epoch - 1) % len(owners)
        order = generator.permutation(np.arange(int(owners[grouping].max()) + 1))
        batch = settings["batch_groups"]
        steps = math.ceil(len(order) / batch)
        totals = dict(allocation=0., coarse_total=0., observed_count=0.)
        examples = 0
        for begin in range(0, len(order), batch):
            rows, parents = group_batch(owners[grouping], order[begin:begin + batch])
            lr = learning_rate(settings, epoch - 1 + (begin // batch + 1) / steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            values = backward_views(field, full, contexts[grouping], quadrature, rows,
                                    target[torch.as_tensor(rows, device=device)], parents, settings["query_chunk"])
            torch.nn.utils.clip_grad_norm_(field.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            examples += len(rows)
            for name, value in values.items():
                totals[name] += value * len(rows)
        record = dict(epoch=epoch, updates=updates, learning_rate=lr,
                      **{name: value / examples for name, value in totals.items()})
        history.append(record)
        write_json(output / "progress.json", record)
        if epoch == 1 or epoch % 3 == 0 or epoch == settings["epochs"]:
            print(f"epoch {epoch}/{settings['epochs']}  loss={sum(totals.values()) / examples:.6f}", flush=True)
    field.set_context(full)
    field.eval()
    observations = {key: torch.from_numpy(data[key]) for key in ("counts", "centers", "radii")}
    torch.save(dict(format_version=1, state_dict=field.state_dict(), observations=observations,
        gene_ids=data["gene_ids"], barcodes=data["barcodes"], config=settings, epochs=settings["epochs"],
        optimizer_updates=updates, feature_tile_offset_yx=list(field.tile_offset_yx)), output / "model.pt")
    write_json(output / "training.json", dict(epochs=settings["epochs"], optimizer_updates=updates,
        parameter_count=sum(p.numel() for p in field.parameters()), seconds=time.monotonic() - started,
        selection="fixed_endpoint", history=history))
    return output / "model.pt"
