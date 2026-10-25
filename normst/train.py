"""Fixed-endpoint fitting from raw observed spot counts and frozen image features."""

import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .conditioning import condition_with_prior, coarsened_counts, group_batch, spatial_groups
from .artifacts import environment, file_identity, save_checkpoint, writer_lock
from .data import context_to, load_image_features, load_observations, load_settings, manifest, read_json, runtime, validate_settings, write_json
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


def random_state(generator, device):
    legacy = np.random.get_state()
    return dict(python=random.getstate(), numpy_global=[legacy[0], legacy[1].tolist(), *legacy[2:]],
        grouping=generator.bit_generator.state, torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_random_state(state, generator, device):
    random.setstate(state["python"])
    legacy = state["numpy_global"]
    np.random.set_state((legacy[0], np.asarray(legacy[1], dtype=np.uint32), *legacy[2:]))
    generator.bit_generator.state = state["grouping"]
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def fit(data_path, output, *, device="cuda:0", config=None, resume=None, checkpoint_every=100):
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer number of optimizer updates")
    restored = torch.load(resume, map_location="cpu", weights_only=True) if resume is not None else None
    if restored is not None and (restored.get("kind") != "normst_training" or restored.get("resume_version") != 1):
        raise ValueError("resume requires a NORMST training checkpoint (resume.pt)")
    settings = restored["config"] if restored is not None and config is None else load_settings(config)
    validate_settings(settings)
    if restored is not None and settings != restored["config"]:
        raise ValueError("resume configuration must match the saved configuration, including the fixed epoch endpoint")
    device = runtime(device, settings["cpu_threads"])
    spec = manifest(data_path)
    data = load_observations(spec)
    features, valid, feature_metadata = load_image_features(spec, return_record=True)
    inputs = dict(observations=file_identity(spec["observations"]), **feature_metadata["files"],
        bounds_yxyx_um=spec["bounds_yxyx_um"], feature_tile_offset_yx=list(spec.get("feature_tile_offset_yx", (0, 0))))
    software = environment(device)
    if restored is not None:
        if inputs != restored["inputs"]:
            raise ValueError("resume inputs differ from the saved content identifiers or geometry")
        if software != restored["environment"]:
            raise ValueError("resume requires the saved software, source code, device and thread configuration")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(output.parent / f".{output.name}.lock"):
        if output.exists():
            status = read_json(output / "status.json") if (output / "status.json").exists() else {}
            if restored is None or status.get("kind") != "training" or status.get("status") == "complete":
                raise FileExistsError(f"{output}: choose a new output or resume an incomplete training run")
            if (output / "run.json").exists():
                previous = read_json(output / "run.json")
                if previous.get("inputs") != inputs or previous.get("config") != settings:
                    raise ValueError("output directory belongs to a different training run; choose a new output")
        else:
            output.mkdir()
        write_json(output / "status.json", dict(kind="training", status="running"))
        try:
            result = _fit(data, spec, features, valid, output, device, settings, restored,
                          checkpoint_every, inputs, software, feature_metadata)
            write_json(output / "status.json", dict(kind="training", status="complete"))
            return result
        except BaseException:
            write_json(output / "status.json", dict(kind="training", status="failed",
                recoverable=(output / "resume.pt").exists()))
            raise


def _fit(data, spec, features, valid, output, device, settings, restored,
         checkpoint_every, inputs, software, feature_metadata):
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
    cursor = dict(epoch=1, next_group=0, order=None, examples=0,
                  totals=dict(allocation=0., coarse_total=0., observed_count=0.))
    parameter_names = [[name for name, _ in field.named_parameters() if not name.startswith("adapter.")],
                       [name for name, _ in field.named_parameters() if name.startswith("adapter.")]]
    if restored is not None:
        if restored["parameter_groups"] != parameter_names:
            raise ValueError("optimizer parameter-group ordering differs from the checkpoint")
        # Loading mutable context buffers must not overwrite the regenerated
        # coarse view that shared their storage during model construction.
        field.set_context({name: getattr(field, name).clone()
                           for name in ("conditioning", "reference_rate", "prior_density")})
        incompatible = field.load_state_dict(restored["model_state"], strict=False)
        if set(incompatible.missing_keys) != {"histology_features", "feature_valid"} or incompatible.unexpected_keys:
            raise ValueError("incompatible training model state")
        optimizer.load_state_dict(restored["optimizer"])
        history, updates, cursor = restored["history"], restored["optimizer_updates"], restored["cursor"]
        restore_random_state(restored["random_state"], generator, device)
    write_json(output / "config.json", settings)
    write_json(output / "run.json", dict(config=settings, environment=software, inputs=inputs,
        feature_provenance=feature_metadata, checkpoint_every_updates=checkpoint_every,
        resumed=restored is not None))

    def checkpoint():
        # Immutable image arrays are reloaded and verified by their content hashes.
        state = {name: value for name, value in field.state_dict().items()
                 if name not in ("histology_features", "feature_valid")}
        save_checkpoint(output / "resume.pt", dict(kind="normst_training", resume_version=1,
            model_state=state, optimizer=optimizer.state_dict(), parameter_groups=parameter_names,
            config=settings, inputs=inputs, environment=software, cursor=cursor,
            optimizer_updates=updates, history=history, random_state=random_state(generator, device)))

    if restored is None:
        checkpoint()
    for epoch in range(cursor["epoch"], settings["epochs"] + 1):
        field.train()
        grouping = (epoch - 1) % len(owners)
        if cursor["order"] is None:
            cursor["order"] = generator.permutation(np.arange(int(owners[grouping].max()) + 1)).tolist()
        order = np.asarray(cursor["order"], dtype=np.int64)
        batch = settings["batch_groups"]
        steps = math.ceil(len(order) / batch)
        totals, examples = cursor["totals"], cursor["examples"]
        lr = optimizer.param_groups[0]["lr"]
        for begin in range(cursor["next_group"], len(order), batch):
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
            cursor.update(next_group=min(begin + batch, len(order)), examples=examples)
            if updates % checkpoint_every == 0:
                checkpoint()
        record = dict(epoch=epoch, updates=updates, learning_rate=lr,
                      **{name: value / examples for name, value in totals.items()})
        history.append(record)
        write_json(output / "progress.json", record)
        if epoch == 1 or epoch % 3 == 0 or epoch == settings["epochs"]:
            print(f"epoch {epoch}/{settings['epochs']}  loss={sum(totals.values()) / examples:.6f}", flush=True)
        cursor = dict(epoch=epoch + 1, next_group=0, order=None, examples=0,
                      totals=dict(allocation=0., coarse_total=0., observed_count=0.))
        checkpoint()
    field.set_context(full)
    field.eval()
    observations = {key: torch.from_numpy(data[key]) for key in ("counts", "centers", "radii")}
    save_checkpoint(output / "model.pt", dict(format_version=1, state_dict=field.state_dict(), observations=observations,
        gene_ids=data["gene_ids"], barcodes=data["barcodes"], config=settings, epochs=settings["epochs"],
        optimizer_updates=updates, feature_tile_offset_yx=list(field.tile_offset_yx),
        provenance=dict(inputs=inputs, environment=software)))
    write_json(output / "training.json", dict(epochs=settings["epochs"], optimizer_updates=updates,
        parameter_count=sum(p.numel() for p in field.parameters()), seconds=time.monotonic() - started,
        selection="fixed_endpoint", history=history))
    return output / "model.pt"
