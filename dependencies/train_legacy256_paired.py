"""
OURS-DEF: Dual-path Evidential Fusion (+ improvement flags, 2026-07-04).
Base: per-view DS path (robust) + joint concat path (complementarity) + DS/gate combine.
New improvement flags (default OFF -> identical to base oursdefg):
  --kl_reg F   : standard EDL annealed KL-to-uniform regularizer on misleading evidence (calibration). [3]
  --vdiscount  : dbf-style conflict->uncertainty discounting in the per-view DS combine
                 (no Dempster belief inflation under conflict; helps calibration + conflict + mammography per-view path). [2]
  --cgate      : learnable global joint-path gain (data-driven complementarity detection;
                 shrinks joint path toward 0 where it does not help, e.g. mammography). [1]
  --jointnorm  : missing-aware rescaling of the joint MLP input (preserve magnitude when views drop). [4]
Usage: python3 train_ours_def_fair.py --method oursdef --level view --fold 0 --seed 0 --device cuda
"""
import argparse, hashlib, importlib.util, io, json, sys, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


PROTOCOL = "DEF_LEGACY256_PAIRED_V1_20260820"
FORMAL_METHOD = "dsonly256_paired"
FORMAL_HIDDEN = 256
FORMAL_EPOCHS = 100
FORMAL_SWEEP = (
    ("clean", 0.0), ("missing", 0.25), ("missing", 0.50), ("missing", 0.75),
    ("view_noise", 0.05), ("view_noise", 0.10), ("feature_dropout", 0.25),
    ("feature_dropout", 0.50), ("view_conflict", 0.25), ("view_conflict", 0.50),
    ("single_view", 1.0),
)
HISTORICAL_SOURCE_SHA256 = "cec614cb3b97a337fe238a08ce78bf6508206a8fc8de0818870735210e7aaab5"
LOADER_SHA256 = "a3acbec210c527b02663db6d2fa406adf67fecc1f4d5d5f9004890f0183ab717"
CORRUPTION_SHA256 = "769d11a5fa5d895bcefe8df8b52057d76ebb65d3921e08aeefd1b4ae626eb377"
INPUT_MANIFEST_SHA256 = "e63dc0e431683933d7d0cc162aee681d6039c04bcbca59987199c9da472bbd6a"


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def _verify_sha(path, expected, name):
    if not isinstance(expected, str) or len(expected) != 64: raise ValueError(f"invalid expected {name} sha256")
    actual = _sha256(path)
    if actual != expected.lower(): raise ValueError(f"{name} SHA mismatch: {actual} != {expected.lower()}")
    return actual


def _import_exact(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None: raise ValueError(f"cannot import {module_name}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def preflight_authorities(*, root, feature_dir, fold, runner_path, expected_runner_sha256,
                          legacy_path, expected_legacy_sha256, loader_path, expected_loader_sha256,
                          corruption_path, expected_corruption_sha256, manifest_path, expected_manifest_sha256):
    root = Path(root)
    authority = {"runner_sha256": _verify_sha(runner_path, expected_runner_sha256, "runner"),
        "historical_source_sha256": _verify_sha(legacy_path, expected_legacy_sha256, "historical source"),
        "loader_sha256": _verify_sha(loader_path, expected_loader_sha256, "loader"),
        "corruption_sha256": _verify_sha(corruption_path, expected_corruption_sha256, "corruption"),
        "input_manifest_sha256": _verify_sha(manifest_path, expected_manifest_sha256, "input manifest")}
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if type(manifest) is not dict or type(manifest.get("files")) is not dict: raise ValueError("invalid input manifest schema")
    input_sha = {}
    for split in ("train", "val"):
        rel = f"{feature_dir}/fold{fold}_{split}_view.npz"; row = manifest["files"].get(rel); path = root / rel
        if type(row) is not dict or type(row.get("bytes")) is not int or type(row.get("sha256")) is not str:
            raise ValueError(f"missing manifest authority for {rel}")
        if not path.is_file() or path.stat().st_size != row["bytes"]: raise ValueError(f"input size mismatch for {rel}")
        input_sha[split] = _verify_sha(path, row["sha256"], f"{split} input")
    loader_module = _import_exact(loader_path, "legacy256_exact_loader")
    corruption_module = _import_exact(corruption_path, "legacy256_exact_corruption")
    load_npz_fair = getattr(loader_module, "load_npz_fair", None); apply_corruption = getattr(corruption_module, "apply_corruption", None)
    if not callable(load_npz_fair) or not callable(apply_corruption): raise ValueError("loader/corruption API is not callable")
    authority.update({"input_sha256": input_sha, "load_npz_fair": load_npz_fair, "apply_corruption": apply_corruption,
                      "_file_authority": {
                          str(Path(runner_path)): authority["runner_sha256"],
                          str(Path(legacy_path)): authority["historical_source_sha256"],
                          str(Path(loader_path)): authority["loader_sha256"],
                          str(Path(corruption_path)): authority["corruption_sha256"],
                          str(Path(manifest_path)): authority["input_manifest_sha256"],
                          str(root / f"{feature_dir}/fold{fold}_train_view.npz"): input_sha["train"],
                          str(root / f"{feature_dir}/fold{fold}_val_view.npz"): input_sha["val"],
                      }})
    verify_authorities_unchanged(authority)
    return authority


def verify_authorities_unchanged(authority):
    files = authority.get("_file_authority")
    if type(files) is not dict or not files:
        raise ValueError("missing authority file snapshot")
    for path, expected in files.items():
        _verify_sha(path, expected, "authority snapshot")


def _fsync_dir(path):
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(Path(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_all(fd, payload):
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0: raise OSError("short write")
        view = view[written:]


def _exclusive_bytes(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try: _write_all(fd, payload); os.fsync(fd)
    finally: os.close(fd)
    _fsync_dir(path.parent)
    return path


def acquire_cell_lease(root, cell_id, payload):
    if not isinstance(cell_id, str) or not cell_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in cell_id):
        raise ValueError("invalid cell id")
    lease = Path(root) / ".legacy256_paired_leases" / f"{cell_id}.lease.json"
    body = dict(payload); body.update({"protocol": PROTOCOL, "status": "LEASED"})
    return _exclusive_bytes(lease, json.dumps(body, sort_keys=True).encode("utf-8"))


def formal_run_config(*, seed, fold, corruption_seed, authority):
    serializable = {k: v for k, v in authority.items() if k not in {"load_npz_fair", "apply_corruption"} and not k.startswith("_")}
    return {"protocol": PROTOCOL, "method": FORMAL_METHOD, "level": "view", "training_seed": seed, "fold": fold,
        "corruption_seed": corruption_seed, "epochs": 100, "batch_size": 16, "lr": 1e-3, "hidden": 256,
        "w_aux": 0.3, "combine": "gate", "path": "view", "vagg": "ds", "fdiscount": True,
        "prediction_path": "classical_per_view_ds", "training_auxiliary": "joint_head_plus_per_view_deep_supervision",
        "vacuous_missing_views": True, "vdiscount": False, "unreachable_final_gate_fdiscount": True, "authority": serializable}


def _npz_bytes(mapping):
    buffer = io.BytesIO(); np.savez(buffer, **mapping); return buffer.getvalue()


def _is_hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_artifacts(artifacts):
    if set(artifacts) != {"clean_scores", "metrics", "rows", "robust_scores", "config"}:
        raise ValueError("invalid artifact object keys")
    rows = artifacts["rows"]
    if type(rows) is not list or [(r.get("corruption"), r.get("corruption_level")) for r in rows] != list(FORMAL_SWEEP):
        raise ValueError("robust conditions do not match exact formal conditions")
    row_keys = {"corruption", "corruption_level", "auc70_last", "clean_last", "drop_from_clean", "corruption_seed"}
    for row in rows:
        if type(row) is not dict or set(row) != row_keys or type(row["corruption_seed"]) is not int:
            raise ValueError("invalid robust row schema")
        if any(type(row[key]) not in (int, float) or not np.isfinite(row[key])
               for key in ("corruption_level", "auc70_last", "clean_last", "drop_from_clean")):
            raise ValueError("invalid robust row value")
    metrics = artifacts["metrics"]
    common = {"method", "level", "fold", "classes", "n_val"}
    expected_metric_keys = common | ({"auc_last", "auprc_last"} if metrics.get("classes") == 2 else {"acc_last"})
    if type(metrics) is not dict or set(metrics) != expected_metric_keys:
        raise ValueError("invalid metrics schema")
    if metrics.get("method") != FORMAL_METHOD or metrics.get("level") != "view":
        raise ValueError("invalid metrics identity")
    if any(type(metrics.get(k)) is not int for k in ("fold", "classes", "n_val")):
        raise ValueError("invalid metrics integer")
    for key in expected_metric_keys - common:
        if type(metrics[key]) not in (int, float) or not np.isfinite(metrics[key]):
            raise ValueError("invalid metric value")
    config = artifacts["config"]
    config_keys = {"protocol", "method", "level", "training_seed", "fold", "corruption_seed", "epochs",
                   "batch_size", "lr", "hidden", "w_aux", "combine", "path", "vagg", "fdiscount",
                   "prediction_path", "training_auxiliary", "vacuous_missing_views", "vdiscount",
                   "unreachable_final_gate_fdiscount", "authority"}
    if type(config) is not dict or set(config) != config_keys or config.get("protocol") != PROTOCOL:
        raise ValueError("invalid run config schema")
    auth = config["authority"]
    auth_keys = {"runner_sha256", "historical_source_sha256", "loader_sha256", "corruption_sha256",
                 "input_manifest_sha256", "input_sha256"}
    if type(auth) is not dict or set(auth) != auth_keys or type(auth["input_sha256"]) is not dict or set(auth["input_sha256"]) != {"train", "val"}:
        raise ValueError("invalid config authority schema")
    if not all(_is_hex64(auth[k]) for k in auth_keys - {"input_sha256"}) or not all(_is_hex64(v) for v in auth["input_sha256"].values()):
        raise ValueError("invalid config authority SHA")
    clean_scores, robust_scores = artifacts["clean_scores"], artifacts["robust_scores"]
    if set(clean_scores) != {"y", "score"}: raise ValueError("invalid clean score schema")
    expected = {"y"} | {f"{c}_{v:.2f}" for c, v in FORMAL_SWEEP}
    if set(robust_scores) != expected: raise ValueError("invalid robust score schema")
    y = np.asarray(clean_scores["y"]); clean_score = np.asarray(clean_scores["score"])
    if y.ndim != 1 or not np.array_equal(y, np.asarray(robust_scores["y"])): raise ValueError("score labels disagree")
    if metrics["n_val"] != y.shape[0] or metrics["fold"] != config["fold"]: raise ValueError("metrics/config mismatch")
    if clean_score.shape[0] != y.shape[0] or not np.isfinite(clean_score).all(): raise ValueError("invalid clean score")
    if np.any(clean_score < 0) or np.any(clean_score > 1): raise ValueError("clean score outside probability range")
    for key, value in robust_scores.items():
        arr = np.asarray(value)
        if key == "y": continue
        if arr.shape != clean_score.shape or not np.isfinite(arr).all() or np.any(arr < 0) or np.any(arr > 1):
            raise ValueError(f"invalid robust score array {key}")

def publish_complete_cell(lease, clean_dir, robust_dir, artifacts):
    lease = Path(lease)
    if not lease.is_file(): raise ValueError("missing cell lease")
    _validate_artifacts(artifacts); clean_dir, robust_dir = Path(clean_dir), Path(robust_dir)
    stage = lease.parents[1] / ".legacy256_paired_staging" / lease.stem; stage.mkdir(parents=True, exist_ok=False)
    cfg = json.dumps(artifacts["config"], indent=2, sort_keys=True).encode("utf-8")
    staged = {"clean/val_scores.npz": _npz_bytes(artifacts["clean_scores"]),
        "clean/metrics.json": json.dumps(artifacts["metrics"], indent=2, sort_keys=True).encode("utf-8"),
        "clean/run_config.json": cfg, "robust/robustness.json": json.dumps(artifacts["rows"], indent=2, sort_keys=True).encode("utf-8"),
        "robust/robust_scores.npz": _npz_bytes(artifacts["robust_scores"]), "robust/run_config.json": cfg}
    stage_paths = {rel: _exclusive_bytes(stage / rel, payload) for rel, payload in staged.items()}
    targets = {"clean/val_scores.npz": clean_dir / "val_scores.npz", "clean/metrics.json": clean_dir / "metrics.json",
        "clean/run_config.json": clean_dir / "run_config.json", "robust/robustness.json": robust_dir / "robustness.json",
        "robust/robust_scores.npz": robust_dir / "robust_scores.npz", "robust/run_config.json": robust_dir / "run_config.json"}
    if any(path.exists() for path in list(targets.values()) + [robust_dir / "COMPLETE.json"]): raise FileExistsError("refusing existing formal output")
    for rel, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            _exclusive_bytes(target, stage_paths[rel].read_bytes())  # independent no-clobber copy, never a hard-link alias
        else:
            os.link(stage_paths[rel], target)
    _fsync_dir(clean_dir); _fsync_dir(robust_dir)
    if os.name != "nt":
        for staged_path in stage_paths.values():
            staged_path.unlink()
        for directory in (stage / "clean", stage / "robust", stage):
            directory.rmdir()
        _fsync_dir(stage.parent)
    receipt = {"protocol": PROTOCOL, "status": "COMPLETE", "lease_sha256": _sha256(lease),
        "artifacts": {rel: {"sha256": _sha256(path), "bytes": path.stat().st_size} for rel, path in targets.items()},
        "configuration": artifacts["config"]}
    return _exclusive_bytes(robust_dir / "COMPLETE.json", json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8"))


def corruption_seed_for_fold(fold):
    if type(fold) is not int or not 0 <= fold <= 9:
        raise ValueError("fold must be an integer in [0, 9]")
    return 1000 + fold


def validate_formal_protocol(values):
    wanted = {
        "method": FORMAL_METHOD, "level": "view", "epochs": FORMAL_EPOCHS,
        "batch_size": 16, "lr": 1e-3, "hidden": FORMAL_HIDDEN,
        "combine": "gate", "path": "view", "vdiscount": False, "fdiscount": True,
        "w_aux": 0.3, "vrel": False, "kl_reg": 0.0, "cgate": False,
        "jointnorm": False, "pvdeep": False, "pvrefine": False, "xmodal": False,
        "disc": False, "xref": False, "vagg": "ds", "xrefjoint": False,
    }
    seed = values["seed"] if isinstance(values, dict) else values.seed
    fold = values["fold"] if isinstance(values, dict) else values.fold
    if type(seed) is not int or seed not in (0, 1, 2): raise ValueError("legacy256 protocol drift: seed")
    if type(fold) is not int or fold not in range(5): raise ValueError("legacy256 protocol drift: fold")
    for name, expected in wanted.items():
        actual = values[name] if isinstance(values, dict) else getattr(values, name)
        same = np.isclose(actual, expected) if isinstance(expected, float) else actual == expected
        if not same:
            raise ValueError(f"legacy256 protocol drift: {name}={actual!r}, expected {expected!r}")


def assert_fresh_output_targets(clean_dir, robust_dir):
    targets = (
        Path(clean_dir) / "val_scores.npz", Path(clean_dir) / "metrics.json",
        Path(clean_dir) / "run_config.json", Path(robust_dir) / "robustness.json",
        Path(robust_dir) / "robust_scores.npz", Path(robust_dir) / "run_config.json",
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError("refusing existing formal output: " + ", ".join(existing))


class DEF(nn.Module):
    def __init__(s, V, dim, classes=2, hidden=128, combine="ds", path="both", vrel=False,
                 vdiscount=False, cgate=False, jointnorm=False, fdiscount=False,
                 pvdeep=False, pvrefine=False, xmodal=False, vagg="ds", xrefjoint=False):
        super().__init__(); s.V = V; s.C = classes; s.combine = combine; s.path = path; s.vrel = vrel
        s.vdiscount = vdiscount; s.cgate = cgate; s.jointnorm = jointnorm; s.fdiscount = fdiscount
        s.pvrefine = pvrefine; s.xmodal = xmodal; s.vagg = vagg; s.xrefjoint = xrefjoint
        s.enc = nn.ModuleList([nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1)) for _ in range(V)])
        if xmodal:   # [innovation] cross-modal attention joint path (replaces masked-concat->MLP) to exploit genuine cross-modal complementarity
            s.xattn = nn.MultiheadAttention(hidden, num_heads=4, dropout=0.1, batch_first=True)
            s.xnorm = nn.LayerNorm(hidden)
            s.xproj = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, classes))
        s.disc = False   # set by main; discriminative-primary head (reliability-weighted cross-modal, CE-trained)
        s.dattn = nn.MultiheadAttention(hidden, num_heads=4, dropout=0.1, batch_first=True)
        s.dnorm = nn.LayerNorm(hidden)
        s.dhead = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, classes))
        # [xref] cross-view RESIDUAL refinement, then CONCAT all V refined tokens (never < concat, since residual keeps originals)
        s.xref = False
        s.xr_attn = nn.MultiheadAttention(hidden, num_heads=4, dropout=0.1, batch_first=True)
        s.xr_n1 = nn.LayerNorm(hidden); s.xr_n2 = nn.LayerNorm(hidden)
        s.xr_ff = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, hidden))
        s.xr_head = nn.Sequential(nn.Linear(V * hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, classes))
        # [xref zero-init] refinement starts as IDENTITY -> Hs'=Hs -> classifier sees EXACTLY concat at init;
        # training only ADDS cross-view refinement where it helps -> provably >= concat generalization floor.
        nn.init.zeros_(s.xr_attn.out_proj.weight); nn.init.zeros_(s.xr_attn.out_proj.bias)
        nn.init.zeros_(s.xr_ff[-1].weight); nn.init.zeros_(s.xr_ff[-1].bias)
        # [router] TMUR/TUNED-style GLOBAL cross-view router: per-view fusion weights come from cross-view
        # global context (attention), NOT branch-local vacuity u=K/S (which is scale-incomparable, TMUR Thm 4.1).
        s.rt_attn = nn.MultiheadAttention(hidden, num_heads=4, dropout=0.1, batch_first=True)
        s.rt_norm = nn.LayerNorm(hidden)
        s.rt_gate = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        s.rt_tau = 1.0
        if pvdeep:   # [root-1] deeper per-view evidence heads (more capacity, keeps per-view independence)
            s.head_v = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, classes)) for _ in range(V)])
        else:
            s.head_v = nn.ModuleList([nn.Linear(hidden, classes) for _ in range(V)])
        if pvrefine:  # [root-1] masked cross-view context refines each view's embedding BEFORE evidence (missing views excluded -> robustness-safe)
            s.refiner = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        s.joint = nn.Sequential(nn.Linear(V * hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, classes))
        if combine == "gate":
            s.gate = nn.Linear(4, 2)
        if vrel:
            s.rel = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)) for _ in range(V)])
        if cgate:               # global joint-path gain in (0,2), init ~1; learns to shrink where joint hurts
            s.jgain = nn.Parameter(torch.zeros(1))

    def ds_two(s, a1, a2, discount=False):
        C = s.C; S1 = a1.sum(1, keepdim=True); S2 = a2.sum(1, keepdim=True)
        b1 = (a1 - 1) / S1; b2 = (a2 - 1) / S2; u1 = C / S1; u2 = C / S2
        bb = torch.bmm(b1.unsqueeze(2), b2.unsqueeze(1))
        Cf = (bb.sum((1, 2)) - torch.diagonal(bb, dim1=1, dim2=2).sum(1)).unsqueeze(1)  # (N,1) conflict
        if discount:   # dbf-style: send conflict mass to uncertainty (no 1/(1-Cf) belief inflation)
            b_a = b1 * b2 + b1 * u2 + b2 * u1
            u_a = u1 * u2 + Cf
            tot = b_a.sum(1, keepdim=True) + u_a
            b_a = b_a / tot; u_a = u_a / tot
            return b_a * (C / u_a.clamp_min(1e-6)) + 1.0
        den = (1 - Cf).clamp_min(1e-6)
        b_a = (b1 * b2 + b1 * u2 + b2 * u1) / den; u_a = (u1 * u2) / den
        return b_a * (C / u_a.clamp_min(1e-6)) + 1.0

    def forward(s, xs, mask):
        H = [enc(x) for enc, x in zip(s.enc, xs)]
        m = mask if mask.shape[0] == s.V else mask.t()                  # (V,N)
        if s.pvrefine:   # masked cross-view context (present views only) refines each per-view embedding
            Hs = torch.stack(H, 0); mm = m.unsqueeze(-1)               # (V,N,hidden),(V,N,1)
            ctx = (Hs * mm).sum(0) / mm.sum(0).clamp_min(1.0)          # (N,hidden) present-view mean
            r = s.refiner(ctx)
            H = [h + r for h in H]
        alphas = []
        for v in range(s.V):
            ev = F.softplus(s.head_v[v](H[v]))
            if s.vrel: ev = torch.sigmoid(s.rel[v](H[v])) * ev
            av = ev + 1.0
            mv = m[v].unsqueeze(-1)
            alphas.append(torch.where(mv > 0, av, torch.ones_like(av)))
        if s.vagg == "ds":
            a_views = alphas[0]
            for v in range(1, s.V): a_views = s.ds_two(a_views, alphas[v], discount=s.vdiscount)
        elif s.vagg == "router":
            # [router] GLOBAL cross-view router: per-view weights from attention-enhanced global context
            # (scale-invariant; fixes DS order-dependence + branch-local-u bias). e_fused = Σ π_v e_v.
            Hs = torch.stack([H[v] for v in range(s.V)], 1)                 # (N,V,hidden)
            kpm = (m.t() <= 0); kpm = kpm & (~kpm.all(1, keepdim=True))
            ctx, _ = s.rt_attn(Hs, Hs, Hs, key_padding_mask=kpm)           # cross-view attention
            ctx = s.rt_norm(ctx + Hs)                                       # (N,V,hidden) context-enhanced
            lw = s.rt_gate(ctx).squeeze(-1).masked_fill(m.t() <= 0, -1e9)   # (N,V) routing logits
            pi = torch.softmax(lw / s.rt_tau, dim=1)                        # (N,V) global-context weights
            Ev = torch.stack([alphas[v] - 1.0 for v in range(s.V)], 1)      # (N,V,C) evidence
            a_views = (pi.unsqueeze(-1) * Ev).sum(1) + 1.0                  # routed Dirichlet
        else:
            # RCML/FUML-style aggregation of per-view EVIDENCE (avoids DS over-concentration on many views)
            E = torch.stack([alphas[v] - 1.0 for v in range(s.V)], 0)        # (V,N,C) evidence (0 for missing)
            Mv = m.unsqueeze(-1)                                             # (V,N,1) present mask
            if s.vagg == "wavg":                                            # FUML-style reliability weighting
                S = E.sum(-1, keepdim=True) + s.C                           # (V,N,1) Dirichlet strength
                rel = (1.0 - s.C / S.clamp_min(1e-6)).clamp_min(1e-3)       # reliability = 1 - vacuity
                w = torch.softmax(rel.masked_fill(Mv <= 0, -1e9), dim=0)    # softmax over present views
            else:                                                           # "avg": RCML-style uniform mean
                w = Mv / Mv.sum(0, keepdim=True).clamp_min(1e-6)
            a_views = (w * E).sum(0) + 1.0                                  # fused Dirichlet
        # joint path: cross-modal attention (xmodal) OR masked concat -> evidence
        if s.xmodal:
            Hstk = torch.stack([H[v] for v in range(s.V)], dim=1)          # (N,V,hidden)
            kpm = (m.t() <= 0)                                             # (N,V) True = missing modality
            kpm = kpm & (~kpm.all(1, keepdim=True))                        # guard all-missing rows (avoid NaN)
            att, _ = s.xattn(Hstk, Hstk, Hstk, key_padding_mask=kpm)       # each modality attends to others
            att = s.xnorm(att + Hstk)
            mm = m.t().unsqueeze(-1)                                       # (N,V,1) present mask
            hj = (att * mm).sum(1) / mm.sum(1).clamp_min(1.0)             # pool over present modalities
            e_joint = F.softplus(s.xproj(hj)); a_joint = e_joint + 1.0
        elif s.xrefjoint:
            # [unified] complementarity path uses CROSS-VIEW REFINED tokens (xref) instead of raw concat,
            # then the evidential joint head -> combined with the robust view path by the gate (best-of-both:
            # XR accuracy on agreeing views + DS robustness/calibration on conflict/low-quality).
            Hs = torch.stack([H[v] * m[v].unsqueeze(-1) for v in range(s.V)], 1)  # (N,V,hidden)
            kpm = (m.t() <= 0); kpm = kpm & (~kpm.all(1, keepdim=True))
            a, _ = s.xr_attn(Hs, Hs, Hs, key_padding_mask=kpm)
            Hs = s.xr_n1(Hs + a); Hs = s.xr_n2(Hs + s.xr_ff(Hs))                  # residual refine (zero-init -> starts = concat)
            Hj = (Hs * m.t().unsqueeze(-1)).reshape(Hs.shape[0], -1)              # (N, V*hidden)
            e_joint = F.softplus(s.joint(Hj)); a_joint = e_joint + 1.0
        else:
            blocks = [H[v] * m[v].unsqueeze(-1) for v in range(s.V)]
            Hj = torch.cat(blocks, dim=-1)
            if s.jointnorm:
                scale = (s.V / m.sum(0, keepdim=True).clamp_min(1.0)).t()
                Hj = Hj * scale
            e_joint = F.softplus(s.joint(Hj)); a_joint = e_joint + 1.0
        if s.cgate:                                                     # learnable global complementarity gain
            jg = 2.0 * torch.sigmoid(s.jgain)
            a_joint = jg * (a_joint - 1.0) + 1.0
        if s.path == "joint": return a_joint, a_joint, [a - 1.0 for a in alphas]
        if s.path == "view": return a_views, a_joint, [a - 1.0 for a in alphas]
        if s.combine == "gate":
            def stats(a):
                S = a.sum(1, keepdim=True); return torch.cat([s.C / S, ((a - 1) / S).max(1, keepdim=True).values], -1)
            r = torch.sigmoid(s.gate(torch.cat([stats(a_views), stats(a_joint)], -1)))
            a_views = r[:, 0:1] * (a_views - 1) + 1.0; a_joint = r[:, 1:2] * (a_joint - 1) + 1.0
        a_final = s.ds_two(a_views, a_joint, discount=s.fdiscount)   # [fdiscount] no DS confidence inflation in final combine
        return a_final, a_joint, [a - 1.0 for a in alphas]

    def forward_disc(s, xs, mask):
        """Discriminative-primary: reliability-weighted cross-modal attention -> CE logits (uncapped
        accuracy = concat floor + cross-modal gain), robustified by per-view evidential reliability.
        Returns (logits, per_view_evidences). Predict = softmax(logits); evidences give reliability+calibration."""
        H = [enc(x) for enc, x in zip(s.enc, xs)]
        m = mask if mask.shape[0] == s.V else mask.t()
        if s.pvrefine:
            Hs0 = torch.stack(H, 0); mm0 = m.unsqueeze(-1)
            ctx = (Hs0 * mm0).sum(0) / mm0.sum(0).clamp_min(1.0)
            H = [h + s.refiner(ctx) for h in H]
        evs = []; rel = []
        for v in range(s.V):
            ev = F.softplus(s.head_v[v](H[v])); evs.append(ev)
            a = ev + 1.0; conf = 1.0 - s.C / a.sum(1, keepdim=True)      # (N,1) reliability = 1-vacuity
            rel.append(conf.clamp_min(1e-3) * m[v].unsqueeze(-1))        # missing -> 0
        Hstk = torch.stack([H[v] for v in range(s.V)], 1)               # (N,V,hidden)
        kpm = (m.t() <= 0); kpm = kpm & (~kpm.all(1, keepdim=True))
        att, _ = s.dattn(Hstk, Hstk, Hstk, key_padding_mask=kpm)
        att = s.dnorm(att + Hstk)
        w = torch.stack(rel, 1); w = w / w.sum(1, keepdim=True).clamp_min(1e-6)   # (N,V,1) reliability weights
        fused = (att * w).sum(1)                                          # reliability-weighted cross-modal fusion
        return s.dhead(fused), evs

    def forward_xref(s, xs, mask):
        """Cross-view RESIDUAL refinement then CONCAT all V refined tokens (the user's idea):
        A'=A+f(A,B), B'=B+f(B,A); classifier sees [A',B'] which CONTAINS [A,B] (concat floor) plus
        cross-view enhancement -> provably >= plain concat in information. CE-trained. Returns (logits, per_view_ev)."""
        H = [enc(x) for enc, x in zip(s.enc, xs)]
        m = mask if mask.shape[0] == s.V else mask.t()
        Hs = torch.stack([H[v] * m[v].unsqueeze(-1) for v in range(s.V)], 1)      # (N,V,hidden) masked tokens
        kpm = (m.t() <= 0); kpm = kpm & (~kpm.all(1, keepdim=True))
        a, _ = s.xr_attn(Hs, Hs, Hs, key_padding_mask=kpm)
        Hs = s.xr_n1(Hs + a)                                                      # residual cross-view attention
        Hs = s.xr_n2(Hs + s.xr_ff(Hs))                                            # residual FFN
        flat = (Hs * m.t().unsqueeze(-1)).reshape(Hs.shape[0], -1)                # concat all V refined tokens (N, V*hidden)
        evs = [F.softplus(s.head_v[v](H[v])) for v in range(s.V)]                 # per-view evidence (aux calibration)
        return s.xr_head(flat), evs


def kl_dirichlet(alpha, C):
    beta = torch.ones_like(alpha)
    S_a = alpha.sum(1, keepdim=True); S_b = beta.sum(1, keepdim=True)
    lnB = torch.lgamma(S_a) - torch.lgamma(alpha).sum(1, keepdim=True)
    lnB_u = torch.lgamma(beta).sum(1, keepdim=True) - torch.lgamma(S_b)
    dg = torch.digamma(alpha) - torch.digamma(S_a)
    return (((alpha - beta) * dg).sum(1, keepdim=True) + lnB + lnB_u).mean()


def edl_loss(evidence, y, epoch=1, kl_reg=0.0, classes=2):
    alpha = evidence + 1.0; S = alpha.sum(1, keepdim=True)
    ce = (torch.gather(torch.digamma(S) - torch.digamma(alpha), 1, y.unsqueeze(1))).mean()
    if kl_reg <= 0.0: return ce
    yoh = F.one_hot(y, classes).float()
    alpha_t = yoh + (1.0 - yoh) * alpha           # remove evidence for the true class (misleading evidence)
    lam = min(1.0, epoch / 10.0) * kl_reg          # annealed
    return ce + lam * kl_dirichlet(alpha_t, classes)


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) >= 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/private_data/MutiView/dacvg_project/baseline/MV_LQF")
    ap.add_argument("--method", default="oursdef"); ap.add_argument("--level", default="view", choices=["patient", "view"])
    ap.add_argument("--fold", type=int, default=0); ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cpu")
    ap.add_argument("--w_aux", type=float, default=0.3)
    ap.add_argument("--combine", default="ds", choices=["ds", "gate"])
    ap.add_argument("--path", default="both", choices=["both", "joint", "view"])
    ap.add_argument("--vrel", action="store_true")
    ap.add_argument("--kl_reg", type=float, default=0.0)        # [3]
    ap.add_argument("--vdiscount", action="store_true")        # [2]
    ap.add_argument("--cgate", action="store_true")            # [1]
    ap.add_argument("--jointnorm", action="store_true")        # [4]
    ap.add_argument("--fdiscount", action="store_true")        # [root-2] conflict-discount the FINAL DS combine too
    ap.add_argument("--pvdeep", action="store_true")           # [root-1] deeper per-view evidence heads
    ap.add_argument("--pvrefine", action="store_true")         # [root-1] masked cross-view refinement of per-view evidence
    ap.add_argument("--xmodal", action="store_true")           # [innovation] cross-modal attention joint path
    ap.add_argument("--disc", action="store_true")             # [root-change] discriminative-primary reliability-weighted fusion (CE, uncapped accuracy)
    ap.add_argument("--xref", action="store_true")             # [user idea] cross-view residual refine + concat all tokens (>= concat floor)
    ap.add_argument("--hidden", type=int, default=128)         # hidden width (sweepable)
    ap.add_argument("--vagg", default="ds", choices=["ds", "avg", "wavg", "router"])  # view aggregation: DS / RCML-avg / FUML-reliability / TMUR-global-router
    ap.add_argument("--xrefjoint", action="store_true")        # [unified] joint/complementarity path uses cross-view-refined tokens (best-of DS+XR)
    ap.add_argument("--expected-runner-sha256", required=True)
    ap.add_argument("--legacy-source", default=str(Path(__file__).with_name("train_ours_def_v2.py")))
    ap.add_argument("--loader-source", default=str(Path(__file__).with_name("train_mv_feature_baselines_fair.py")))
    ap.add_argument("--corruption-source", default=str(Path(__file__).with_name("train_mv_feature_robustness_fair.py")))
    ap.add_argument("--input-manifest", default=str(Path(__file__).with_name("input_manifest_20260805.json")))
    args = ap.parse_args()
    validate_formal_protocol(vars(args))
    root = Path(args.root)
    feature_dir = os.environ["MV_FEATURES_DIR"]
    authority = preflight_authorities(
        root=root, feature_dir=feature_dir, fold=args.fold,
        runner_path=Path(__file__), expected_runner_sha256=args.expected_runner_sha256,
        legacy_path=args.legacy_source, expected_legacy_sha256=HISTORICAL_SOURCE_SHA256,
        loader_path=args.loader_source, expected_loader_sha256=LOADER_SHA256,
        corruption_path=args.corruption_source, expected_corruption_sha256=CORRUPTION_SHA256,
        manifest_path=args.input_manifest, expected_manifest_sha256=INPUT_MANIFEST_SHA256,
    )
    load_npz_fair = authority["load_npz_fair"]
    apply_corruption = authority["apply_corruption"]
    suffix = ("_s%d" % args.seed) if args.seed else ""
    cd = root / os.environ["MV_RUNS_DIR"] / ("%s_%s_fold%d%s" % (
        args.method, args.level, args.fold, suffix))
    rd = root / os.environ["MV_ROBUST_DIR"] / ("%s_%s_fold%d%s" % (
        args.method, args.level, args.fold, suffix))
    assert_fresh_output_targets(cd, rd)
    cell_id = f"{feature_dir}.fold{args.fold}.seed{args.seed}.{FORMAL_METHOD}"
    lease = acquire_cell_lease(root, cell_id, {"cell": cell_id, "authority": {
        k: v for k, v in authority.items() if k not in {"load_npz_fair", "apply_corruption"}}})
    corruption_seed = corruption_seed_for_fold(args.fold)
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(2)
    if args.device == "cpu":
        torch.Tensor.cuda = lambda self, *a, **k: self; torch.cuda.is_available = lambda: False
    Xtr, ytr, mtr, Xva, yva, mva = load_npz_fair(root, args.fold, args.level)
    verify_authorities_unchanged(authority)
    device = torch.device(args.device); V, Ntr, D = Xtr.shape
    Xtr_v = [torch.from_numpy(Xtr[v]).float().to(device) for v in range(V)]
    ytr_t = torch.from_numpy(ytr).long().to(device); mtr_t = torch.from_numpy(mtr).float().to(device)
    C = int(ytr.max()) + 1; MC = C > 2          # multi-class (multimodal benchmarks) -> accuracy metric
    model = DEF(V, D, classes=C, hidden=args.hidden, combine=args.combine, path=args.path, vrel=args.vrel,
                vdiscount=args.vdiscount, cgate=args.cgate, jointnorm=args.jointnorm, fdiscount=args.fdiscount,
                pvdeep=args.pvdeep, pvrefine=args.pvrefine, xmodal=args.xmodal, vagg=args.vagg,
                xrefjoint=args.xrefjoint).to(device)
    model.disc = args.disc; model.xref = args.xref
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(Ntr)
        for st in range(0, Ntr, args.batch_size):
            b = perm[st:st + args.batch_size]; xb = [Xtr_v[v][b] for v in range(V)]; yb = ytr_t[b]; kl = args.kl_reg
            if args.xref:
                logits, evs = model.forward_xref(xb, mtr_t[:, b])
                loss = F.cross_entropy(logits, yb) + args.w_aux * sum(edl_loss(e, yb, ep + 1, kl, C) for e in evs) / V
            elif args.disc:
                logits, evs = model.forward_disc(xb, mtr_t[:, b])
                loss = F.cross_entropy(logits, yb) + args.w_aux * sum(edl_loss(e, yb, ep + 1, kl, C) for e in evs) / V
            else:
                a_final, a_joint, evs = model(xb, mtr_t[:, b])
                loss = edl_loss(a_final - 1.0, yb, ep + 1, kl, C) + args.w_aux * (
                    edl_loss(a_joint - 1.0, yb, ep + 1, kl, C) + sum(edl_loss(e, yb, ep + 1, kl, C) for e in evs) / V)
            opt.zero_grad(); loss.backward(); opt.step()

    def evalc(Xc, mc):
        model.eval()
        with torch.no_grad():
            xs = [torch.from_numpy(Xc[v]).float().to(device) for v in range(V)]
            mt = torch.from_numpy(mc).float().to(device)
            if args.xref:
                logits, _ = model.forward_xref(xs, mt)
                return torch.softmax(logits, 1).cpu().numpy()              # (N,C) probs
            if args.disc:
                logits, _ = model.forward_disc(xs, mt)
                return torch.softmax(logits, 1).cpu().numpy()              # (N,C) probs
            a_final, _, _ = model(xs, mt)
            return (a_final / a_final.sum(1, keepdim=True)).cpu().numpy()   # (N,C) probs

    def metric(y, P):
        return float((P.argmax(1) == y).mean()) if MC else auc(y, P[:, 1])

    P = evalc(Xva, mva); clean = metric(yva, P)
    sc = P if MC else P[:, 1]
    mj = {"method": args.method, "level": args.level, "fold": args.fold, "classes": C,
          ("acc_last" if MC else "auc_last"): clean, "n_val": int(len(yva))}
    if not MC:
        mj["auprc_last"] = float(average_precision_score(yva, P[:, 1])) if len(np.unique(yva)) >= 2 else float("nan")

    sweep = FORMAL_SWEEP
    rows = []
    rscores = {"y": np.asarray(yva)}
    for corr, lv in sweep:
        if corr == "clean":
            Xc, mc = Xva, mva
        else:
            Xc, mc = apply_corruption(Xva.copy(), mva.copy(), corr, lv, corruption_seed)
        Pc = evalc(Xc, mc)
        a = metric(yva, Pc)
        if not np.isfinite(Pc).all() or not np.isfinite(a):
            raise ValueError(f"non-finite result for {corr}_{lv:.2f}")
        if Pc.ndim == 2 and not np.allclose(Pc.sum(1), 1.0, atol=1e-5):
            raise ValueError(f"invalid probability rows for {corr}_{lv:.2f}")
        rows.append({"corruption": corr, "corruption_level": lv, "auc70_last": a,
                     "clean_last": clean, "drop_from_clean": clean - a,
                     "corruption_seed": corruption_seed})
        rscores["%s_%.2f" % (corr, lv)] = Pc if MC else Pc[:, 1]
    verify_authorities_unchanged(authority)
    config = formal_run_config(seed=args.seed, fold=args.fold, corruption_seed=corruption_seed, authority=authority)
    artifacts = {"clean_scores": {"y": np.asarray(yva), "score": np.asarray(sc)},
                 "metrics": mj, "rows": rows, "robust_scores": rscores, "config": config}
    receipt = publish_complete_cell(lease, cd, rd, artifacts)
    print("DONE_COMPLETE", args.method, "fold", args.fold, "seed", args.seed,
          "corruption_seed", corruption_seed, "receipt", receipt, flush=True)


if __name__ == "__main__":
    main()
