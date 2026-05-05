# hyperparameter grid search
# trains each config, evaluates on val set, logs results, plots parallel coordinates

import gc
import json
import time
import torch
import os
import plotly.graph_objects as go

from utils.training.simple_train import simple_train, simple_eval


def hypersearch(
    build_model,        # fn(config_dict) -> nn.Module
    configs,            # [{"layers": 2, "embed_dim": 256, ...}, ...]
    train_loader,
    val_loader,
    max_batches=10000,
    accumulation_steps=1,
    clip_grad_norm=1.0,
    lr=3e-4,
    weight_decay=0.1,
    batches_per_log=100,
    weights_folder="hypersearch_weights",
    metadata_folder="hypersearch_metadata",
):
    os.makedirs(weights_folder, exist_ok=True)
    os.makedirs(metadata_folder, exist_ok=True)

    log_path = os.path.join(metadata_folder, "results.json")

    # load existing results for resume
    results = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")

    # configs already done (frozen dict keys as tuple for comparison)
    done_keys = {tuple(sorted(r['config'].items())) for r in results}

    print(f"Hypersearch: {len(configs)} configs, {max_batches} batches each")

    for i, config in enumerate(configs):
        config_key = tuple(sorted(config.items()))
        if config_key in done_keys:
            print(f"\n[{i+1}/{len(configs)}] Skipping (already done): {config}")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs)}] Config: {config}")
        print(f"{'='*60}")

        model = build_model(config)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Params: {num_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # config-specific weight folder (no checkpointing between configs — train from scratch)
        config_name = "_".join(f"{k}{v}" for k, v in config.items())
        config_weights = os.path.join(weights_folder, config_name)

        start_time = time.time()
        train_loss = simple_train(
            model, train_loader, optimizer,
            max_batches=max_batches,
            accumulation_steps=accumulation_steps,
            clip_grad_norm=clip_grad_norm,
            save_folder_path=config_weights,
            batches_per_log=batches_per_log,
            # save only at the end (simple_train does a final save)
            batches_per_save=max_batches + 1,
        )
        train_time = time.time() - start_time

        val_loss = simple_eval(model, val_loader)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Time: {train_time:.1f}s")

        result = {
            'config': config,
            'params': num_params,
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            'train_time': round(train_time, 1),
        }
        results.append(result)

        # save after each config so we can resume
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)

        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    # summary
    print(f"\n{'='*60}")
    print("HYPERSEARCH COMPLETE")
    print(f"{'='*60}")

    by_val = sorted(results, key=lambda x: x['val_loss'])
    print("\nResults (sorted by val loss):")
    for r in by_val:
        print(f"  {r['config']} | val: {r['val_loss']:.4f} | train: {r['train_loss']:.4f} | params: {r['params']:,}")

    plot_results(results, metadata_folder)
    return results


def plot_results(results, metadata_folder):
    if len(results) < 2:
        return

    # collect all config keys that vary across configs
    all_keys = list(results[0]['config'].keys())
    varying_keys = [k for k in all_keys if len(set(r['config'][k] for r in results)) > 1]

    dims = []
    for key in varying_keys:
        vals = [r['config'][key] for r in results]
        dims.append(dict(label=key, values=vals, range=[min(vals), max(vals)]))

    dims.append(dict(label='params', values=[r['params'] for r in results]))

    val_losses = [r['val_loss'] for r in results]
    dims.append(dict(label='val_loss', values=val_losses, range=[min(val_losses), max(val_losses)]))

    fig = go.Figure()
    fig.add_trace(go.Parcoords(
        line=dict(
            color=val_losses,
            # green = low loss (good), red = high loss (bad)
            colorscale='RdYlGn_r',
            showscale=True,
            cmin=min(val_losses),
            cmax=max(val_losses),
        ),
        dimensions=dims,
    ))

    fig.update_layout(
        title='Hypersearch Results (color = val loss, green = better)',
        font=dict(size=14),
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e',
        font_color='white',
    )

    html_path = os.path.join(metadata_folder, "results.html")
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    print(f"Plot saved: {html_path}")
