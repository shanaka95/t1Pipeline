import argparse
import os
from typing import Any, Dict, List, Tuple

import yaml
import matplotlib.pyplot as plt


def format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{k}: {format_value(v)}" for k, v in value.items())
        return "{" + inner + "}"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def dict_to_lines(d: Dict[str, Any], indent: int = 0) -> List[str]:
    lines: List[str] = []
    pad = "  " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{pad}- {key}:")
            lines.extend(dict_to_lines(value, indent + 1))
        else:
            val_str = format_value(value)
            lines.append(f"{pad}- {key}: {val_str}")
    return lines


def build_sections(cfg: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    sections: List[Tuple[str, List[str]]] = []

    # Work dir & general
    general: List[str] = []
    for k in [
        "work_dir",
        "phase",
        "device",
        "seed",
        "model_saved_name",
    ]:
        if k in cfg:
            general.append(f"- {k}: {format_value(cfg[k])}")
    sections.append(("General", general))

    # Feeder
    feeder_lines: List[str] = []
    if "feeder" in cfg:
        feeder_lines.append(f"- class: {cfg['feeder']}")
    if "train_feeder_args" in cfg:
        feeder_lines.append("- train_feeder_args:")
        feeder_lines.extend(dict_to_lines(cfg["train_feeder_args"], indent=1))
    if "test_feeder_args" in cfg:
        feeder_lines.append("- test_feeder_args:")
        feeder_lines.extend(dict_to_lines(cfg["test_feeder_args"], indent=1))
    sections.append(("Feeder", feeder_lines))

    # Model
    model_lines: List[str] = []
    if "model" in cfg:
        model_lines.append(f"- class: {cfg['model']}")
    if "model_args" in cfg:
        model_lines.append("- model_args:")
        model_lines.extend(dict_to_lines(cfg["model_args"], indent=1))
    if "weights" in cfg:
        model_lines.append(f"- weights: {format_value(cfg['weights'])}")
    if "ignore_weights" in cfg:
        model_lines.append(f"- ignore_weights: {format_value(cfg['ignore_weights'])}")
    sections.append(("Model", model_lines))

    # Optimization & Training
    optim_train_lines: List[str] = []
    for k in [
        "optimizer",
        "base_lr",
        "weight_decay",
        "lr_decay_rate",
        "step",
        "warm_up_epoch",
    ]:
        if k in cfg:
            optim_train_lines.append(f"- {k}: {format_value(cfg[k])}")

    optim_train_lines.append("")
    optim_train_lines.append("- training:")
    for k in [
        "batch_size",
        "test_batch_size",
        "num_epoch",
        "nesterov",
        "start_epoch",
        "save_score",
        "log_interval",
        "save_interval",
        "save_epoch",
        "eval_interval",
        "print_log",
        "show_topk",
        "num_worker",
    ]:
        if k in cfg:
            optim_train_lines.append(f"  - {k}: {format_value(cfg[k])}")
    sections.append(("Optimization & Training", optim_train_lines))

    # Regularization & Extras
    reg_lines: List[str] = []
    for k in [
        "early_stopping",
        "patience",
        "min_delta",
        "use_joint_stream",
        "use_bone_stream",
        "use_motion_stream",
        "label_smoothing",
        "gradient_clip",
    ]:
        if k in cfg:
            reg_lines.append(f"- {k}: {format_value(cfg[k])}")
    sections.append(("Regularization & Extras", reg_lines))

    return sections


def draw_sections(sections: List[Tuple[str, List[str]]], title: str, out_path: str) -> None:
    # Layout: 2 rows x 2 columns (4 panels). If more, adjust dynamically.
    num = len(sections)
    cols = 2
    rows = (num + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 10), constrained_layout=True)
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=18, fontweight="bold")

    for idx, (section_title, lines) in enumerate(sections):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        ax.axis("off")

        display_text = section_title + "\n" + "\n".join(lines)
        ax.text(
            0.01,
            0.99,
            display_text,
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7F9FB", edgecolor="#AEB6BF"),
        )

    # Hide any unused axes
    for idx in range(num, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CTR-GCN config into an infographic figure")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (PNG). If not provided, will be placed under visualizations/ with an inferred name.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    title = f"CTR-GCN Config: {cfg_name}"

    out_path = args.out
    if out_path is None:
        out_dir = os.path.join("visualizations")
        out_path = os.path.join(out_dir, f"ctrgcn_config_{cfg_name}.png")

    sections = build_sections(cfg)
    draw_sections(sections, title, out_path)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()

