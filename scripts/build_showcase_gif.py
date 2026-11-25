"""Build a unified side-by-side WaterWorld showcase GIF in docs/ for all MARL methods.

Renders a synchronized 2x3 comparison grid on the exact same held-out PettingZoo
WaterWorld `base` seed (5 agents, n_coop=2, 30 lidar rays, pymunk physics):
  - Row 1 (Baselines & Ray-Transformer before `--coop`):
      1. GA (SO(2) Equivariant, 17 params)
      2. Double DQN (Ray-Transformer, 10.4K)
      3. MAPPO (Ray-Transformer, 18.9K)
  - Row 2 (MLP Reference & Cooperative MARL Upgrades `--coop`):
      4. MAPPO (MLP Reference, 128.1K)
      5. Hybrid QMIX (Coop Ray-Tx, 10.4K)
      6. PPO-Coop (Actor-Comm + Attn-GRU + Shaping)

Every tile displays a color-coded method badge and live per-step counters
(Team Return, Food captures, Poison hits) so all 6 methods can be compared side-by-side.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from waterworld_marl.run_experiments import build_action_fn_from_result
from waterworld_marl.scenarios import (
    SCENARIOS,
    build_agent_phase_vec,
    make_waterworld_env,
)


@dataclass
class PanelTrace:
    title: str
    subtitle: str
    accent_rgb: tuple[int, int, int]
    frames: list[np.ndarray]
    hud_lines: list[str]


def _load_font(
    size: int, bold: bool = False, mono: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if mono:
        candidates = [
            "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/courbd.ttf" if bold else "C:/Windows/Fonts/cour.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _record_waterworld_panel(
    title: str,
    subtitle: str,
    accent_rgb: tuple[int, int, int],
    algo: str,
    ckpt_path: Path,
    seed: int,
    device: str,
    stride: int,
    max_steps: int,
) -> PanelTrace:
    sc = SCENARIOS["base"]
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    action_fn = build_action_fn_from_result(sc, algo, payload, device=device)

    env = make_waterworld_env(sc, render_mode="rgb_array")
    frames: list[np.ndarray] = []
    hud_lines: list[str] = []
    try:
        obs_dict, _ = env.reset(seed=int(seed))
        if hasattr(action_fn, "reset"):
            action_fn.reset()
        agent_ids = list(env.possible_agents)
        phase = build_agent_phase_vec(len(agent_ids))

        team_ret = 0.0
        food_events = 0
        poison_hits = 0

        fr0 = env.render()
        if fr0 is not None:
            frames.append(fr0)
            hud_lines.append("Ret:  +0.0 | Food:  0 | Poison:  0")

        for step in range(1, max_steps + 1):
            if not env.agents:
                break
            obs_mat = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
            acts = np.clip(action_fn(obs_mat, phase), -1.0, 1.0).astype(np.float32)
            act_dict = {agent_ids[i]: acts[i] for i in range(len(agent_ids))}
            obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(act_dict)
            r_vec = np.array([float(rew_dict[a]) for a in agent_ids], dtype=np.float32)
            o_next = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)

            team_ret += float(r_vec.sum())
            food_events += int(np.sum(r_vec >= 4.0))
            poison_hits += int(np.sum(o_next[:, -1] > 0.5))

            done = (not env.agents) or any(term_dict.values()) or any(trunc_dict.values())
            if step % stride == 0 or done or step == max_steps:
                fr = env.render()
                if fr is not None:
                    frames.append(fr)
                    hud_lines.append(
                        f"Ret:{team_ret:+6.1f} | Food:{food_events:2d} | Poison:{poison_hits:2d}"
                    )
            if done:
                break
    finally:
        env.close()

    print(f"  [WaterWorld] {title:28s} -> {len(frames)} frames | {hud_lines[-1]}", flush=True)
    return PanelTrace(
        title=title,
        subtitle=subtitle,
        accent_rgb=accent_rgb,
        frames=frames,
        hud_lines=hud_lines,
    )


def _resize_square(frame: np.ndarray, size: int) -> Image.Image:
    img = Image.fromarray(frame)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size[0] == size and img.size[1] == size:
        return img
    return img.resize((size, size), resample=Image.Resampling.LANCZOS)


def compose_showcase_gif(
    row1_panels: list[PanelTrace],
    row2_panels: list[PanelTrace],
    seed: int,
    out_gif: Path,
    tile_size: int = 264,
    stride: int = 4,
    max_steps: int = 500,
    fps: int = 11,
    end_hold_frames: int = 14,
    n_colors: int = 96,
    preview_png: Path | None = None,
) -> None:
    """Compose a 2x3 labeled WaterWorld multi-method animated GIF with live HUD."""
    pad_x = 12
    pad_y = 10
    header_h = 54
    section_h = 26
    badge_h = 34
    hud_h = 24
    tile_h = badge_h + tile_size + hud_h

    canvas_w = 3 * tile_size + 4 * pad_x
    canvas_h = (
        header_h
        + pad_y
        + section_h
        + 6
        + tile_h
        + pad_y
        + section_h
        + 6
        + tile_h
        + pad_y
    )

    f_title = _load_font(16, bold=True)
    f_sub = _load_font(11, bold=False)
    f_sec = _load_font(11, bold=True)
    f_badge = _load_font(12, bold=True)
    f_bsub = _load_font(10, bold=False)
    f_hud = _load_font(10 if tile_size <= 268 else 11, bold=True, mono=True)

    n_frames = max(
        max(len(p.frames) for p in row1_panels),
        max(len(p.frames) for p in row2_panels),
    )

    bg_main = (15, 23, 42)
    bg_header = (9, 14, 28)
    bg_section = (30, 41, 59)
    bg_card = (22, 32, 50)
    bg_hud = (17, 24, 39)
    border_col = (51, 65, 85)

    base_canvas = Image.new("RGB", (canvas_w, canvas_h), bg_main)
    d0 = ImageDraw.Draw(base_canvas)

    # Top header
    d0.rectangle([0, 0, canvas_w, header_h], fill=bg_header)
    d0.text(
        (pad_x, 8),
        "PettingZoo WaterWorld — All 6 MARL Methods Side-by-Side",
        fill=(248, 250, 252),
        font=f_title,
    )
    d0.text(
        (pad_x, 30),
        f"Synchronized held-out pymunk episode (base: 5 agents, n_coop=2, seed={seed})  "
        "|  Green: Food (+10)  •  Red: Poison (-1)  •  Purple: Agents",
        fill=(148, 163, 184),
        font=f_sub,
    )

    row1_sec_y = header_h + pad_y
    row1_tile_y = row1_sec_y + section_h + 6
    row2_sec_y = row1_tile_y + tile_h + pad_y
    row2_tile_y = row2_sec_y + section_h + 6

    # Section banners
    d0.rounded_rectangle(
        [pad_x, row1_sec_y, canvas_w - pad_x, row1_sec_y + section_h],
        radius=5,
        fill=bg_section,
    )
    d0.text(
        (pad_x + 10, row1_sec_y + 5),
        "ROW 1  •  Evolution & Baseline Ray-Transformer Architectures (before --coop)",
        fill=(226, 232, 240),
        font=f_sec,
    )

    d0.rounded_rectangle(
        [pad_x, row2_sec_y, canvas_w - pad_x, row2_sec_y + section_h],
        radius=5,
        fill=bg_section,
    )
    d0.text(
        (pad_x + 10, row2_sec_y + 5),
        "ROW 2  •  MLP Reference & Cooperative MARL Upgrades (Hybrid QMIX & PPO-Coop Comm+Attn-GRU)",
        fill=(226, 232, 240),
        font=f_sec,
    )

    def _draw_card_chrome(panels: list[PanelTrace], top_y: int) -> None:
        for col, p in enumerate(panels):
            x0 = pad_x + col * (tile_size + pad_x)
            x1 = x0 + tile_size
            d0.rectangle([x0 - 1, top_y - 1, x1 + 1, top_y + tile_h + 1], outline=border_col, width=1)
            d0.rectangle([x0, top_y, x1, top_y + badge_h], fill=bg_card)
            d0.rectangle([x0, top_y, x0 + 5, top_y + badge_h], fill=p.accent_rgb)
            d0.text((x0 + 11, top_y + 3), p.title, fill=(248, 250, 252), font=f_badge)
            d0.text((x0 + 11, top_y + 18), p.subtitle, fill=(148, 163, 184), font=f_bsub)

    _draw_card_chrome(row1_panels, row1_tile_y)
    _draw_card_chrome(row2_panels, row2_tile_y)

    composed_rgb: list[Image.Image] = []
    for idx in range(n_frames):
        canvas = base_canvas.copy()
        draw = ImageDraw.Draw(canvas)

        step_num = min(idx * stride, max_steps)
        if idx == n_frames - 1:
            step_num = max_steps

        step_txt = f"Step {step_num:03d} / {max_steps}"
        draw.text((canvas_w - 118, 10), step_txt, fill=(56, 189, 248), font=f_hud)
        frac = step_num / float(max(max_steps, 1))
        draw.rectangle([0, header_h - 3, canvas_w, header_h], fill=(30, 41, 59))
        draw.rectangle([0, header_h - 3, int(canvas_w * frac), header_h], fill=(56, 189, 248))

        for row_panels, top_y in ((row1_panels, row1_tile_y), (row2_panels, row2_tile_y)):
            for col, p in enumerate(row_panels):
                f_idx = min(idx, len(p.frames) - 1)
                x0 = pad_x + col * (tile_size + pad_x)
                img_y = top_y + badge_h
                hud_y = img_y + tile_size

                tile_img = _resize_square(p.frames[f_idx], tile_size)
                canvas.paste(tile_img, (x0, img_y))

                draw.rectangle([x0, hud_y, x0 + tile_size, hud_y + hud_h], fill=bg_hud)
                draw.rectangle([x0, hud_y, x0 + 5, hud_y + hud_h], fill=p.accent_rgb)
                draw.text((x0 + 10, hud_y + 5), p.hud_lines[f_idx], fill=(226, 232, 240), font=f_hud)

        composed_rgb.append(canvas)

    if preview_png is not None and composed_rgb:
        preview_png.parent.mkdir(parents=True, exist_ok=True)
        composed_rgb[int(0.75 * (len(composed_rgb) - 1))].save(preview_png)

    durations = [int(round(1000.0 / max(fps, 1)))] * len(composed_rgb)
    if durations:
        durations[-1] = int(round(1000.0 / max(fps, 1))) * max(end_hold_frames, 1)

    sample_indices = np.linspace(0, len(composed_rgb) - 1, min(6, len(composed_rgb)), dtype=int)
    palette_strip = Image.new("RGB", (canvas_w, canvas_h * len(sample_indices)))
    for k, si in enumerate(sample_indices):
        palette_strip.paste(composed_rgb[int(si)], (0, k * canvas_h))
    master_pal = palette_strip.quantize(
        colors=n_colors,
        method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.NONE,
    )

    quantized_frames = [
        fr.quantize(palette=master_pal, dither=Image.Dither.NONE) for fr in composed_rgb
    ]

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    quantized_frames[0].save(
        out_gif,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    size_mb = out_gif.stat().st_size / (1024 * 1024)
    print(
        f"Saved {out_gif} ({canvas_w}x{canvas_h}, {len(quantized_frames)} frames, {size_mb:.2f} MB)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified 2x3 WaterWorld methods showcase GIF in docs/")
    parser.add_argument("--seed", type=int, default=950001)
    parser.add_argument("--tile-size", type=int, default=264)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=11)
    parser.add_argument("--colors", type=int, default=96)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "methods_showcase.gif")
    parser.add_argument("--preview-png", type=Path, default=None)
    args = parser.parse_args()

    print(f"Recording 6 synchronized WaterWorld methods on held-out seed={args.seed}...", flush=True)
    row1_panels = [
        _record_waterworld_panel(
            title="GA (SO(2) Equivariant)",
            subtitle="Neuroevolution • 17 params • CRN",
            accent_rgb=(37, 99, 235),
            algo="GA",
            ckpt_path=ROOT / "runs/waterworld_coop/checkpoints/base_ga.pt",
            seed=args.seed,
            device=args.device,
            stride=args.stride,
            max_steps=args.max_steps,
        ),
        _record_waterworld_panel(
            title="DQN (Ray-Transformer)",
            subtitle="Baseline value-based • 10.4K params",
            accent_rgb=(217, 119, 6),
            algo="DQN",
            ckpt_path=ROOT / "runs/waterworld_tx/checkpoints/base_dqn.pt",
            seed=args.seed,
            device=args.device,
            stride=args.stride,
            max_steps=args.max_steps,
        ),
        _record_waterworld_panel(
            title="PPO (Ray-Transformer)",
            subtitle="Baseline CTDE MAPPO • 18.9K params",
            accent_rgb=(5, 150, 105),
            algo="PPO",
            ckpt_path=ROOT / "runs/waterworld_tx/checkpoints/base_ppo.pt",
            seed=args.seed,
            device=args.device,
            stride=args.stride,
            max_steps=args.max_steps,
        ),
    ]

    row2_panels = [
        _record_waterworld_panel(
            title="PPO (MLP Reference)",
            subtitle="CTDE MAPPO • 128.1K params • dense",
            accent_rgb=(99, 102, 241),
            algo="PPO",
            ckpt_path=ROOT / "runs/waterworld/checkpoints/base_ppo.pt",
            seed=args.seed,
            device=args.device,
            stride=args.stride,
            max_steps=args.max_steps,
        ),
        _record_waterworld_panel(
            title="Hybrid QMIX (Coop-Tx)",
            subtitle="Monotonic mixer + curriculum • 5/5 vs DQN",
            accent_rgb=(180, 83, 9),
            algo="DQN",
            ckpt_path=ROOT / "runs/waterworld_coop/checkpoints/base_dqn.pt",
            seed=args.seed,
            device=args.device,
            stride=args.stride,
            max_steps=args.max_steps,
        ),
        _record_waterworld_panel(
            title="PPO-Coop (Salience+Comm)",
            subtitle="Ray Salience + Dodge + Comm • #1 (+108.3)",
            accent_rgb=(16, 185, 129),
            algo="PPO",
            ckpt_path=ROOT / "runs/waterworld_coop/checkpoints/base_ppo.pt",
            seed=args.seed,
            device=args.device,
            stride=args.stride,
            max_steps=args.max_steps,
        ),
    ]

    compose_showcase_gif(
        row1_panels=row1_panels,
        row2_panels=row2_panels,
        seed=args.seed,
        out_gif=args.out,
        tile_size=args.tile_size,
        stride=args.stride,
        max_steps=args.max_steps,
        fps=args.fps,
        n_colors=args.colors,
        preview_png=args.preview_png,
    )


if __name__ == "__main__":
    main()
