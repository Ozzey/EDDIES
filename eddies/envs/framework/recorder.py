"""Full-HD video recording: main camera + close-up inset + status overlay."""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]


def _font(size: int):
    for p in _FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class VideoRecorder:
    def __init__(self, model: mujoco.MjModel, path: str | Path, main_camera: str,
                 detail_camera: str | None = None, *, width: int = 1920, height: int = 1080,
                 fps: int = 30, inset_scale: float = 0.36, crf: int = 20,
                 show_contact_forces_in_detail: bool = True):
        self.model = model
        self.path = Path(path)
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.main_camera, self.detail_camera = main_camera, detail_camera
        if model.vis.global_.offwidth < width or model.vis.global_.offheight < height:
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.main = mujoco.Renderer(model, self.height, self.width)
        self.detail = None
        if detail_camera:
            self.iw = int(round(self.width * inset_scale / 8)) * 8
            self.ih = int(round(self.height * inset_scale / 8)) * 8
            self.detail = mujoco.Renderer(model, self.ih, self.iw)
            self.detail_opt = mujoco.MjvOption()
            if show_contact_forces_in_detail:
                self.detail_opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            model.vis.scale.forcewidth = 0.02
            model.vis.map.force = 0.01
        self.font = _font(max(18, self.height // 42))
        self.small = _font(max(14, self.height // 56))
        self.writer = imageio.get_writer(
            str(self.path), fps=self.fps, codec="libx264", macro_block_size=8,
            ffmpeg_params=["-crf", str(crf), "-preset", "veryfast", "-pix_fmt", "yuv420p"],
            ffmpeg_log_level="error")
        self.frames = 0

    def capture(self, data: mujoco.MjData, title: str = "", lines: list[str] | None = None) -> None:
        self.main.update_scene(data, camera=self.main_camera)
        frame = self.main.render()
        img = Image.fromarray(frame)
        if self.detail is not None:
            self.detail.update_scene(data, camera=self.detail_camera, scene_option=self.detail_opt)
            inset = Image.fromarray(self.detail.render())
            x0 = self.width - self.iw - 16
            y0 = 16
            border = Image.new("RGB", (self.iw + 6, self.ih + 6), (20, 20, 20))
            img.paste(border, (x0 - 3, y0 - 3))
            img.paste(inset, (x0, y0))
        draw = ImageDraw.Draw(img, "RGBA")
        text = [title] + list(lines or [])
        if any(text):
            pad = 10
            lh = int(self.font.size * 1.3)
            box_h = pad * 2 + lh * len(text)
            box_w = int(self.width * 0.46)
            draw.rectangle([12, 12, 12 + box_w, 12 + box_h], fill=(0, 0, 0, 140))
            for i, line in enumerate(text):
                f = self.font if i == 0 else self.small
                draw.text((12 + pad, 12 + pad + i * lh), str(line)[:110], font=f,
                          fill=(255, 255, 255, 255))
        self.writer.append_data(np.asarray(img))
        self.frames += 1

    def close(self) -> None:
        try:
            self.writer.close()
        finally:
            for r in (self.main, self.detail):
                if r is not None:
                    try:
                        r.close()
                    except Exception:
                        pass


def render_still(model, data, camera: str, path: str | Path, width=1280, height=720) -> None:
    r = mujoco.Renderer(model, height, width)
    try:
        r.update_scene(data, camera=camera)
        Image.fromarray(r.render()).save(str(path))
    finally:
        r.close()
