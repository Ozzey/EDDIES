"""RTX video recording: the task's own cameras (main + close-up inset), 1920x1080 H.264.

The task's MuJoCo cameras (``main_camera`` / ``detail_camera``, incl. body-mounted and
tracking cameras) are reproduced as USD cameras whose world pose and vertical field of
view are copied from the kinematic mirror every frame, so the Isaac videos show exactly
the same views as the MuJoCo videos.  Frames come from Replicator RGB annotators on
offscreen render products (rendering never advances physics); a status overlay is drawn
with Pillow and frames are piped to the system ffmpeg (libx264, CRF 18, yuv420p).  MuJoCo
floor planes (not rendered by the converter) are rebuilt with their texture.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(size):
    try:
        return ImageFont.truetype(_FONT, size)
    except OSError:
        return ImageFont.load_default()


def add_studio_lighting(stage) -> None:
    from pxr import Gf, UsdGeom, UsdLux
    if stage.GetPrimAtPath("/BistlLights").IsValid():
        return
    UsdGeom.Xform.Define(stage, "/BistlLights")
    dome = UsdLux.DomeLight.Define(stage, "/BistlLights/Dome")
    dome.CreateIntensityAttr(900.0)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/BistlLights/Sun")
    sun.CreateIntensityAttr(2600.0)
    sun.CreateAngleAttr(0.8)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 25.0, 35.0))


def add_floor(stage, m, d) -> None:
    """The USD converter does not render MuJoCo planes: rebuild every world plane as a
    visual-only textured quad with the MuJoCo material (texture, colour, texrepeat)."""
    import mujoco
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    from eddies.isaac.env import CACHE
    if stage.GetPrimAtPath("/BistlFloor").IsValid():
        return
    UsdGeom.Xform.Define(stage, "/BistlFloor")
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE or m.geom_bodyid[g] != 0:
            continue
        sx, sy = float(m.geom_size[g, 0]) or 50.0, float(m.geom_size[g, 1]) or 50.0
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        corners = [R @ np.array([x, y, -0.0002]) + p for x, y in ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy))]
        mesh = UsdGeom.Mesh.Define(stage, f"/BistlFloor/plane_{g}")
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, c)) for c in corners])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, R[:, 2]))] * 4)
        mat_id = int(m.geom_matid[g])
        rgba = m.mat_rgba[mat_id] if mat_id >= 0 else m.geom_rgba[g]
        shader_path = f"/BistlFloor/mat_{g}"
        mat = UsdShade.Material.Define(stage, shader_path)
        sh = UsdShade.Shader.Define(stage, shader_path + "/surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        tex = int(m.mat_texid[mat_id][1]) if mat_id >= 0 else -1
        if tex >= 0 and m.tex_nchannel[tex] >= 3:
            w, h, nc = int(m.tex_width[tex]), int(m.tex_height[tex]), int(m.tex_nchannel[tex])
            img = m.tex_data[m.tex_adr[tex]:m.tex_adr[tex] + w * h * nc].reshape(h, w, nc)[..., :3]
            png = CACHE / "textures" / f"plane_tex_{tex}_{w}x{h}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            if not png.exists():
                Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8)).save(png)
            rep = m.mat_texrepeat[mat_id]
            ru, rv = (rep[0] * 2 * sx, rep[1] * 2 * sy) if m.mat_texuniform[mat_id] else (rep[0], rep[1])
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                    UsdGeom.Tokens.faceVarying).Set(
                [Gf.Vec2f(0, 0), Gf.Vec2f(float(ru), 0), Gf.Vec2f(float(ru), float(rv)), Gf.Vec2f(0, float(rv))])
            reader = UsdShade.Shader.Define(stage, shader_path + "/st")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tx = UsdShade.Shader.Define(stage, shader_path + "/tex")
            tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(png))
            tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*map(float, rgba[:3]), 1.0))
            tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgba[:3])))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


class IsaacRecorder:
    def __init__(self, bridge, path, main_camera: str, detail_camera: str | None, *,
                 width: int = 1920, height: int = 1080, fps: int = 30, inset_scale: float = 0.36):
        import omni.replicator.core as rep
        from pxr import UsdGeom
        self.bridge, self.width, self.height, self.fps = bridge, int(width), int(height), int(fps)
        m = bridge.m
        stage = bridge.stage
        add_studio_lighting(stage)
        add_floor(stage, m, bridge.d)
        UsdGeom.Xform.Define(stage, "/BistlCams")
        self.cams = {}
        self.annot = {}
        self.rps = []                     # render products: destroyed in close() (GPU memory)
        self.iw = int(round(self.width * inset_scale / 8)) * 8
        self.ih = int(round(self.height * inset_scale / 8)) * 8
        for role, name, (w, h) in (("main", main_camera, (self.width, self.height)),
                                   ("detail", detail_camera, (self.iw, self.ih))):
            if not name:
                continue
            cid = int(m.camera(name).id)
            prim_path = f"/BistlCams/{role}"
            cam = UsdGeom.Camera.Define(stage, prim_path)
            f = 18.0
            va = 2 * f * np.tan(np.deg2rad(float(m.cam_fovy[cid])) / 2)
            cam.CreateFocalLengthAttr(f)
            cam.CreateVerticalApertureAttr(float(va))
            cam.CreateHorizontalApertureAttr(float(va * w / h))
            cam.CreateClippingRangeAttr((0.01, 100.0))
            xf = UsdGeom.Xformable(cam)
            xf.ClearXformOpOrder()
            self.cams[role] = (cid, xf.AddTransformOp())
            rp = rep.create.render_product(prim_path, (w, h))
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach([rp])
            self.annot[role] = ann
            self.rps.append(rp)
        self.font, self.small = _font(max(18, self.height // 42)), _font(max(14, self.height // 56))
        self.path = Path(path)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
             str(self.path)], stdin=subprocess.PIPE)
        self.frames = 0
        self._warm = False

    def _update_cameras(self) -> None:
        from pxr import Gf
        d = self.bridge.d
        for _, (cid, op) in self.cams.items():
            R = d.cam_xmat[cid].reshape(3, 3)
            p = d.cam_xpos[cid]
            M = Gf.Matrix4d(R[0, 0], R[1, 0], R[2, 0], 0.0,
                            R[0, 1], R[1, 1], R[2, 1], 0.0,
                            R[0, 2], R[1, 2], R[2, 2], 0.0,
                            float(p[0]), float(p[1]), float(p[2]), 1.0)
            op.Set(M)

    def _grab(self, role):
        data = self.annot[role].get_data()
        if data is None or not hasattr(data, "shape") or data.size == 0:
            return None
        return np.asarray(data)[..., :3]

    def capture(self, title: str = "", lines: list | None = None) -> None:
        import omni.replicator.core as rep
        self._update_cameras()
        # render without stepping physics; extra subframes when warming up the path tracer
        rep.orchestrator.step(rt_subframes=1 if self._warm else 8, pause_timeline=False, delta_time=0.0)
        self._warm = True
        main = self._grab("main")
        if main is None:
            return
        img = Image.fromarray(np.ascontiguousarray(main, dtype=np.uint8))
        if "detail" in self.annot:
            det = self._grab("detail")
            if det is not None:
                x0, y0 = self.width - self.iw - 16, 16
                img.paste(Image.new("RGB", (self.iw + 6, self.ih + 6), (20, 20, 20)), (x0 - 3, y0 - 3))
                img.paste(Image.fromarray(np.ascontiguousarray(det, dtype=np.uint8)), (x0, y0))
        draw = ImageDraw.Draw(img, "RGBA")
        text = [title] + list(lines or [])
        pad, lh = 10, int(self.font.size * 1.3)
        draw.rectangle([12, 12, 12 + int(self.width * 0.46), 12 + pad * 2 + lh * len(text)], fill=(0, 0, 0, 140))
        for i, line in enumerate(text):
            draw.text((12 + pad, 12 + pad + i * lh), str(line)[:110], font=self.font if i == 0 else self.small,
                      fill=(255, 255, 255, 255))
        self.proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=120)
        except Exception:
            self.proc.kill()
        # release the render products (their GPU frame buffers): one process records many
        # episodes, and leaked 1080p render targets otherwise accumulate on the GPU
        for ann in self.annot.values():
            try:
                ann.detach()
            except Exception:
                pass
        for rp in self.rps:
            try:
                rp.destroy()
            except Exception:
                pass
        self.annot, self.rps = {}, []
