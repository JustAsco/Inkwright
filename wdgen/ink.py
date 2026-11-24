"""Ink primitives: a supersampled single-channel canvas and brush-like strokes.

Everything is drawn as ink coverage (alpha) only. The final PNG is a single ink
colour on transparency, so Wonderdraft / the map shading supplies the colour.
"""
import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

TAU = 2 * math.pi


# --------------------------------------------------------------------------- geometry

def chaikin(pts, iters=2):
    pts = np.asarray(pts, float)
    for _ in range(iters):
        if len(pts) < 3:
            break
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * len(q), 2))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def resample(pts, step):
    pts = np.asarray(pts, float)
    d = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total < 1e-6:
        return pts[[0, 0]], np.array([0.0, 0.0])
    n = max(2, int(total / step) + 1)
    ss = np.linspace(0, total, n)
    return np.stack([np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])], 1), ss


def clip_halfplane(poly, n, c):
    """Keep the part of polygon `poly` where dot(p, n) >= c (Sutherland-Hodgman)."""
    n = np.asarray(n, float)
    out = []
    P = [np.asarray(p, float) for p in poly]
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        da, db = a @ n - c, b @ n - c
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            out.append(a + (b - a) * (da / (da - db)))
    return out


def ellipse_pts(cx, cy, rx, ry, n=32, a0=0.0, a1=TAU):
    t = np.linspace(a0, a1, n, endpoint=(a1 - a0) < TAU - 1e-9)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def blob_pts(rng, cx, cy, r, n=7, irregular=0.3, squash=0.75):
    t = np.sort(rng.uniform(0, TAU, n))
    rr = r * (1 + rng.uniform(-irregular, irregular, n))
    return np.stack([cx + rr * np.cos(t), cy + rr * np.sin(t) * squash], 1)


# --------------------------------------------------------------------------- canvas

class Canvas:
    """Supersampled ink + paper masks. Coordinates are logical units; `res` output px per unit.

    `erase()` knocks out ink behind a surface and marks it as paper. Paper is either
    rendered white (filled style, like Wonderdraft's B/W buildings) or left
    transparent (pure line art, like the brush trees).
    """

    def __init__(self, w, h, res=2, ss=3):
        self.w, self.h, self.res, self.k = w, h, res, res * ss
        self.ss = ss
        self.mask = Image.new("L", (w * self.k, h * self.k), 0)
        self.paper = Image.new("L", (w * self.k, h * self.k), 0)
        self._d = ImageDraw.Draw(self.mask)
        self._p = ImageDraw.Draw(self.paper)
        self.washm = None  # optional colour wash layer (e.g. brown cliff shading)

    def _scale(self, pts):
        k = self.k
        return [(float(p[0]) * k, float(p[1]) * k) for p in pts]

    def fill(self, pts, value=255):
        if len(pts) >= 3:
            self._d.polygon(self._scale(pts), fill=value)

    def erase(self, pts):
        if len(pts) >= 3:
            sp = self._scale(pts)
            self._d.polygon(sp, fill=0)
            self._p.polygon(sp, fill=255)

    def wash(self, pts, value=255):
        """Add soft colour shading (max-blended, so overlapping bands don't fight)."""
        if len(pts) < 3:
            return
        if self.washm is None:
            self.washm = Image.new("L", self.mask.size, 0)
        tmp = Image.new("L", self.mask.size, 0)
        ImageDraw.Draw(tmp).polygon(self._scale(pts), fill=int(value))
        self.washm = ImageChops.lighter(self.washm, tmp)

    def clear(self, pts):
        """Remove ink, paper and wash (true transparency), e.g. ground hiding a canyon floor."""
        if len(pts) >= 3:
            sp = self._scale(pts)
            self._d.polygon(sp, fill=0)
            self._p.polygon(sp, fill=0)
            if self.washm is not None:
                ImageDraw.Draw(self.washm).polygon(sp, fill=0)

    def disc(self, x, y, r, value=255):
        k = self.k
        self._d.ellipse([(x - r) * k, (y - r) * k, (x + r) * k, (y + r) * k], fill=value)

    def to_image(self, ink=(30, 30, 30), paper=(255, 255, 255), filled=True, pad=6,
                 wash=(150, 98, 62), wash_alpha=0.95, seed=0):
        m = np.asarray(self.mask.reduce(self.ss), np.float32) / 255.0
        H, W = m.shape
        rgb = np.zeros((H, W, 3), np.float32)
        a = np.zeros((H, W), np.float32)

        def over(color, alpha):
            nonlocal rgb, a
            out = alpha + a * (1 - alpha)
            safe = np.maximum(out, 1e-6)[..., None]
            rgb = (np.array(color, np.float32) * alpha[..., None] + rgb * (a * (1 - alpha))[..., None]) / safe
            a = out

        if filled:
            over(paper, np.asarray(self.paper.reduce(self.ss), np.float32) / 255.0)
        if self.washm is not None and wash is not None:
            wm = self.washm.reduce(self.ss).filter(ImageFilter.GaussianBlur(1.2 * self.res / 2))
            grain = np.random.default_rng(seed).uniform(0.55, 1.0, (H, W)).astype(np.float32)
            over(wash, np.asarray(wm, np.float32) / 255.0 * grain * wash_alpha)
        over(ink, m)
        arr = np.dstack([rgb, a * 255]).round().clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGBA")
        bbox = Image.fromarray((a > 0.04).astype(np.uint8) * 255).getbbox()
        if bbox:
            l, t, r, b = bbox
            img = img.crop((max(0, l - pad), max(0, t - pad), min(img.width, r + pad), min(img.height, b + pad)))
        return img


# --------------------------------------------------------------------------- pen

class Pen:
    def __init__(self, cv, rng, width=1.6):
        self.cv, self.rng, self.width = cv, rng, width

    # -- core brush stroke -------------------------------------------------
    def stroke(self, pts, w=None, taper=(0.2, 0.2), taper_max=10.0, wobble=0.4, smooth=0,
               overshoot=0.0, pressure=0.15, min_w=0.2, caps=True):
        rng = self.rng
        w = self.width if w is None else w
        P = np.asarray(pts, float)
        if len(P) < 2:
            return
        if smooth:
            P = chaikin(P, smooth)
        if overshoot > 0:
            P = P.copy()
            for i, j in ((0, 1), (-1, -2)):
                d = P[i] - P[j]
                n = math.hypot(*d)
                if n > 1e-6:
                    P[i] = P[i] + d / n * rng.uniform(0, overshoot)
        P, s = resample(P, 0.25)
        total = s[-1]
        if total < 0.6:
            self.cv.disc(P[0][0], P[0][1], w * 0.45)
            return
        T = np.gradient(P, axis=0)
        T /= np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-9)[:, None]
        N = np.stack([-T[:, 1], T[:, 0]], 1)
        if wobble:
            amp = wobble * min(1.0, total / 25)
            k1, k2 = rng.uniform(25, 60), rng.uniform(8, 16)
            p1, p2 = rng.uniform(0, TAU, 2)
            off = amp * (0.7 * np.sin(TAU * s / k1 + p1) + 0.3 * np.sin(TAU * s / k2 + p2))
            P = P + N * off[:, None]
        l0 = min(taper[0] * total, taper_max)
        l1 = min(taper[1] * total, taper_max)
        prof = np.ones_like(s)
        if l0 > 0:
            prof *= np.clip(s / l0, 0, 1) ** 0.6
        if l1 > 0:
            prof *= np.clip((total - s) / l1, 0, 1) ** 0.6
        k3, p3 = rng.uniform(15, 40), rng.uniform(0, TAU)
        width = w * np.maximum(min_w, prof) * (1 + pressure * np.sin(TAU * s / k3 + p3))
        L = P + N * (width / 2)[:, None]
        R = P - N * (width / 2)[:, None]
        self.cv.fill(np.vstack([L, R[::-1]]))
        if caps:
            if l0 <= 0:
                self.cv.disc(P[0][0], P[0][1], width[0] / 2)
            if l1 <= 0:
                self.cv.disc(P[-1][0], P[-1][1], width[-1] / 2)

    def line(self, a, b, w=None, sketch=0.2, **kw):
        kw.setdefault("overshoot", 1.0)
        kw.setdefault("taper", (0.1, 0.15))
        kw.setdefault("wobble", 0.35)
        w = self.width if w is None else w
        self.stroke([a, b], w, **kw)
        if self.rng.random() < sketch:
            off = self.rng.normal(0, 0.5, 2)
            a2, b2 = np.asarray(a, float) + off, np.asarray(b, float) + off
            self.stroke([a2, b2], w * 0.5, taper=(0.35, 0.35), overshoot=2.0, wobble=0.5)

    def closed(self, pts, w=None, **kw):
        pts = list(pts)
        kw.setdefault("taper", (0.04, 0.04))
        self.stroke(pts + [pts[0], pts[1]], w, **kw)

    def blot(self, pts, jitter=0.3):
        P, _ = resample(list(pts) + [pts[0]], 1.0)
        P = P + self.rng.normal(0, jitter, P.shape)
        self.cv.fill(P)

    # -- hatching ----------------------------------------------------------
    def hatch(self, poly, angle_deg, spacing, w=None, coverage=(0.6, 1.0), jitter=0.5,
              taper=(0.3, 0.3), wobble=0.25):
        rng = self.rng
        poly = np.asarray(poly, float)
        if len(poly) < 3:
            return
        a = math.radians(angle_deg)
        d = np.array([math.cos(a), math.sin(a)])
        n = np.array([-d[1], d[0]])
        pn = poly @ n
        c = pn.min() + spacing * rng.uniform(0.3, 0.8)
        while c < pn.max():
            xs = []
            for i in range(len(poly)):
                p, q = poly[i], poly[(i + 1) % len(poly)]
                dp, dq = p @ n - c, q @ n - c
                if (dp < 0) != (dq < 0):
                    xs.append((p + (q - p) * (dp / (dp - dq))) @ d)
            xs.sort()
            for k in range(0, len(xs) - 1, 2):
                lo, hi = xs[k], xs[k + 1]
                L = hi - lo
                if L < 1.5:
                    continue
                cov = rng.uniform(*coverage)
                st = lo + rng.uniform(0, (1 - cov) * L)
                cc = c + rng.uniform(-jitter, jitter) * 0.4
                self.stroke([d * st + n * cc, d * (st + cov * L) + n * cc], w, taper=taper,
                            wobble=wobble, pressure=0.1)
            c += spacing * rng.uniform(0.8, 1.2)

    # -- organic marks -----------------------------------------------------
    def dab(self, x, y, size, angle=0.0, curve=0.45, w=None):
        """Crescent 'smile' brush mark like the needle marks on the reference trees."""
        t = np.linspace(-1, 1, 7)
        px, py = t * size, curve * size * (1 - t * t) * 0.6
        ca, sa = math.cos(angle), math.sin(angle)
        pts = np.stack([x + px * ca - py * sa, y + px * sa + py * ca], 1)
        self.stroke(pts, (w or self.width) * 1.15, taper=(0.45, 0.45), wobble=0.05,
                    min_w=0.05, pressure=0.05)

    def tuft(self, x, y, size=5.0):
        rng = self.rng
        for _ in range(rng.integers(3, 6)):
            dx = rng.uniform(-size * 0.4, size * 0.4)
            h = size * rng.uniform(0.6, 1.2)
            lean = dx * 0.8 + rng.uniform(-1.5, 1.5)
            pts = [(x + dx * 0.3, y), (x + dx * 0.6 + lean * 0.3, y - h * 0.55), (x + dx + lean, y - h)]
            self.stroke(pts, self.width * 0.7, taper=(0.05, 0.75), smooth=2, wobble=0.1)

    def bush(self, x, y, rx, ry=None, outline=True):
        """Hand-inked leafy foliage clump with scalloped lobes and right-side shadow, base at (x, y)."""
        rng, W0 = self.rng, self.width
        ry = ry or rx * 0.72
        cy = y - ry * 0.85
        # Scalloped multi-lobe silhouette (flat-grounded at bottom, billowy lobes on top)
        n_lobes = max(5, int(rx / 1.8) + 2)
        ts = np.linspace(0, TAU, n_lobes * 6, endpoint=False)
        ph = rng.uniform(0, TAU)
        scallop = 1.0 + 0.13 * np.abs(np.sin(0.5 * n_lobes * ts + ph)) + 0.05 * np.sin(3 * ts + ph)
        xs = x + rx * scallop * np.cos(ts)
        ys = cy + ry * scallop * np.sin(ts)
        ys = np.minimum(ys, y + ry * 0.12)
        blob = np.stack([xs, ys], 1)
        self.cv.erase(blob)

        # Leafy dabs: dense & darker along bottom/east shadow, airy on lit upper-left
        rows = max(2, int(ry * 2.1 / 2.6))
        for i in range(rows):
            v = (i + 0.5) / rows  # 0 top .. 1 bottom
            yy = cy - ry * 0.92 + v * ry * 1.85
            half = rx * math.sqrt(max(0.0, 1 - (2 * v - 1) ** 2)) * 0.92
            n = max(1, int(half * 2 / 3.4) + (1 if v > 0.55 else 0))
            for j in range(n):
                u = (j + rng.uniform(0.2, 0.8)) / n  # 0 left .. 1 right
                xx = x - half + u * 2 * half
                shadow = (u > 0.48) or (v > 0.62)
                if not shadow and rng.random() < 0.48:
                    continue  # leave sunlit breathing space on upper-left
                size = rng.uniform(1.3, 2.2) * (1.1 if shadow else 0.85)
                self.dab(xx, yy + rng.normal(0, 0.45), size, angle=rng.normal(0.15 if u > 0.5 else -0.1, 0.25),
                         w=W0 * (0.95 if (u > 0.55 and v > 0.45) else 0.72 if shadow else 0.55))

        if outline:
            # Crisp scalloped leafy rim (broken slightly on sunlit upper-left for hand-inked charm)
            ring = np.vstack([blob, blob[:1]])
            k = len(blob)
            i = 0
            while i < k:
                # bottom & east rim stay mostly continuous; upper-left can breathe gently
                t_mid = ts[i % k]
                is_lit = math.sin(t_mid) < -0.25 and math.cos(t_mid) < -0.1
                span = int(rng.integers(k // 5, k // 2))
                j = min(k, i + span)
                if not (is_lit and rng.random() < 0.28):
                    self.stroke(ring[i:j + 1], W0 * (0.88 if not is_lit else 0.72),
                                taper=(0.2, 0.2), wobble=0.25)
                i = j + (int(rng.integers(1, 3)) if is_lit else 0)
        # Subtle grounded base shadow stroke
        self.stroke([(x - rx * 0.72, y + 0.3), (x + rx * 0.82, y + 0.3)], W0 * 0.85,
                    taper=(0.25, 0.2), wobble=0.2)

    def stone(self, x, y, r, w=None):
        rng = self.rng
        pts = blob_pts(rng, x, y, r, n=rng.integers(5, 8), irregular=0.25, squash=0.7)
        self.cv.erase(pts)
        if rng.random() < 0.3:  # dark (shadowed) stone
            self.blot(pts, jitter=0.1)
            return
        self.closed(pts, (w or self.width) * 0.85, wobble=0.15, overshoot=0)
        shade = clip_halfplane(list(pts), (0.6, 0.8), x * 0.6 + y * 0.8 + r * 0.15)
        if len(shade) >= 3:
            self.blot(shade, jitter=0.05)

    def rubble(self, cx, cy, rx, ry, n=None, rmax=3.2):
        rng = self.rng
        n = n or max(3, int(rx * ry / 9))
        items = []
        for _ in range(n):
            a, rr = rng.uniform(0, TAU), math.sqrt(rng.uniform(0, 1))
            x, y = cx + math.cos(a) * rx * rr, cy + math.sin(a) * ry * rr
            items.append((y, x, rng.uniform(1.2, rmax) * (1.15 - 0.5 * rr)))
        for y, x, r in sorted(items):
            self.stone(x, y, r)

    def beam(self, a, b, thick=2.2):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L = np.hypot(*d)
        if L < 1:
            return
        n = np.array([-d[1], d[0]]) / L * thick / 2
        quad = [a + n, b + n, b - n, a - n]
        self.cv.erase(quad)
        w = self.width * 0.7
        self.line(a + n, b + n, w, sketch=0)
        self.line(a - n, b - n, w, sketch=0)
        self.line(a + n, a - n, w * 0.8, sketch=0, overshoot=0)
        self.line(b + n, b - n, w * 0.8, sketch=0, overshoot=0)

    def dead_tree(self, x, y, h, lean=0.0, depth=3, w=None):
        rng = self.rng
        w = w or self.width * 1.6

        def branch(x0, y0, ang, L, wd, lvl):
            x1, y1 = x0 + math.cos(ang) * L, y0 + math.sin(ang) * L
            mid = (x0 + (x1 - x0) * 0.5 + rng.normal(0, L * 0.06), y0 + (y1 - y0) * 0.5)
            self.stroke([(x0, y0), mid, (x1, y1)], wd, taper=(0.0, 0.8), smooth=1, wobble=0.3)
            if lvl <= 0:
                return
            for _ in range(rng.integers(2, 4)):
                fx = rng.uniform(0.55, 1.0)
                bx, by = x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fx
                branch(bx, by, ang + rng.choice([-1, 1]) * rng.uniform(0.3, 0.8),
                       L * rng.uniform(0.45, 0.7), wd * 0.6, lvl - 1)

        branch(x, y, -math.pi / 2 + lean, h, w, depth)
