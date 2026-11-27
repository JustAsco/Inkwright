"""Mountain-tribe and Viking-style settlements: hide tents, yurts, longhouses, drying racks, camps & villages.

Shares the 3/4 orthographic projection (`Proj`, lit from the west, shadowed on the east) and
hand-inked paper/line aesthetic of the rest of Inkwright, with rugged timber, lashed poles,
stretched hides, carved gable horns, stone roundhouses, sharpened-log palisades and alpine spruces.
"""
import math

import numpy as np

from .flora import spruce
from .houses import _overlaps
from .ink import TAU, chaikin, ellipse_pts
from .proj import sort_boxes
from .solids import box, cone, cylinder, hull


# --------------------------------------------------------------------------- rugged ground & timber props

def _draw_rock(pen, P, rng, x, y, r=4.5):
    """Compact grounded alpine boulder (never oval/floating — flat seated base with east hatch)."""
    W0 = pen.width
    cx, cy = P(x, y, 0.0)
    pts = [
        (cx - r, cy + r * 0.10),
        (cx - r * 0.72, cy - r * 0.62),
        (cx - r * 0.12, cy - r * 0.92),
        (cx + r * 0.60, cy - r * 0.72),
        (cx + r * 0.96, cy + r * 0.08),
        (cx + r * 0.35, cy + r * 0.32),
        (cx - r * 0.45, cy + r * 0.30),
    ]
    pts = [(px + rng.normal(0, r * 0.06), py + rng.normal(0, r * 0.05)) for px, py in pts]
    pen.cv.erase(pts)
    sh = [(cx, cy - r * 0.85), pts[3], pts[4], pts[5], (cx, cy + r * 0.25)]
    pen.hatch(sh, 78, 1.5, W0 * 0.48, coverage=(0.68, 1.0))
    pen.closed(pts, W0 * 0.72, wobble=0.14)


def _draw_smoke(pen, P, x, y, z, h=14.0):
    """Delicate curling wood-smoke wisp rising from a hearth or smoke hole."""
    W0 = pen.width
    sx, sy = P(x, y, z)
    pts = [
        (sx, sy),
        (sx + 2.6, sy - h * 0.32),
        (sx + 0.6, sy - h * 0.66),
        (sx + 4.2, sy - h),
    ]
    pen.stroke(pts, W0 * 0.42, taper=(0.35, 0.75), smooth=2, wobble=0.18)


def _draw_horn_finials(pen, P, x, y, z, dx, dy, sz=6.5, curved=True):
    """Crossed Nordic bargeboard / curved dragon-prow gable horns at a roof ridge peak."""
    W0 = pen.width
    p0 = P(x, y, z)
    # Two crossed timber gable arms jutting upward & slightly outward along the roof plane
    for sgn in (-1.0, 1.0):
        px, py = P(x + dx * sgn * sz * 0.62, y + dy * sgn * sz * 0.62, z + sz * 0.82)
        if curved:
            mid = ((p0[0] + px) * 0.5 + sgn * 1.4, (p0[1] + py) * 0.5 - 1.6)
            tip = (px + sgn * 2.2, py - 1.8)
            pen.stroke([p0, mid, tip], W0 * 0.88, taper=(0.05, 0.45), smooth=1, wobble=0.06)
        else:
            pen.stroke([p0, (px, py)], W0 * 0.86, taper=(0.05, 0.25), wobble=0.05)


def draw_drying_rack(pen, P, rng, X, Y, style=None, scale=1.0):
    """Tribal hide-stretching frame OR hanging meat/fish/pelt drying rack in true 3/4."""
    W0 = pen.width
    style = style or ("hide_frame" if rng.random() < 0.52 else "hanging")

    if style == "hide_frame":
        # Lashed vertical timber frame with a taut stretched animal pelt laced inside by tie-cords!
        W = rng.uniform(16, 21) * scale
        H = rng.uniform(14, 18) * scale
        x0, x1 = X - W * 0.5, X + W * 0.5
        # Supporting angled rear diagonal brace legs
        pen.stroke([P(x0 + 1.5, Y + 5.0 * scale, 0.0), P(x0 + 1.0, Y, H * 0.72)], W0 * 0.62, taper=(0.05, 0.1), wobble=0.06)
        pen.stroke([P(x1 - 1.5, Y + 5.0 * scale, 0.0), P(x1 - 1.0, Y, H * 0.72)], W0 * 0.62, taper=(0.05, 0.1), wobble=0.06)
        # Main lashed wooden frame (2 vertical posts + top & bottom crossbars)
        pen.beam(P(x0, Y, 0.0), P(x0, Y, H + 2.2 * scale), 1.45 * scale)
        pen.beam(P(x1, Y, 0.0), P(x1, Y, H + 2.2 * scale), 1.45 * scale)
        pen.beam(P(x0 - 2.0 * scale, Y, H), P(x1 + 2.0 * scale, Y, H), 1.35 * scale)
        pen.beam(P(x0 - 1.6 * scale, Y, 2.5 * scale), P(x1 + 1.6 * scale, Y, 2.5 * scale), 1.20 * scale)

        # Irregular taut stretched animal pelt polygon suspended inside frame
        hx0, hx1 = x0 + 2.6 * scale, x1 - 2.6 * scale
        hz0, hz1 = 4.2 * scale, H - 1.8 * scale
        hxm, hzm = (hx0 + hx1) * 0.5, (hz0 + hz1) * 0.5
        pelt3 = [
            (hx0 - 0.6 * scale, Y, hz0), (hxm, Y, hz0 + 1.1 * scale), (hx1 + 0.6 * scale, Y, hz0),
            (hx1 - 0.8 * scale, Y, hzm), (hx1 + 0.7 * scale, Y, hz1),
            (hxm, Y, hz1 - 0.8 * scale), (hx0 - 0.7 * scale, Y, hz1),
            (hx0 + 0.8 * scale, Y, hzm),
        ]
        pelt2 = [P(*p) for p in pelt3]
        # Radial tie-cords stretching pelt corners & edges to the frame posts
        ties = [
            (pelt3[0], (x0, Y, 2.8 * scale)), (pelt3[2], (x1, Y, 2.8 * scale)),
            (pelt3[4], (x1, Y, H)), (pelt3[6], (x0, Y, H)),
            (pelt3[3], (x1, Y, hzm)), (pelt3[7], (x0, Y, hzm)),
            (pelt3[5], (hxm, Y, H)), (pelt3[1], (hxm, Y, 2.6 * scale)),
        ]
        for a3, b3 in ties:
            pen.stroke([P(*a3), P(*b3)], W0 * 0.36, taper=(0.1, 0.1), wobble=0.05)
        pen.cv.erase(pelt2)
        # Subtle hide crease/fur shading on east half of stretched skin
        sh = [P(hxm, Y, hz0 + 0.8 * scale), pelt2[2], pelt2[3], pelt2[4], P(hxm, Y, hz1 - 0.8 * scale)]
        pen.hatch(sh, 72, 1.9, W0 * 0.42, coverage=(0.45, 0.88))
        pen.closed(pelt2, W0 * 0.74, wobble=0.14)
        # Lashings at the 4 wooden frame joints
        for lx, lz in ((x0, H), (x1, H), (x0, 2.5 * scale), (x1, 2.5 * scale)):
            px, py = P(lx, Y, lz)
            pen.stroke([(px - 1.5, py - 1.4), (px + 1.5, py + 1.4)], W0 * 0.55, taper=(0, 0), wobble=0)
            pen.stroke([(px - 1.5, py + 1.4), (px + 1.5, py - 1.4)], W0 * 0.55, taper=(0, 0), wobble=0)
    else:
        # Twin A-frame tripod poles supporting a long horizontal crossbar draped with hanging pelts / smoked fish!
        W = rng.uniform(19, 25) * scale
        H = rng.uniform(12, 16) * scale
        x0, x1 = X - W * 0.5, X + W * 0.5
        # Rear A-legs (back-to-front)
        for xp in (x0, x1):
            pen.beam(P(xp, Y + 4.5 * scale, 0.0), P(xp, Y, H + 2.4 * scale), 1.25 * scale)
        # Long horizontal crossbar pole
        pen.beam(P(x0 - 3.2 * scale, Y, H), P(x1 + 3.2 * scale, Y, H), 1.45 * scale)
        # Hanging pelts / smoked fish / dried meat slabs along crossbar
        n_items = int(rng.integers(3, 5))
        for k, xp in enumerate(np.linspace(x0 + 3.2 * scale, x1 - 3.2 * scale, n_items)):
            if k % 2 == 0:
                # Draped fur pelt hanging folded over the bar
                pw = rng.uniform(2.6, 3.8) * scale
                ph = rng.uniform(5.8, 8.6) * scale
                poly = [
                    P(xp - pw * 0.5, Y, H), P(xp + pw * 0.5, Y, H),
                    P(xp + pw * 0.62, Y - 0.6 * scale, H - ph * 0.78),
                    P(xp + pw * 0.20, Y - 0.6 * scale, H - ph),
                    P(xp - pw * 0.42, Y - 0.6 * scale, H - ph * 0.88),
                ]
                pen.cv.erase(poly)
                pen.hatch(poly, 82, 1.7, W0 * 0.44, coverage=(0.55, 0.92))
                pen.closed(poly, W0 * 0.64, wobble=0.10)
            else:
                # Hanging smoked fish / meat haunch on a short cord
                hl = rng.uniform(5.2, 7.6) * scale
                pen.stroke([P(xp, Y, H), P(xp, Y, H - 2.2 * scale)], W0 * 0.36, taper=(0, 0), wobble=0)
                body = [
                    P(xp, Y, H - 2.0 * scale),
                    P(xp + 1.5 * scale, Y, H - 2.0 * scale - hl * 0.45),
                    P(xp, Y, H - 2.0 * scale - hl),
                    P(xp - 1.5 * scale, Y, H - 2.0 * scale - hl * 0.45),
                ]
                pen.cv.erase(body)
                pen.closed(body, W0 * 0.62, wobble=0.08)
        # Front A-legs crossing over crossbar
        for xp in (x0, x1):
            pen.beam(P(xp, Y - 4.5 * scale, 0.0), P(xp, Y, H + 2.4 * scale), 1.35 * scale)


def draw_bonfire(pen, P, rng, X, Y, scale=1.0, cauldron=None):
    """Clan stone-ring firepit with crossed logs, leaping flames, smoke & optional roasting spit/benches."""
    W0 = pen.width
    R = rng.uniform(6.5, 8.2) * scale
    if cauldron is None:
        cauldron = rng.random() < 0.45

    # Ring of rough hearth stones sorted back-to-front
    n_st = 9
    stones = []
    for k in range(n_st):
        ang = k * TAU / n_st + rng.uniform(-0.15, 0.15)
        sx, sy = X + R * math.cos(ang), Y + R * math.sin(ang)
        stones.append((P.depth(sx, sy), sx, sy))
    # Split stones into back half and front half around the fire!
    back_st = [s for s in stones if s[0] >= P.depth(X, Y)]
    front_st = [s for s in stones if s[0] < P.depth(X, Y)]
    for _, sx, sy in sorted(back_st, key=lambda it: -it[0]):
        _draw_rock(pen, P, rng, sx, sy, r=rng.uniform(1.8, 2.6) * scale)

    # Dark charred ash bed & crossed timber fire logs
    cx, cy = P(X, Y, 0.2)
    pen.blot(ellipse_pts(cx, cy, R * 0.72, R * 0.72 * P.se, 14), jitter=0.08)
    for ang in (0.35, 1.35, 2.45):
        lx, ly = math.cos(ang) * R * 0.68, math.sin(ang) * R * 0.68
        pen.beam(P(X - lx, Y - ly, 0.6), P(X + lx, Y + ly, 2.2 * scale), 1.45 * scale)

    # Optional wooden tripod + elevated iron cauldron (flames lick cleanly around its base!)
    if cauldron:
        th = 16.5 * scale
        for ang in (0.6, 2.4, 4.7):
            pen.stroke([P(X + R * 1.05 * math.cos(ang), Y + R * 1.05 * math.sin(ang), 0.0), P(X, Y, th)],
                       W0 * 0.66, taper=(0.0, 0.15), wobble=0.06)
        pen.stroke([P(X, Y, th), P(X, Y, 9.0 * scale)], W0 * 0.36, taper=(0, 0), wobble=0)
        pc = P(X, Y, 7.2 * scale)
        pen.blot(ellipse_pts(pc[0], pc[1], 2.5 * scale, 2.0 * scale, 12), jitter=0.05)

    # Stylized leaping bright hand-inked flames (drawn crisp and tall so the fire reads immediately!)
    fh = rng.uniform(10.5, 14.5) * scale if not cauldron else rng.uniform(8.0, 10.5) * scale
    fl_pts = [
        (cx - 4.4 * scale, cy + 0.5),
        (cx - 5.2 * scale, cy - fh * 0.48),
        (cx - 2.4 * scale, cy - fh * 0.32),
        (cx - 0.4 * scale, cy - fh),
        (cx + 1.9 * scale, cy - fh * 0.44),
        (cx + 4.4 * scale, cy - fh * 0.66),
        (cx + 4.6 * scale, cy + 0.5),
    ]
    pen.cv.erase(fl_pts)
    pen.closed(fl_pts, W0 * 0.78, wobble=0.12)
    in_fl = [(cx - 1.8 * scale, cy - 0.4), (cx - 0.2 * scale, cy - fh * 0.56), (cx + 1.8 * scale, cy - 0.4)]
    pen.stroke(in_fl, W0 * 0.46, taper=(0.2, 0.2), wobble=0.08)

    for _, sx, sy in sorted(front_st, key=lambda it: -it[0]):
        _draw_rock(pen, P, rng, sx, sy, r=rng.uniform(1.8, 2.6) * scale)
    _draw_smoke(pen, P, X, Y, max(fh * 1.15, (16.5 if cauldron else 0.0) * scale), h=14.0 * scale)


def draw_totem(pen, P, rng, X, Y, scale=1.0):
    """Carved mountain-clan totem pole OR horned bone-and-banner war standard in a stone cairn."""
    W0 = pen.width
    # Stacked cairn boulders anchoring the base
    for ox, oy, rr in ((-3.0, 1.4, 2.4), (2.8, 1.2, 2.2), (0.2, 2.4, 2.2)):
        _draw_rock(pen, P, rng, X + ox * scale, Y + oy * scale, r=rr * scale)

    H = rng.uniform(24, 34) * scale
    r = rng.uniform(2.0, 2.6) * scale
    if rng.random() < 0.55:
        # Carved wooden spirit totem pole with angular carved wings & mask faces
        cylinder(pen, P, X, Y, r, 0.0, H, top=True, w=W0 * 0.80)
        # Carved horizontal spirit-mask band notches along front face
        for frac in (0.30, 0.56, 0.80):
            hz = H * frac
            p_l, p_r = P(X - r * 0.9, Y - r * 0.4, hz), P(X + r * 0.9, Y - r * 0.4, hz)
            pen.stroke([p_l, p_r], W0 * 0.58, taper=(0.1, 0.1), wobble=0.06)
            # Carved dark eyes above bar
            for ex in (-r * 0.42, r * 0.42):
                pe = P(X + ex, Y - r * 0.6, hz + 1.8 * scale)
                pen.blot(ellipse_pts(pe[0], pe[1], 0.75 * scale, 0.65 * scale, 8), jitter=0.02)
        # Sweeping carved wooden thunderbird/eagle wings jutting out near top
        wz = H * 0.78
        for sgn in (-1.0, 1.0):
            wing = [
                P(X + sgn * r * 0.8, Y, wz + 2.4 * scale),
                P(X + sgn * 10.5 * scale, Y, wz + 5.2 * scale),
                P(X + sgn * 8.8 * scale, Y, wz - 0.8 * scale),
                P(X + sgn * r * 0.8, Y, wz - 2.0 * scale),
            ]
            pen.cv.erase(wing)
            if sgn > 0:
                pen.hatch(wing, 78, 1.6, W0 * 0.46, coverage=(0.65, 1.0))
            pen.closed(wing, W0 * 0.72, wobble=0.10)
    else:
        # Rugged highland clan banner pole with curved ram/elk horns & hanging fur/bone talismans
        pen.beam(P(X, Y, 0.0), P(X, Y, H), 2.0 * scale)
        # Cross-spar + ragged notched war banner hanging below
        cz = H * 0.78
        cw = 7.2 * scale
        pen.beam(P(X - cw, Y, cz), P(X + cw, Y, cz), 1.35 * scale)
        banner = [
            P(X - cw * 0.62, Y, cz), P(X + cw * 0.62, Y, cz),
            P(X + cw * 0.54, Y - 0.5, cz - 11.5 * scale),
            P(X + cw * 0.12, Y - 0.5, cz - 8.5 * scale),
            P(X - cw * 0.48, Y - 0.5, cz - 12.2 * scale),
        ]
        pen.cv.erase(banner)
        pen.hatch([banner[0], banner[1], banner[2], P(X, Y - 0.5, cz - 9.0 * scale)], 82, 2.0, W0 * 0.40, coverage=(0.4, 0.8))
        pen.closed(banner, W0 * 0.70, wobble=0.12)
        # Bold curved mountain-beast horns & skull crowning the pole top!
        for sgn in (-1.0, 1.0):
            h_pts = [
                P(X + sgn * 0.8 * scale, Y, H - 1.5 * scale),
                P(X + sgn * 5.2 * scale, Y, H + 0.5 * scale),
                P(X + sgn * 7.8 * scale, Y, H + 4.8 * scale),
                P(X + sgn * 6.0 * scale, Y, H + 7.2 * scale),
            ]
            pen.stroke(h_pts, W0 * 0.88, taper=(0.05, 0.75), smooth=2, wobble=0.06)
        sk = P(X, Y - 0.5, H - 0.8 * scale)
        sk_p = ellipse_pts(sk[0], sk[1], 2.4 * scale, 2.8 * scale, 10)
        pen.cv.erase(sk_p)
        pen.closed(sk_p, W0 * 0.68, wobble=0.08)

    # Front cairn stones strictly in front of pole foot
    for ox, oy, rr in ((-2.4, -2.0, 2.2), (1.8, -2.2, 2.3)):
        _draw_rock(pen, P, rng, X + ox * scale, Y + oy * scale, r=rr * scale)


def draw_palisade_seg(pen, P, rng, a, b, H=16.5, log_r=1.85, show_inner=False):
    """Sharpened vertical timber log palisade wall running between world points `a` and `b`."""
    W0 = pen.width
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.hypot(*(b - a)))
    if L < 3.0:
        return
    u = (b - a) / L
    n_logs = max(2, int(round(L / (log_r * 1.92))))
    # Sort individual sharpened logs back-to-front by P.depth so every run occludes cleanly!
    logs = []
    for i in range(n_logs):
        f = (i + 0.5) / n_logs
        g = a + u * (f * L)
        lh = H * rng.uniform(0.90, 1.12)
        logs.append((P.depth(g[0], g[1]), g[0], g[1], lh))

    # Determine whether this wall face is predominantly east-shaded
    n_vec = np.array([-u[1], u[0]])
    if float(n_vec @ (-(a + b) * 0.5)) < 0:
        n_vec = -n_vec
    face_n = n_vec if show_inner else -n_vec
    shaded = face_n[0] > 0.25

    for _, lx, ly, lh in sorted(logs, key=lambda it: -it[0]):
        b0 = P(lx - u[0] * log_r, ly - u[1] * log_r, 0.0)
        b1 = P(lx + u[0] * log_r, ly + u[1] * log_r, 0.0)
        t0 = P(lx - u[0] * log_r, ly - u[1] * log_r, lh - 3.6)
        t1 = P(lx + u[0] * log_r, ly + u[1] * log_r, lh - 3.6)
        tip = P(lx, ly, lh)
        poly = [b0, b1, t1, tip, t0]
        pen.cv.erase(poly)
        if shaded:
            pen.hatch(poly, 82, 1.55, W0 * 0.56, coverage=(0.72, 1.0))
        else:
            # Subtle east-side curve shadow on each round log
            sh = [P(lx, ly, 0.0), b1, t1, tip]
            pen.hatch(sh, 84, 1.7, W0 * 0.36, coverage=(0.45, 0.85))
        pen.closed(poly, W0 * 0.64, wobble=0.10)

    # Two horizontal lashed cross-beams binding the palisade logs together
    for hz in (H * 0.30, H * 0.68):
        p0 = P(a[0], a[1], hz)
        p1 = P(b[0], b[1], hz)
        pen.beam(p0, p1, 1.25)


def draw_tribal_watchtower(pen, P, rng, X, Y, scale=1.0):
    """Rugged timber tribal watch-platform tower perched on 4 braced log stilts with hide/bark roof."""
    W0 = pen.width
    s = rng.uniform(7.2, 8.8) * scale
    H_deck = rng.uniform(19, 25) * scale
    H_roof = H_deck + rng.uniform(11, 14) * scale
    x0, x1, y0, y1 = X - s, X + s, Y - s, Y + s

    # 1. Rear vertical log stilts + rear X-bracing
    pen.beam(P(x0, y1, 0.0), P(x0, y1, H_deck), 1.6 * scale)
    pen.beam(P(x1, y1, 0.0), P(x1, y1, H_deck), 1.6 * scale)
    pen.stroke([P(x0, y1, 2.0), P(x1, y1, H_deck - 2.0)], W0 * 0.62, taper=(0, 0), wobble=0.06)
    pen.stroke([P(x1, y1, 2.0), P(x0, y1, H_deck - 2.0)], W0 * 0.62, taper=(0, 0), wobble=0.06)

    # 2. Front stilts + south/east X-bracing + leaning wooden ladder
    pen.beam(P(x0, y0, 0.0), P(x0, y0, H_deck), 1.7 * scale)
    pen.beam(P(x1, y0, 0.0), P(x1, y0, H_deck), 1.7 * scale)
    pen.stroke([P(x0, y0, 2.0), P(x1, y0, H_deck - 2.0)], W0 * 0.68, taper=(0, 0), wobble=0.06)
    pen.stroke([P(x1, y0, 2.0), P(x0, y0, H_deck - 2.0)], W0 * 0.68, taper=(0, 0), wobble=0.06)
    pen.stroke([P(x1, y0, 2.0), P(x1, y1, H_deck - 2.0)], W0 * 0.58, taper=(0, 0), wobble=0.06)
    # Leaning wooden ladder on south face
    lx = X - s * 0.25
    pen.stroke([P(lx - 1.4 * scale, y0 - 4.2 * scale, 0.0), P(lx - 1.4 * scale, y0, H_deck)], W0 * 0.62, taper=(0, 0), wobble=0.04)
    pen.stroke([P(lx + 1.4 * scale, y0 - 4.2 * scale, 0.0), P(lx + 1.4 * scale, y0, H_deck)], W0 * 0.62, taper=(0, 0), wobble=0.04)
    for f in np.linspace(0.12, 0.88, 6):
        pen.stroke([P(lx - 1.4 * scale, y0 - 4.2 * scale * (1 - f), H_deck * f),
                    P(lx + 1.4 * scale, y0 - 4.2 * scale * (1 - f), H_deck * f)], W0 * 0.44, taper=(0, 0), wobble=0)

    # 3. Overhanging log parapet platform box
    sp = s * 1.18
    box(pen, P, X - sp, Y - sp, X + sp, Y + sp, H_deck - 1.5 * scale, H_deck + 5.2 * scale, w=W0 * 0.78)
    # Corner roof support posts
    for px, py in ((X - sp + 1.0, Y + sp - 1.0), (X + sp - 1.0, Y + sp - 1.0),
                   (X - sp + 1.0, Y - sp + 1.0), (X + sp - 1.0, Y - sp + 1.0)):
        pen.beam(P(px, py, H_deck + 4.8 * scale), P(px, py, H_roof), 1.25 * scale)
    # Sloped rugged hide/bark lean-to OR pyramidal timber roof
    ov = 2.6 * scale
    if rng.random() < 0.55:
        # Sloped lean-to timber roof
        r_poly = [P(X - sp - ov, Y - sp - ov, H_roof - 1.5 * scale),
                  P(X + sp + ov, Y - sp - ov, H_roof - 1.5 * scale),
                  P(X + sp + ov, Y + sp + ov, H_roof + 3.2 * scale),
                  P(X - sp - ov, Y + sp + ov, H_roof + 3.2 * scale)]
        pen.cv.erase(r_poly)
        pen.hatch(r_poly, 68, 2.2, W0 * 0.48, coverage=(0.65, 1.0))
        pen.closed(r_poly, W0 * 0.84, wobble=0.14)
        flag_pt = P(X + sp * 0.85, Y + sp * 0.85, H_roof + 3.2 * scale)
    else:
        apex_h = H_roof + sp * 1.25
        cone(pen, P, X, Y, sp * 1.42, H_roof - 1.0 * scale, apex_h, broken=0.0)
        flag_pt = P(X, Y, apex_h)
    # Tattered clan pennant fluttering proudly ABOVE the roof ridge/apex!
    pen.stroke([flag_pt, (flag_pt[0], flag_pt[1] - 9.5 * scale)], W0 * 0.62, taper=(0, 0.2), wobble=0)
    pen.stroke([(flag_pt[0], flag_pt[1] - 9.0 * scale), (flag_pt[0] + 7.2 * scale, flag_pt[1] - 7.4 * scale),
                (flag_pt[0], flag_pt[1] - 5.5 * scale)], W0 * 0.56, taper=(0.1, 0.1), wobble=0.08)


# --------------------------------------------------------------------------- 5 distinct mountain-tribe dwellings

def draw_tent(pen, P, rng, X, Y, r=14.0, H=24.0, smoke=True):
    """Conical mountain-tribe hide teepee / clan pavilion tent with crossed apex ridge poles & open flap."""
    W0 = pen.width
    a = P.az
    # 1. Crossed wooden teepee poles jutting out the top smoke hole
    pole_top = H + r * 0.44
    for ang in np.linspace(0.25, TAU + 0.25, 5, endpoint=False):
        dx, dy = math.cos(ang), math.sin(ang)
        pen.stroke([P(X - dx * r * 0.26, Y - dy * r * 0.26, H * 0.74),
                    P(X + dx * r * 0.30, Y + dy * r * 0.30, pole_top + rng.uniform(-1.2, 1.5))],
                   W0 * 0.72, taper=(0.0, 0.28), wobble=0.05)

    # 2. Conical stretched-hide body with organic slight skirt sag
    ring_t = np.linspace(0.0, TAU, 36, endpoint=False)
    base = [P(X + r * (1.0 + 0.03 * math.sin(5 * t)) * math.cos(t),
              Y + r * (1.0 + 0.03 * math.sin(5 * t)) * math.sin(t), 0.0) for t in ring_t]
    apex = P(X, Y, H)
    shape = hull(base + [apex])
    pen.cv.erase(shape)

    # East-side woodcut shadow hatching on the shadowed flank
    ts_sh = np.linspace(a + TAU - 1.15, a + TAU, 10)
    sh = hull([P(X + r * math.cos(t), Y + r * math.sin(t), 0.0) for t in ts_sh] + [apex])
    pen.hatch(sh, 72, 1.65, W0 * 0.56, coverage=(0.72, 1.0))

    # Stitched hide seam lines radiating from apex down the tent flanks
    for t in np.linspace(a + math.pi + 0.35, a + TAU - 0.35, 5):
        b = P(X + r * math.cos(t), Y + r * math.sin(t), 0.0)
        pen.stroke([b, (b[0] + (apex[0] - b[0]) * 0.88, b[1] + (apex[1] - b[1]) * 0.88)],
                   W0 * 0.40, taper=(0.15, 0.55), wobble=0.12)

    # Painted tribal decorative zigzag trim band around lower third of the tent
    band_h = H * 0.22
    band_r = r * (1.0 - 0.22)
    b_ts = np.linspace(a + math.pi + 0.16, a + TAU - 0.16, 15)
    zig = [P(X + band_r * math.cos(t), Y + band_r * math.sin(t), band_h + (1.4 if k % 2 == 0 else -1.2))
           for k, t in enumerate(b_ts)]
    pen.stroke(zig, W0 * 0.46, taper=(0.15, 0.15), wobble=0.05)

    # Triangular dark entrance opening + peeled-back hide flaps on sunlit south face
    t_door = a + 1.5 * math.pi - 0.12
    dh = H * 0.42
    d_l = P(X + r * math.cos(t_door - 0.28), Y + r * math.sin(t_door - 0.28), 0.0)
    d_r = P(X + r * math.cos(t_door + 0.28), Y + r * math.sin(t_door + 0.28), 0.0)
    d_top = P(X + r * 0.58 * math.cos(t_door), Y + r * 0.58 * math.sin(t_door), dh)
    pen.blot([d_l, d_r, d_top], jitter=0.06)
    # Folded-back leather side flaps
    fl_l = P(X + r * 0.88 * math.cos(t_door - 0.44), Y + r * 0.88 * math.sin(t_door - 0.44), dh * 0.36)
    fl_r = P(X + r * 0.88 * math.cos(t_door + 0.44), Y + r * 0.88 * math.sin(t_door + 0.44), dh * 0.36)
    for flap in ([d_l, fl_l, d_top], [d_r, fl_r, d_top]):
        pen.cv.erase(flap)
        pen.closed(flap, W0 * 0.62, wobble=0.08)

    # Outer outline + grounded bottom skirt + pegged guy-ropes
    pen.closed(shape, W0 * 0.92, wobble=0.14)
    pen.stroke([P(X + r * math.cos(t), Y + r * math.sin(t), 0.0) for t in np.linspace(a + math.pi, a + TAU, 24)],
               W0 * 1.12, taper=(0.05, 0.05), wobble=0.16)
    for t_g in (a + math.pi + 0.10, a + TAU - 0.10):
        pen.stroke([P(X + r * 0.52 * math.cos(t_g), Y + r * 0.52 * math.sin(t_g), H * 0.48),
                    P(X + r * 1.24 * math.cos(t_g), Y + r * 1.24 * math.sin(t_g), 0.0)],
                   W0 * 0.40, taper=(0.05, 0.20), wobble=0.06)
    if smoke and rng.random() < 0.45:
        _draw_smoke(pen, P, X, Y, H + 2.0, h=11.0)


def draw_yurt(pen, P, rng, X, Y, r=15.0, H=9.5, smoke=True):
    """Mountain-nomad felt yurt / ger with lashed lattice wall, domed roof, crown smoke ring & rugs."""
    W0 = pen.width
    a = P.az
    C = cylinder(pen, P, X, Y, r, 0.0, H, top=False, w=W0 * 0.88)
    # Horizontal braided rope bands wrapping the cylindrical felt walls
    for frac in (0.34, 0.72):
        pen.stroke([C(t, H * frac) for t in np.linspace(a + math.pi, a + TAU, 24)],
                   W0 * 0.50, taper=(0.05, 0.05), wobble=0.08)
    # Low arched wooden frame doorway on sunlit south face
    t_d = a + 1.5 * math.pi - 0.12
    dt = 0.24
    dq = [C(t_d - dt, 0.0), C(t_d + dt, 0.0), C(t_d + dt, H * 0.84), C(t_d - dt, H * 0.84)]
    pen.blot(dq, jitter=0.05)
    pen.closed(dq, W0 * 0.72, wobble=0.06)
    # Gently pitched conical felt roof + central wooden smoke-hole crown ring (toono)
    apex_h = H + r * rng.uniform(0.62, 0.82)
    cone(pen, P, X, Y, r * 1.07, H, apex_h, broken=0.0)
    # Raised circular smoke-ring crown perched on apex
    cylinder(pen, P, X, Y, r * 0.24, apex_h - 2.2, apex_h + 1.4, top=True, w=W0 * 0.68)
    cx, cy = P(X, Y, apex_h + 1.4)
    pen.blot(ellipse_pts(cx, cy, r * 0.15, r * 0.15 * P.se, 10), jitter=0.04)
    if smoke and rng.random() < 0.55:
        _draw_smoke(pen, P, X, Y, apex_h + 2.0, h=12.0)


def draw_aframe_hut(pen, P, rng, X0, Y0, W=24.0, D=28.0, H_base=4.2, R=19.5, ridge_x=False, smoke=True):
    """Steep alpine timber & turf A-frame mountain lodge with crossed horned bargeboards & recessed porch."""
    W0 = pen.width
    X1, Y1 = X0 + W, Y0 + D
    cx, cy = (X0 + X1) * 0.5, (Y0 + Y1) * 0.5

    # Low stone-and-log plinth + steep triangular gable front/east wall + sweeping roof slopes
    if not ridge_x:
        # Gable faces south (iconic alpine A-frame silhouette!)
        south = [P(X0, Y0, 0), P(X1, Y0, 0), P(X1, Y0, H_base), P(cx, Y0, H_base + R), P(X0, Y0, H_base)]
        pen.cv.erase(south)
        # Horizontal log planks across recessed triangular timber gable
        for f in np.linspace(0.12, 0.82, 5):
            hz = H_base + R * f
            hw = W * 0.5 * (1.0 - f)
            pen.stroke([P(cx - hw, Y0, hz), P(cx + hw, Y0, hz)], W0 * 0.42, taper=(0.08, 0.08), wobble=0.08)
        # Dark wooden doorway + heavy timber lintel
        dw, dh = min(5.6, W * 0.28), min(9.5, H_base + R * 0.42)
        pen.blot([P(cx - dw * 0.5, Y0, 0.0), P(cx - dw * 0.5, Y0, dh * 0.78),
                  P(cx, Y0, dh), P(cx + dw * 0.5, Y0, dh * 0.78), P(cx + dw * 0.5, Y0, 0.0)], jitter=0.05)
        pen.closed(south, W0 * 0.86, wobble=0.14)

        # Steep overhanging east roof slope (shadowed!)
        ov = 2.0
        eh = max(0.8, H_base - 1.2)
        east_roof = [P(X1 + ov, Y0 - ov, eh), P(X1 + ov, Y1 + ov, eh),
                     P(cx, Y1 + ov, H_base + R), P(cx, Y0 - ov, H_base + R)]
        pen.cv.erase(east_roof)
        # Rugged timber shakes / turf thatch lines + heavy shadow hatch
        for u in np.linspace(0.05, 0.95, max(8, int(D / 2.2))):
            p_b = P(X1 + ov, Y0 - ov + (D + 2 * ov) * u, eh)
            p_t = P(cx, Y0 - ov + (D + 2 * ov) * u, H_base + R)
            pen.stroke([p_b, p_t], W0 * 0.58, taper=(0.05, 0.45), wobble=0.12)
        pen.hatch(east_roof, 76, 1.9, W0 * 0.48, coverage=(0.68, 1.0))
        pen.closed(east_roof, W0 * 0.90, wobble=0.16)
        pen.stroke([east_roof[0], east_roof[1]], W0 * 1.22, taper=(0.05, 0.05), wobble=0.14)

        # Bold heavy front gable bargeboards + crossed Viking/alpine horned ridge tips!
        pen.beam(P(X0 - ov, Y0 - ov, eh), P(cx, Y0 - ov, H_base + R), 1.5)
        pen.beam(P(X1 + ov, Y0 - ov, eh), P(cx, Y0 - ov, H_base + R), 1.5)
        _draw_horn_finials(pen, P, cx, Y0 - ov, H_base + R, 1.0, 0.0, sz=5.5, curved=True)
        _draw_horn_finials(pen, P, cx, Y1 + ov, H_base + R, 1.0, 0.0, sz=4.8, curved=True)
        if smoke and rng.random() < 0.45:
            _draw_smoke(pen, P, cx, cy, H_base + R + 1.0, h=11.0)
    else:
        # Side-facing alpine A-frame (ridge runs E-W!)
        east = [P(X1, Y0, 0), P(X1, Y1, 0), P(X1, Y1, H_base), P(X1, cy, H_base + R), P(X1, Y0, H_base)]
        pen.cv.erase(east)
        pen.hatch(east, 82, 1.65, W0 * 0.68, coverage=(0.76, 1.0))
        pen.closed(east, W0 * 0.86, wobble=0.14)
        ov = 2.0
        eh = max(0.8, H_base - 1.2)
        south_roof = [P(X0 - ov, Y0 - ov, eh), P(X1 + ov, Y0 - ov, eh),
                      P(X1 + ov, cy, H_base + R), P(X0 - ov, cy, H_base + R)]
        pen.cv.erase(south_roof)
        for u in np.linspace(0.04, 0.96, max(10, int(W / 2.3))):
            p_b = P(X0 - ov + (W + 2 * ov) * u, Y0 - ov, eh)
            p_t = P(X0 - ov + (W + 2 * ov) * u, cy, H_base + R)
            pen.stroke([p_b, p_t], W0 * 0.46, taper=(0.05, 0.55), wobble=0.14)
        for v in (0.35, 0.68):
            pen.stroke([P(X0 - ov, Y0 - ov + (cy - Y0 + ov) * v, eh + (H_base + R - eh) * v),
                        P(X1 + ov, Y0 - ov + (cy - Y0 + ov) * v, eh + (H_base + R - eh) * v)],
                       W0 * 0.34, taper=(0.15, 0.15), wobble=0.22)
        pen.closed(south_roof, W0 * 0.88, wobble=0.16)
        pen.stroke([south_roof[0], south_roof[1]], W0 * 1.22, taper=(0.05, 0.05), wobble=0.14)
        # Recessed projecting timber doorway hood cutting into south slope
        dw = min(6.2, W * 0.22)
        pen.blot([P(cx - dw * 0.5, Y0 - ov * 0.5, 0.0), P(cx + dw * 0.5, Y0 - ov * 0.5, 0.0),
                  P(cx + dw * 0.5, Y0 - ov * 0.5, 7.2), P(cx, Y0 - ov * 0.5, 9.6),
                  P(cx - dw * 0.5, Y0 - ov * 0.5, 7.2)], jitter=0.05)
        _draw_horn_finials(pen, P, X0 - ov, cy, H_base + R, 0.0, 1.0, sz=5.2, curved=True)
        _draw_horn_finials(pen, P, X1 + ov, cy, H_base + R, 0.0, 1.0, sz=5.2, curved=True)
        if smoke and rng.random() < 0.45:
            _draw_smoke(pen, P, cx, cy, H_base + R + 1.0, h=11.0)


def draw_viking_longhouse(pen, P, rng, X0, Y0, W=54.0, D=22.0, H=11.5, R=13.5, ridge_x=True, grand=False):
    """Iconic Viking / highland clan longhouse or chieftain mead-hall with notched logs, porch & dragon-prow gables."""
    W0 = pen.width
    X1, Y1 = X0 + W, Y0 + D
    cx, cy = (X0 + X1) * 0.5, (Y0 + Y1) * 0.5

    # 1. Walls with horizontal log courses, low stone foundation & notched log corner ends
    if ridge_x:
        south = [P(X0, Y0, 0), P(X1, Y0, 0), P(X1, Y0, H), P(X0, Y0, H)]
        east = [P(X1, Y0, 0), P(X1, Y1, 0), P(X1, Y1, H), P(X1, cy, H + R), P(X1, Y0, H)]
        pen.cv.erase(south)
        pen.cv.erase(east)
        # Horizontal log courses on sunlit south wall
        n_logs = max(4, int(H / 2.4))
        for k in range(1, n_logs):
            hz = H * k / n_logs
            pen.stroke([P(X0 - 1.2, Y0, hz), P(X1 + 1.2, Y0, hz)], W0 * 0.44, taper=(0.05, 0.05), wobble=0.10)
        # Heavy carved stave pillars flanking the south wall
        for frac in (0.04, 0.34, 0.66, 0.96) if grand else (0.04, 0.96):
            px = X0 + W * frac
            pen.stroke([P(px, Y0, 0.0), P(px, Y0, H)], W0 * 0.72, taper=(0, 0), wobble=0.05)
        pen.hatch(east, 82, 1.65, W0 * 0.68, coverage=(0.76, 1.0))
        pen.closed(south, W0 * 0.88, wobble=0.14)
        pen.closed(east, W0 * 0.88, wobble=0.14)

        # Sweeping steep timber/shake roof with ridge cap & central smoke-louver!
        ov = 2.0
        eh = H - ov * R / (D * 0.5)
        s_roof = [P(X0 - ov, Y0 - ov, eh), P(X1 + ov, Y0 - ov, eh),
                  P(X1 + ov, cy, H + R), P(X0 - ov, cy, H + R)]
        pen.cv.erase(s_roof)
        sp = max(10, int(W / 2.4))
        for u in np.linspace(0.03, 0.97, sp):
            pen.stroke([P(X0 - ov + (W + 2 * ov) * u, Y0 - ov, eh),
                        P(X0 - ov + (W + 2 * ov) * u, cy, H + R)],
                       W0 * 0.46, taper=(0.05, 0.55), wobble=0.14)
        for v in (0.34, 0.68):
            pen.stroke([P(X0 - ov, Y0 - ov + (cy - Y0 + ov) * v, eh + (H + R - eh) * v),
                        P(X1 + ov, Y0 - ov + (cy - Y0 + ov) * v, eh + (H + R - eh) * v)],
                       W0 * 0.36, taper=(0.12, 0.12), wobble=0.20)
        pen.closed(s_roof, W0 * 0.90, wobble=0.16)
        pen.stroke([s_roof[0], s_roof[1]], W0 * 1.24, taper=(0.05, 0.05), wobble=0.16)
        pen.stroke([s_roof[2], s_roof[3]], W0 * 1.06, taper=(0.05, 0.05), wobble=0.12)

        # Grand central stave ridge louver / smoke-vent lantern + curling hearth smoke!
        lw = W * (0.28 if grand else 0.18)
        lh = 5.0 if grand else 3.4
        box(pen, P, cx - lw * 0.5, cy - 3.0, cx + lw * 0.5, cy + 3.0, H + R - 1.5, H + R + lh, w=W0 * 0.66)
        _draw_horn_finials(pen, P, cx - lw * 0.5, cy, H + R + lh, 0.0, 1.0, sz=3.8 if grand else 2.8, curved=True)
        _draw_horn_finials(pen, P, cx + lw * 0.5, cy, H + R + lh, 0.0, 1.0, sz=3.8 if grand else 2.8, curved=True)
        _draw_smoke(pen, P, cx, cy, H + R + lh + 1.0, h=14.0 if grand else 11.0)

        # Sweeping carved dragon-prow / horned gable finials on both main ridge ends!
        _draw_horn_finials(pen, P, X0 - ov, cy, H + R, 0.0, 1.0, sz=7.4 if grand else 5.8, curved=True)
        _draw_horn_finials(pen, P, X1 + ov, cy, H + R, 0.0, 1.0, sz=7.4 if grand else 5.8, curved=True)

        # Projecting gabled timber entrance porch on the sunlit south facade!
        pw = min(11.0 if grand else 8.8, W * 0.24)
        pd = 5.5 if grand else 4.2
        ph = H * 0.88
        pr = pw * 0.58
        px0, px1 = cx - pw * 0.5, cx + pw * 0.5
        py0 = Y0 - pd
        p_south = [P(px0, py0, 0), P(px1, py0, 0), P(px1, py0, ph), P(cx, py0, ph + pr), P(px0, py0, ph)]
        p_east = [P(px1, py0, 0), P(px1, Y0, 0), P(px1, Y0, ph), P(px1, py0, ph)]
        p_roof = [P(px1 + 1.2, py0 - 1.0, ph - 0.6), P(px1 + 1.2, Y0 + 1.0, ph - 0.6),
                  P(cx, Y0 + 1.5, ph + pr), P(cx, py0 - 1.0, ph + pr)]
        pen.cv.erase(p_east)
        pen.hatch(p_east, 82, 1.6, W0 * 0.62, coverage=(0.75, 1.0))
        pen.closed(p_east, W0 * 0.72, wobble=0.10)
        pen.cv.erase(p_south)
        pen.blot([P(cx - pw * 0.28, py0, 0), P(cx + pw * 0.28, py0, 0),
                  P(cx + pw * 0.28, py0, ph * 0.72), P(cx, py0, ph * 0.94),
                  P(cx - pw * 0.28, py0, ph * 0.72)], jitter=0.05)
        pen.closed(p_south, W0 * 0.78, wobble=0.10)
        pen.cv.erase(p_roof)
        pen.hatch(p_roof, 74, 1.8, W0 * 0.52, coverage=(0.7, 1.0))
        pen.closed(p_roof, W0 * 0.78, wobble=0.10)
        _draw_horn_finials(pen, P, cx, py0 - 1.0, ph + pr, 1.0, 0.0, sz=4.0, curved=True)
    else:
        # N-S oriented longhouse (gable facing south!)
        south = [P(X0, Y0, 0), P(X1, Y0, 0), P(X1, Y0, H), P(cx, Y0, H + R), P(X0, Y0, H)]
        east = [P(X1, Y0, 0), P(X1, Y1, 0), P(X1, Y1, H), P(X1, Y0, H)]
        pen.cv.erase(south)
        pen.cv.erase(east)
        n_logs = max(4, int(H / 2.4))
        for k in range(1, n_logs):
            hz = H * k / n_logs
            pen.stroke([P(X0 - 1.0, Y0, hz), P(X1 + 1.0, Y0, hz)], W0 * 0.44, taper=(0.05, 0.05), wobble=0.10)
        dw, dh = min(6.2, W * 0.30), min(9.5, H * 0.84)
        pen.blot([P(cx - dw * 0.5, Y0, 0), P(cx + dw * 0.5, Y0, 0),
                  P(cx + dw * 0.5, Y0, dh * 0.75), P(cx, Y0, dh),
                  P(cx - dw * 0.5, Y0, dh * 0.75)], jitter=0.05)
        pen.hatch(east, 82, 1.65, W0 * 0.68, coverage=(0.76, 1.0))
        pen.closed(south, W0 * 0.88, wobble=0.14)
        pen.closed(east, W0 * 0.88, wobble=0.14)

        ov = 2.0
        eh = H - ov * R / (W * 0.5)
        e_roof = [P(X1 + ov, Y0 - ov, eh), P(X1 + ov, Y1 + ov, eh),
                  P(cx, Y1 + ov, H + R), P(cx, Y0 - ov, H + R)]
        pen.cv.erase(e_roof)
        for u in np.linspace(0.03, 0.97, max(10, int(D / 2.3))):
            pen.stroke([P(X1 + ov, Y0 - ov + (D + 2 * ov) * u, eh),
                        P(cx, Y0 - ov + (D + 2 * ov) * u, H + R)],
                       W0 * 0.56, taper=(0.05, 0.45), wobble=0.14)
        pen.hatch(e_roof, 76, 1.9, W0 * 0.48, coverage=(0.68, 1.0))
        pen.closed(e_roof, W0 * 0.90, wobble=0.16)
        pen.stroke([e_roof[0], e_roof[1]], W0 * 1.22, taper=(0.05, 0.05), wobble=0.14)
        pen.beam(P(X0 - ov, Y0 - ov, eh), P(cx, Y0 - ov, H + R), 1.45)
        pen.beam(P(X1 + ov, Y0 - ov, eh), P(cx, Y0 - ov, H + R), 1.45)
        _draw_horn_finials(pen, P, cx, Y0 - ov, H + R, 1.0, 0.0, sz=6.8 if grand else 5.5, curved=True)
        _draw_horn_finials(pen, P, cx, Y1 + ov, H + R, 1.0, 0.0, sz=6.2 if grand else 5.0, curved=True)
        _draw_smoke(pen, P, cx, cy, H + R + 1.0, h=12.0)


def draw_stone_roundhouse(pen, P, rng, X, Y, r=14.5, H=10.0, smoke=True):
    """Highland dry-stone clan roundhouse with rough stone wall courses, timber porch & overhanging thatch cone."""
    W0 = pen.width
    a = P.az
    C = cylinder(pen, P, X, Y, r, 0.0, H, top=False, w=W0 * 0.90)
    # Rough dry-stone masonry courses on the lit west/south wall
    for _ in range(int(r * 0.65)):
        t = a + 1.5 * math.pi + rng.uniform(-0.85, 0.25)
        hz = rng.uniform(1.8, H - 1.8)
        pen.stroke([C(t, hz), C(t + 3.4 / r, hz)], W0 * 0.40, taper=(0.25, 0.25), wobble=0.08)

    # Projecting low timber-framed doorway porch facing south
    pw, pd = 6.6, 5.0
    ph = H * 0.84
    px0, px1 = X - pw * 0.5, X + pw * 0.5
    py0 = Y - r - pd * 0.55
    box(pen, P, px0, py0, px1, Y - r * 0.72, 0.0, ph, w=W0 * 0.76)
    pen.blot([P(X - pw * 0.28, py0, 0), P(X + pw * 0.28, py0, 0),
              P(X + pw * 0.28, py0, ph * 0.76), P(X - pw * 0.28, py0, ph * 0.76)], jitter=0.05)

    # Heavy overhanging conical heather/thatch roof
    apex_h = H + r * rng.uniform(1.18, 1.48)
    cone(pen, P, X, Y, r * 1.24, H - 0.8, apex_h, broken=0.0)
    if smoke and rng.random() < 0.55:
        _draw_smoke(pen, P, X, Y, apex_h + 1.0, h=11.0)


def draw_log_hut(pen, P, rng, X0, Y0, W=28.0, D=20.0, H=10.5, R=10.5, ridge_x=True):
    """Rugged mountain-tribe log cabin hut with notched corner logs, turf/shake roof & firewood stack."""
    draw_viking_longhouse(pen, P, rng, X0, Y0, W=W, D=D, H=H, R=R, ridge_x=ridge_x, grand=False)


# --------------------------------------------------------------------------- standalone tribal categories

def tribal_house(pen, P, rng):
    """Standalone modular mountain-tribe dwelling (hide tent, yurt, A-frame lodge, Viking longhouse, stone roundhouse)."""
    kind = str(rng.choice(["tent", "yurt", "aframe", "longhouse", "roundhouse"],
                          p=[0.24, 0.16, 0.22, 0.22, 0.16]))
    items, boxes = [], []
    if kind == "tent":
        r = rng.uniform(13.5, 17.0)
        H = r * rng.uniform(1.65, 2.05)
        fp = (-r - 2, -r - 2, r + 2, r + 2)
        items.append((fp, ("tent", (0.0, 0.0, r, H))))
        boxes.append(fp)
    elif kind == "yurt":
        r = rng.uniform(14.5, 18.0)
        H = rng.uniform(9.5, 12.0)
        fp = (-r - 2, -r - 2, r + 2, r + 2)
        items.append((fp, ("yurt", (0.0, 0.0, r, H))))
        boxes.append(fp)
    elif kind == "aframe":
        ridge_x = rng.random() < 0.42
        W = rng.uniform(30, 38) if ridge_x else rng.uniform(22, 28)
        D = rng.uniform(20, 25) if ridge_x else rng.uniform(26, 34)
        R = (D if ridge_x else W) * rng.uniform(0.74, 0.92)
        X0, Y0 = -W * 0.5, -D * 0.5
        fp = (X0 - 2, Y0 - 2, X0 + W + 2, Y0 + D + 2)
        items.append((fp, ("aframe", (X0, Y0, W, D, 4.0, R, ridge_x))))
        boxes.append(fp)
    elif kind == "longhouse":
        ridge_x = rng.random() < 0.65
        grand = rng.random() < 0.35
        W = (rng.uniform(52, 64) if grand else rng.uniform(36, 48)) if ridge_x else rng.uniform(22, 27)
        D = (rng.uniform(22, 26) if grand else rng.uniform(18, 22)) if ridge_x else (rng.uniform(44, 56) if grand else rng.uniform(34, 44))
        H = rng.uniform(11.5, 14.0) if grand else rng.uniform(9.8, 12.0)
        R = (D if ridge_x else W) * rng.uniform(0.56, 0.72)
        X0, Y0 = -W * 0.5, -D * 0.5
        fp = (X0 - 2, Y0 - 5, X0 + W + 2, Y0 + D + 2)
        items.append((fp, ("longhouse", (X0, Y0, W, D, H, R, ridge_x, grand))))
        boxes.append(fp)
    else:  # stone roundhouse
        r = rng.uniform(13.5, 17.0)
        H = rng.uniform(9.5, 12.0)
        fp = (-r - 2, -r - 5, r + 2, r + 2)
        items.append((fp, ("roundhouse", (0.0, 0.0, r, H))))
        boxes.append(fp)

    # Occasional compact alpine prop tucked cleanly beside the dwelling (~42%)
    if rng.random() < 0.42:
        b = boxes[0]
        side = 1 if rng.random() < 0.60 else -1
        px = (b[2] + 10.5) if side > 0 else (b[0] - 10.5)
        py = rng.uniform(b[1] + 2, b[3] - 2)
        if rng.random() < 0.55:
            pbox = (px - 9, py - 5, px + 9, py + 5)
            if not _overlaps(pbox, boxes, 1.5):
                items.append((pbox, ("rack", (px, py, None, 0.85))))
                boxes.append(pbox)
        else:
            tbox = (px - 4.5, py - 4.5, px + 4.5, py + 4.5)
            if not _overlaps(tbox, boxes, 1.5):
                items.append((tbox, ("spruce", (px, py, rng.uniform(22, 30)))))
                boxes.append(tbox)

    _render_tribal_items(pen, P, rng, items)


def tribal_prop(pen, P, rng, role="rack"):
    """Modular mountain-tribe camp props: drying racks, bonfires, totems, log palisades & watchtowers."""
    if role == "rack":
        draw_drying_rack(pen, P, rng, 0.0, 0.0, style=None, scale=1.18)
        if rng.random() < 0.45:
            _draw_rock(pen, P, rng, rng.uniform(-11, 11), -6.5, r=rng.uniform(2.2, 3.2))
    elif role == "bonfire":
        draw_bonfire(pen, P, rng, 0.0, 0.0, scale=1.20)
        # Optional 1-2 rough log sitting benches tucked cleanly clear of the fire ring
        if rng.random() < 0.65:
            pen.beam(P(-13.5, -12.5, 1.4), P(-4.5, -14.0, 1.4), 2.1)
        if rng.random() < 0.55:
            pen.beam(P(5.5, -13.5, 1.4), P(14.0, -11.2, 1.4), 2.1)
    elif role == "totem":
        draw_totem(pen, P, rng, 0.0, 0.0, scale=1.22)
    elif role == "palisade":
        ew = rng.random() < 0.55
        L = rng.uniform(76, 98)
        H = rng.uniform(16, 21)
        if ew:
            draw_palisade_seg(pen, P, rng, (-L * 0.5, 0.0), (L * 0.5, 0.0), H=H)
        else:
            draw_palisade_seg(pen, P, rng, (0.0, -L * 0.5), (0.0, L * 0.5), H=H)
    else:  # timber watchtower
        draw_tribal_watchtower(pen, P, rng, 0.0, 0.0, scale=1.15)


# --------------------------------------------------------------------------- shared tribal renderer & packer

def _render_tribal_items(pen, P, rng, items):
    """Exact 3D topological back-to-front draw pass for all tribal structures & alpine props."""
    for kind, args in sort_boxes(P, items):
        if kind == "tent":
            draw_tent(pen, P, rng, *args)
        elif kind == "yurt":
            draw_yurt(pen, P, rng, *args)
        elif kind == "aframe":
            draw_aframe_hut(pen, P, rng, *args)
        elif kind == "longhouse":
            draw_viking_longhouse(pen, P, rng, *args)
        elif kind == "roundhouse":
            draw_stone_roundhouse(pen, P, rng, *args)
        elif kind == "rack":
            draw_drying_rack(pen, P, rng, *args)
        elif kind == "bonfire":
            draw_bonfire(pen, P, rng, *args)
        elif kind == "totem":
            draw_totem(pen, P, rng, *args)
        elif kind == "watchtower":
            draw_tribal_watchtower(pen, P, rng, *args)
        elif kind == "spruce":
            x, y, h = args
            spruce(pen, *P(x, y), h)
        elif kind == "rock":
            _draw_rock(pen, P, rng, *args)
        elif kind == "tuft":
            pen.tuft(*P(*args), rng.uniform(2.8, 4.4))


def _make_tribal_dwelling_item(rng, cx, cy, scale=1.0, kinds=None, probs=None):
    """Create a `(footprint, (kind, args))` tuple for one randomized tribal dwelling centered at `(cx, cy)`."""
    kinds = kinds or ["tent", "yurt", "aframe", "longhouse", "roundhouse"]
    probs = probs or [0.32, 0.16, 0.22, 0.14, 0.16]
    kind = str(rng.choice(kinds, p=probs))
    if kind == "tent":
        r = rng.uniform(11.5, 15.0) * scale
        H = r * rng.uniform(1.62, 2.0)
        return (cx - r - 1.5, cy - r - 1.5, cx + r + 1.5, cy + r + 1.5), ("tent", (cx, cy, r, H))
    if kind == "yurt":
        r = rng.uniform(13.0, 16.0) * scale
        H = rng.uniform(8.8, 11.0) * scale
        return (cx - r - 1.5, cy - r - 1.5, cx + r + 1.5, cy + r + 1.5), ("yurt", (cx, cy, r, H))
    if kind == "aframe":
        ridge_x = rng.random() < 0.45
        W = (rng.uniform(27, 34) if ridge_x else rng.uniform(19, 25)) * scale
        D = (rng.uniform(18, 22) if ridge_x else rng.uniform(24, 31)) * scale
        R = (D if ridge_x else W) * rng.uniform(0.72, 0.88)
        X0, Y0 = cx - W * 0.5, cy - D * 0.5
        return (X0 - 1.5, Y0 - 1.5, X0 + W + 1.5, Y0 + D + 1.5), ("aframe", (X0, Y0, W, D, 3.8 * scale, R, ridge_x))
    if kind == "longhouse":
        ridge_x = rng.random() < 0.65
        W = (rng.uniform(34, 44) if ridge_x else rng.uniform(19, 24)) * scale
        D = (rng.uniform(17, 21) if ridge_x else rng.uniform(32, 42)) * scale
        H = rng.uniform(9.5, 11.8) * scale
        R = (D if ridge_x else W) * rng.uniform(0.54, 0.68)
        X0, Y0 = cx - W * 0.5, cy - D * 0.5
        return (X0 - 1.5, Y0 - 4.5 * scale, X0 + W + 1.5, Y0 + D + 1.5), ("longhouse", (X0, Y0, W, D, H, R, ridge_x, False))
    r = rng.uniform(12.5, 15.5) * scale
    H = rng.uniform(8.8, 11.0) * scale
    return (cx - r - 1.5, cy - r - 4.5 * scale, cx + r + 1.5, cy + r + 1.5), ("roundhouse", (cx, cy, r, H))


# --------------------------------------------------------------------------- tribal camps (3-6 dwellings around bonfire)

def tribal_camp(pen, P, rng):
    """Compact mountain-tribe raiding or hunting camp: 3-6 tents/huts around a bonfire, drying racks & totem."""
    items, boxes = [], []
    # 1. Central roaring clan bonfire pit
    bf_box = (-10.0, -10.0, 10.0, 10.0)
    items.append((bf_box, ("bonfire", (0.0, 0.0, 1.0))))
    boxes.append((-12.5, -12.5, 12.5, 12.5))

    # 2. 4-6 tents, yurts, A-frame huts & roundhouses ringed organically around the campfire
    n_dw = int(rng.integers(4, 7))
    camp_nomad = rng.random() < 0.55
    probs = [0.52, 0.24, 0.16, 0.02, 0.06] if camp_nomad else [0.24, 0.14, 0.28, 0.14, 0.20]
    angs = np.linspace(0.0, TAU, n_dw, endpoint=False) + rng.uniform(0.0, 0.6)
    for ang in angs:
        for dist in (rng.uniform(30, 40), rng.uniform(42, 54)):
            cx = dist * math.cos(ang) * 1.06
            cy = dist * math.sin(ang) * 0.88
            fp, payload = _make_tribal_dwelling_item(rng, cx, cy, scale=0.92, probs=probs)
            if not _overlaps(fp, boxes, 2.0):
                boxes.append(fp)
                items.append((fp, payload))
                break

    # 3. 1-2 hide/meat drying racks + carved clan totem
    for _ in range(int(rng.integers(1, 3))):
        for _ in range(35):
            ang = rng.uniform(0.0, TAU)
            dist = rng.uniform(22, 52)
            rx, ry = dist * math.cos(ang) * 1.08, dist * math.sin(ang) * 0.86
            rbox = (rx - 10, ry - 6, rx + 10, ry + 6)
            if not _overlaps(rbox, boxes, 1.8):
                boxes.append(rbox)
                items.append((rbox, ("rack", (rx, ry, None, 0.90))))
                break
    if rng.random() < 0.78:
        for _ in range(30):
            ang = rng.uniform(0.2, math.pi - 0.2)
            dist = rng.uniform(18, 36)
            tx, ty = dist * math.cos(ang), dist * math.sin(ang)
            tbox = (tx - 6, ty - 5, tx + 6, ty + 5)
            if not _overlaps(tbox, boxes, 1.5):
                boxes.append(tbox)
                items.append((tbox, ("totem", (tx, ty, 0.92))))
                break

    # 4. Mountain boulders & alpine spruce pines framing the camp edges
    bldg_boxes = list(boxes)
    for _ in range(int(rng.integers(2, 5))):
        for _ in range(35):
            b_ref = bldg_boxes[int(rng.integers(0, len(bldg_boxes)))]
            x, y = rng.uniform(b_ref[0] - 16, b_ref[2] + 16), rng.uniform(b_ref[1] - 12, b_ref[3] + 14)
            sbox = (x - 5.0, y - 5.0, x + 5.0, y + 5.0)
            if not _overlaps(sbox, boxes, 1.5):
                boxes.append(sbox)
                items.append((sbox, ("spruce", (x, y, rng.uniform(24, 34)))))
                break
    for _ in range(int(rng.integers(2, 5))):
        for _ in range(25):
            b_ref = bldg_boxes[int(rng.integers(0, len(bldg_boxes)))]
            x, y = rng.uniform(b_ref[0] - 12, b_ref[2] + 12), rng.uniform(b_ref[1] - 10, b_ref[3] + 10)
            rkbox = (x - 4.5, y - 3.5, x + 4.5, y + 3.5)
            if not _overlaps(rkbox, boxes, 1.0):
                boxes.append(rkbox)
                items.append((rkbox, ("rock", (x, y, rng.uniform(3.2, 5.2)))))
                break

    _render_tribal_items(pen, P, rng, items)


# --------------------------------------------------------------------------- tribal villages (highland / Viking clan strongholds!)

def _draw_spiked_gate(pen, P, rng, gx, gy, gw=18.0, H=20.0):
    """Rugged spiked-log tribal palisade gatehouse with flanking timber gate-posts, skull/horn lintel & open arch."""
    W0 = pen.width
    # Heavy twin upright log gate-towers / corner bastions flanking the entry
    for sgn in (-1.0, 1.0):
        px = gx + sgn * (gw * 0.5 + 3.0)
        cylinder(pen, P, px, gy, 4.2, 0.0, H * 1.15, top=False, w=W0 * 0.85)
        # Crown of sharpened palisade spikes on each gate post
        for k in range(-1, 2):
            sx = px + k * 2.2
            pen.beam(P(sx, gy, H * 1.02), P(sx, gy, H * 1.38 + (1.5 if k == 0 else 0.0)), 1.3)
    # Massive lashed timber overhead lintel beam & raised horn-skull arch above entry
    pen.beam(P(gx - gw * 0.5 - 3.5, gy, H * 0.86), P(gx + gw * 0.5 + 3.5, gy, H * 0.86), 2.2)
    pen.beam(P(gx - gw * 0.5 - 2.5, gy, H * 0.68), P(gx + gw * 0.5 + 2.5, gy, H * 0.68), 1.5)
    # Short vertical defensive spikes atop the gatehouse beam
    for sx in np.linspace(gx - gw * 0.42, gx + gw * 0.42, 5):
        pen.stroke([P(sx, gy, H * 0.86), P(sx, gy, H * 1.16)], W0 * 0.76, taper=(0.0, 0.35), wobble=0.04)
    # Dramatic curved horns & carved skull crowning the center of the tribe gate!
    for sgn in (-1.0, 1.0):
        h_pts = [
            P(gx + sgn * 1.2, gy, H * 0.90),
            P(gx + sgn * 6.4, gy, H * 1.05),
            P(gx + sgn * 9.2, gy, H * 1.34),
            P(gx + sgn * 7.2, gy, H * 1.50),
        ]
        pen.stroke(h_pts, W0 * 0.92, taper=(0.05, 0.65), smooth=2, wobble=0.06)
    sk = P(gx, gy - 0.5, H * 0.88)
    sk_p = ellipse_pts(sk[0], sk[1], 2.5, 2.8, 10)
    pen.cv.erase(sk_p)
    pen.closed(sk_p, W0 * 0.68, wobble=0.08)


def tribal_village(pen, P, rng):
    """Mountain-tribe / Viking-style clan village: chieftain longhouse, 10-16 tents/huts, racks, bonfire, totems & optional log palisade."""
    palisaded = rng.random() < 0.58
    nordic = rng.random() < 0.55  # Viking timber/longhouse/A-frame heavy vs nomadic hide-tent/yurt/roundhouse heavy

    H_pal = rng.uniform(15.5, 19.5)
    R_pal = rng.uniform(118, 134)
    vs = None
    if palisaded:
        # Organic 6-sided highland hillfort timber palisade ring with flat south gate!
        vs = [
            (R_pal * math.cos(-math.pi * 0.5 - math.pi / 6 + k * math.pi / 3) * rng.uniform(0.95, 1.06),
             R_pal * math.sin(-math.pi * 0.5 - math.pi / 6 + k * math.pi / 3) * 0.90)
            for k in range(6)
        ]
        # Force south gate endpoints to share exact level y_f so gate joins cleanly
        y_f = min(vs[0][1], vs[1][1])
        vs[0] = (vs[0][0], y_f)
        vs[1] = (vs[1][0], y_f)
        v0, v1, v2, v3, v4, v5 = vs

        # 1. Back palisade walls
        draw_palisade_seg(pen, P, rng, v4, v3, H=H_pal, show_inner=True)
        draw_palisade_seg(pen, P, rng, v5, v4, H=H_pal, show_inner=True)

    items, boxes = [], []

    # Optional rear palisade corner watchtower registered in items & boxes so zero huts overlap its stilts!
    if palisaded and rng.random() < 0.72:
        wx_t, wy_t = (vs[3] if rng.random() < 0.5 else vs[4])
        wx_t, wy_t = wx_t * 0.84, wy_t * 0.84
        wt_fp = (wx_t - 11.5, wy_t - 11.5, wx_t + 11.5, wy_t + 11.5)
        boxes.append(wt_fp)
        items.append((wt_fp, ("watchtower", (wx_t, wy_t, 0.88))))

    # 2. Commanding Chieftain's Great Viking Longhouse overlooking the north side of the clan square!
    ch_w = rng.uniform(54, 66) if nordic else rng.uniform(46, 56)
    ch_d = rng.uniform(22, 26)
    ch_h = rng.uniform(12.0, 14.5)
    ch_r = ch_d * rng.uniform(0.60, 0.74)
    ch_x = -ch_w * 0.5 + rng.uniform(-14, 14)
    ch_y = rng.uniform(26, 46)
    ch_fp = (ch_x - 3, ch_y - 7, ch_x + ch_w + 3, ch_y + ch_d + 3)
    boxes.append(ch_fp)
    items.append((ch_fp, ("longhouse", (ch_x, ch_y, ch_w, ch_d, ch_h, ch_r, True, True))))

    # 3. Central Clan Bonfire Plaza + Carved Clan Totem Pole
    bf_x, bf_y = rng.uniform(-8, 8), rng.uniform(-8, 8)
    bf_fp = (bf_x - 10, bf_y - 10, bf_x + 10, bf_y + 10)
    boxes.append((bf_x - 13, bf_y - 13, bf_x + 13, bf_y + 13))
    items.append((bf_fp, ("bonfire", (bf_x, bf_y, 1.05))))

    for _ in range(30):
        tx = bf_x + rng.choice([-1, 1]) * rng.uniform(16, 26)
        ty = bf_y + rng.uniform(-8, 12)
        t_fp = (tx - 6, ty - 5, tx + 6, ty + 5)
        if not _overlaps(t_fp, boxes, 2.0):
            boxes.append(t_fp)
            items.append((t_fp, ("totem", (tx, ty, 0.96))))
            break

    # 4. 2-3 Hide & Meat Drying Racks placed visibly around the clan yard!
    for _ in range(int(rng.integers(2, 4))):
        for _ in range(45):
            ang = rng.uniform(0.0, TAU)
            rad = rng.uniform(26, 76)
            rx, ry = rad * math.cos(ang) * 1.05, rad * math.sin(ang) * 0.82
            rbox = (rx - 10, ry - 6, rx + 10, ry + 6)
            if not _overlaps(rbox, boxes, 2.5):
                boxes.append(rbox)
                items.append((rbox, ("rack", (rx, ry, None, 0.88))))
                break

    # 5. 10-16 Tribal Dwellings (tents, yurts, A-frame huts, Viking longhouses & highland stone roundhouses)
    in_R = (R_pal * 0.80 - 8.0) if palisaded else 128.0
    gate_clr = (-14.0, -in_R - 10, 14.0, -in_R + 24.0) if palisaded else (999, 999, 999, 999)

    def inside_vil(ax, ay, bx, by):
        if palisaded:
            if not (bx < gate_clr[0] or ax > gate_clr[2] or by < gate_clr[1] or ay > gate_clr[3]):
                return False
            for px, py in ((ax, ay), (bx, ay), (bx, by), (ax, by)):
                if abs(py) > in_R * 0.92:
                    return False
                if abs(px) * 0.866 + abs(py) * 0.5 > in_R:
                    return False
        else:
            for px, py in ((ax, ay), (bx, ay), (bx, by), (ax, by)):
                if (px / 138.0) ** 2 + (py / 108.0) ** 2 > 1.0:
                    return False
        return True

    probs = [0.22, 0.12, 0.28, 0.18, 0.20] if nordic else [0.42, 0.22, 0.16, 0.08, 0.12]
    target_n = int(rng.integers(11, 16))
    placed_n = 0
    for attempt in range(950):
        if placed_n >= target_n:
            break
        ang = rng.uniform(0.0, TAU)
        rad = rng.uniform(32, in_R * 0.95)
        cx = rad * math.cos(ang) * 1.04
        cy = rad * math.sin(ang) * 0.88
        sc = rng.uniform(0.76, 0.92) if attempt < 500 else rng.uniform(0.68, 0.82)
        fp, payload = _make_tribal_dwelling_item(rng, cx, cy, scale=sc, probs=probs)
        gap = 2.0 if attempt < 500 else 1.0
        if inside_vil(*fp) and not _overlaps(fp, boxes, gap):
            boxes.append(fp)
            items.append((fp, payload))
            placed_n += 1

    # Optional inner timber watchtower on open (non-palisaded) villages
    if not palisaded and rng.random() < 0.55:
        for _ in range(30):
            wx, wy = rng.choice([-1, 1]) * rng.uniform(55, 88), rng.uniform(22, 58)
            wt_box = (wx - 10, wy - 10, wx + 10, wy + 10)
            if not _overlaps(wt_box, boxes, 3.0):
                boxes.append(wt_box)
                items.append((wt_box, ("watchtower", (wx, wy, 0.90))))
                break

    # 6. Alpine spruce trees & highland boulders tucked organically into yards
    bldg_boxes = list(boxes)
    n_trees = int(rng.integers(3, 6) if palisaded else rng.integers(6, 11))
    for _ in range(n_trees):
        for _ in range(40):
            b_ref = bldg_boxes[int(rng.integers(0, len(bldg_boxes)))]
            x, y = rng.uniform(b_ref[0] - 18, b_ref[2] + 18), rng.uniform(b_ref[1] - 14, b_ref[3] + 14)
            sbox = (x - 5.0, y - 5.0, x + 5.0, y + 5.0)
            if (not palisaded or inside_vil(*sbox)) and not _overlaps(sbox, boxes, 1.5):
                boxes.append(sbox)
                items.append((sbox, ("spruce", (x, y, rng.uniform(22, 32)))))
                break
    for _ in range(int(rng.integers(3, 7))):
        for _ in range(30):
            b_ref = bldg_boxes[int(rng.integers(0, len(bldg_boxes)))]
            x, y = rng.uniform(b_ref[0] - 14, b_ref[2] + 14), rng.uniform(b_ref[1] - 12, b_ref[3] + 12)
            rkbox = (x - 4.5, y - 3.5, x + 4.5, y + 3.5)
            if (not palisaded or inside_vil(*rkbox)) and not _overlaps(rkbox, boxes, 1.0):
                boxes.append(rkbox)
                items.append((rkbox, ("rock", (x, y, rng.uniform(3.2, 5.2)))))
                break

    _render_tribal_items(pen, P, rng, items)

    # 7. Front & side sharpened-log palisade walls + horned spiked gatehouse + outside alpine spruces
    if palisaded:
        v0, v1, v2, v3, v4, v5 = vs
        draw_palisade_seg(pen, P, rng, v3, v2, H=H_pal, show_inner=False)
        draw_palisade_seg(pen, P, rng, v5, v0, H=H_pal, show_inner=False)
        draw_palisade_seg(pen, P, rng, v2, v1, H=H_pal, show_inner=False)
        gw = rng.uniform(16, 20)
        y_f = v0[1]
        draw_palisade_seg(pen, P, rng, v0, (-gw * 0.5 - 3.5, y_f), H=H_pal, show_inner=False)
        _draw_spiked_gate(pen, P, rng, 0.0, y_f, gw=gw, H=H_pal * 1.12)
        draw_palisade_seg(pen, P, rng, (gw * 0.5 + 3.5, y_f), v1, H=H_pal, show_inner=False)

        # 2-4 alpine spruces & boulders strictly outside & south of the palisade front
        fg_items, fg_occ = [], [(-gw - 10, y_f - 30, gw + 10, y_f + 4)]
        for _ in range(int(rng.integers(3, 6))):
            x = rng.uniform(v0[0] + 6, v1[0] - 6)
            y = y_f - rng.uniform(14, 28)
            sbox = (x - 5.5, y - 5.5, x + 5.5, y + 5.5)
            if not _overlaps(sbox, fg_occ, 2.0):
                fg_occ.append(sbox)
                fg_items.append((sbox, ("spruce", (x, y, rng.uniform(22, 32))) if rng.random() < 0.65
                                else ("rock", (x, y, rng.uniform(3.5, 5.5)))))
        _render_tribal_items(pen, P, rng, fg_items)
