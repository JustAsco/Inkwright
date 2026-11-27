"""Wonderdraft Mountain Brush pools: 3/4 bird's-eye hand-inked alpine mountains, varied ranges, crags & hills.

Built in true 3D world space and projected through the shared 3/4 orthographic camera `Proj`:
- Wide, sweeping concave cartographic slopes (~28-44° flanks flaring smoothly into low valley foothills)
- Organic, meandering, broken watershed ridge spines with subtle crag-face notches (no rigid straight
  vertical bisecting lines or origami sawtooth pyramids)
- Airy, ridge-hugging clustered diagonal woodcut hatching strictly on shadowed east/SE faces, leaving
  generous sunlit parchment breathing room + low-angle horizontal foothill rock-terrace accents
- Generous curved soft-feathered foothill masks (`erase_soft_apron`) so overlapping stamped peaks mask
  rear slopes cleanly while melting seamlessly into the terrain beneath
- Four distinct Mountain Brush pools: classic Alpine Mountains, high-variety Varied Mountains (steep
  peaks, sweeping ridges, flat-topped table mountains, low broad ridges & solo spurs), dramatic Crags,
  and gentle rolling 3/4 Hills.
"""
import math

import numpy as np

from .ink import TAU, chaikin
from .proj import Proj


def _level_proj(P):
    """Level front-3/4 orthographic view (az=0°, same elevation as P) so mountain & hill horizons sit flat."""
    return Proj(P.ox, P.oy, az=0.0, el=math.degrees(math.asin(P.se)))


def _clip_line_poly(poly, a, b):
    """Clip segment a->b to polygon `poly` using scanline intersection; return list of (p0, p1) inside segments."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b - a
    L = float(np.hypot(d[0], d[1]))
    if L < 1e-6:
        return []
    u = d / L
    n = np.array([-u[1], u[0]])
    c = float(a @ n)
    ts = []
    P = np.asarray(poly, float)
    m = len(P)
    for i in range(m):
        p, q = P[i], P[(i + 1) % m]
        dp, dq = float(p @ n - c), float(q @ n - c)
        if (dp < 0.0) != (dq < 0.0):
            hit = p + (q - p) * (dp / (dp - dq))
            ts.append(float((hit - a) @ u))
    ts.sort()
    segs = []
    for k in range(0, len(ts) - 1, 2):
        t0 = max(0.0, ts[k])
        t1 = min(L, ts[k + 1])
        if t1 - t0 > 1.0:
            segs.append((a + u * t0, a + u * t1))
    return segs


def _build_3d_ridge(P, rng, p0, p1, n_pts=11, exp_range=(1.42, 1.82), shoulder=True,
                    crag=False, meander=0.12, step_boost=1.0):
    """Build one organic sweeping concave 3D mountain ridge from (x0, y0, h0) down to (x1, y1, h1)."""
    x0, y0, h0 = p0
    x1, y1, h1 = p1
    dx, dy = x1 - x0, y1 - y0
    dist = max(1e-6, math.hypot(dx, dy))
    nx, ny = -dy / dist, dx / dist
    exp_p = rng.uniform(*exp_range)
    m_amp = dist * rng.uniform(-meander, meander)
    m_phase = rng.uniform(0.0, math.pi)

    # 1-2 gentle geological shoulders / rock terraces along the sweeping slope
    bumps = []
    if shoulder and h0 - h1 > 8.0:
        n_b = 2 if (crag or rng.random() < 0.68) else 1
        for bi in range(n_b):
            bf = rng.uniform(0.20, 0.46) if bi == 0 else rng.uniform(0.50, 0.78)
            bw = rng.uniform(0.08, 0.13) if crag else rng.uniform(0.11, 0.17)
            ba = (h0 - h1) * rng.uniform(0.06, 0.14 if crag else 0.11) * step_boost
            bumps.append((bf, bw, ba))

    pts3 = []
    ts = np.linspace(0.0, 1.0, n_pts)
    for i, t in enumerate(ts):
        lat = m_amp * math.sin(math.pi * t + m_phase * 0.35) * math.sin(math.pi * t)
        if 0 < i < n_pts - 1:
            lat += rng.normal(0.0, dist * (0.040 if crag else 0.026)) * math.sin(math.pi * t)
        wx = x0 + dx * t + nx * lat
        wy = y0 + dy * t + ny * lat
        if i == 0:
            wh = h0
        elif i == n_pts - 1:
            wh = h1
        else:
            wh = h1 + (h0 - h1) * ((1.0 - t) ** exp_p)
            for bf, bw, ba in bumps:
                wh += ba * math.exp(-((t - bf) / bw) ** 2)
            # Crisp eroded rock notches (subtle on sweeping slopes, sharper on crags)
            wh += rng.normal(0.0, (h0 - h1) * (0.044 if crag else 0.022)) * ((1.0 - t) ** 0.65)
            wh = max(h1, min(h0 * (1.02 if crag else 1.004), wh))
        pts3.append((float(wx), float(wy), float(wh)))

    pts2 = [P(wx, wy, wh) for wx, wy, wh in pts3]
    return pts3, pts2


def _stroke_broken_ridge(pen, rng, pts, w_base, start_w=1.0, end_taper=0.88,
                         breaks=1, wobble=0.14):
    """Draw a ridge spine with organic hand-inked breaks & light breathing gaps (no rigid solid bar!)."""
    n = len(pts)
    if n < 2:
        return
    if breaks <= 0 or n < 5:
        pen.stroke(pts, w_base * start_w, taper=(0.08, end_taper), wobble=wobble)
        return
    # Split into 2-3 tapered rocky segments with brief sunlit gaps on the mid/lower slope
    cut = int(rng.integers(max(2, n // 3), max(3, (2 * n) // 3)))
    seg1 = pts[:cut + 1]
    pen.stroke(seg1, w_base * start_w, taper=(0.06, 0.72), wobble=wobble)
    if cut + 1 < n - 1 and rng.random() < 0.80:
        # Pick up lower down as a softer fading foothill fold
        seg2 = pts[cut + 1:]
        if len(seg2) >= 2:
            pen.stroke(seg2, w_base * start_w * 0.68, taper=(0.45, end_taper), wobble=wobble)


def _hatch_ridge_shadow(pen, rng, ridge_pts, clip_poly, angle_deg, spacing, w_base,
                        max_len, taper_down=0.55, gap_prob=0.16, crest_accent=True,
                        crag=False, start_t=0.02, end_t=0.84):
    """Draw airy, clustered, ridge-hugging diagonal woodcut hatching strictly down-right from `ridge_pts`.

    Unlike uniform facet-filling prison bars, strokes originate right along the ridge crest and sweep
    outward along the east fall-line in rhythmic broken clusters, leaving open sunlit parchment space
    across the lower slopes and valley floor (matching classic hand-inked parchment cartography!).
    """
    pts = np.asarray(ridge_pts, float)
    if len(pts) < 2:
        return
    # Resample ridge uniformly along its arc length
    seg_l = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg_l)])
    total = float(cum[-1])
    if total < spacing * 1.2:
        return

    a = math.radians(angle_deg)
    d = np.array([math.cos(a), math.sin(a)])
    n_vec = np.array([-d[1], d[0]])

    s = total * start_t + spacing * rng.uniform(0.15, 0.55)
    s_hi = total * end_t
    idx = 0
    # Organic cluster modulation (bundles of 3-6 strokes with airy breath gaps!)
    cl_phase = rng.uniform(0.0, TAU)
    cl_freq = rng.uniform(2.2, 3.8)
    while s < s_hi:
        frac = s / total
        # Interpolate origin along the ridge
        rx = float(np.interp(s, cum, pts[:, 0]))
        ry = float(np.interp(s, cum, pts[:, 1]))
        # Occasional natural sunlit rock-ledge gap in the mid/lower shadow
        if frac > 0.25 and rng.random() < gap_prob * (0.6 + 0.8 * frac):
            s += spacing * rng.uniform(1.1, 1.9)
            idx += 1
            continue

        # Length swells organically in clusters & fades gently down the slope
        wave = 0.70 + 0.30 * math.sin(frac * math.pi * cl_freq + cl_phase)
        bell = (math.sin(math.pi * min(1.0, max(0.06, frac * 0.92 + 0.08))) ** 0.42)
        if idx % 3 == 0:
            L = max_len * wave * bell * rng.uniform(0.76, 1.15)
        elif idx % 3 == 1:
            L = max_len * wave * bell * rng.uniform(0.46, 0.84)
        else:
            L = max_len * wave * bell * rng.uniform(0.24, 0.58)
        L *= (1.0 - taper_down * (frac ** 1.25))
        if L < 2.2:
            s += spacing
            idx += 1
            continue

        # Slight organic angle drift following the slope flare toward the valley
        ang_j = math.radians(angle_deg + rng.uniform(-3.5, 3.5) - 6.0 * frac)
        dj = np.array([math.cos(ang_j), math.sin(ang_j)])
        nj = np.array([-dj[1], dj[0]])

        p0 = np.array([rx, ry]) + dj * rng.uniform(0.0, 1.2)
        p1 = p0 + dj * L
        for sa, sb in _clip_line_poly(clip_poly, p0, p1):
            sl = float(np.hypot(*(sb - sa)))
            if sl < 1.8:
                continue
            mid = (sa + sb) * 0.5 + nj * rng.uniform(-0.35, 0.55)
            w_cur = w_base * (1.08 - 0.32 * frac) * rng.uniform(0.88, 1.12)
            pen.stroke([sa, mid, sb], w_cur, taper=(0.10, 0.88), smooth=1,
                       wobble=0.10 if crag else 0.08, pressure=0.10)

        # Subtle crisp dark under-crest accent tucked right near the upper summit (never muddy!)
        if crest_accent and frac < 0.30 and L > 6.0 and (crag and idx % 2 == 0 or not crag and idx % 3 == 0):
            s_off = n_vec * (spacing * 0.48)
            sh_l = min(L * 0.38, rng.uniform(3.5, 7.5) * (1.05 - 0.8 * frac))
            q0 = p0 + s_off
            q1 = q0 + dj * sh_l
            for sa, sb in _clip_line_poly(clip_poly, q0, q1):
                if float(np.hypot(*(sb - sa))) > 1.6:
                    pen.stroke([sa, sb], w_base * (0.86 if crag else 0.72),
                               taper=(0.06, 0.88), wobble=0.07)

        s += spacing * rng.uniform(0.86, 1.16)
        idx += 1


def _draw_foothill_strata(pen, rng, base_pt, side, clip_poly, W0, scale=1.0):
    """Delicate low-angle horizontal rock-terrace & foothill scree accents trailing softly off sweeping flanks."""
    bx, by = base_pt
    for k in range(int(rng.integers(1, 3))):
        ox = bx + side * rng.uniform(-6.0, 10.0) * scale
        oy = by + rng.uniform(1.0, 5.0) * scale + k * 3.4
        L = rng.uniform(12.0, 22.0) * scale
        ang = math.radians(rng.uniform(-6.0, 4.0) if side < 0 else rng.uniform(-4.0, 6.0))
        p0 = (ox - math.cos(ang) * L * 0.45, oy - math.sin(ang) * L * 0.45)
        p1 = (ox + math.cos(ang) * L * 0.55, oy + math.sin(ang) * L * 0.55)
        mid = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5 - rng.uniform(0.6, 1.6))
        pen.stroke([p0, mid, p1], W0 * 0.40, taper=(0.55, 0.70), smooth=1, wobble=0.14)
        # 2 tiny gentle shadow strokes tucked under the east slope of the terrace
        for j in range(2):
            f = 0.52 + 0.18 * j
            tx = p0[0] + (p1[0] - p0[0]) * f
            ty = p0[1] + (p1[1] - p0[1]) * f
            pen.stroke([(tx, ty), (tx + rng.uniform(2.2, 4.2), ty + rng.uniform(1.6, 3.0))],
                       W0 * 0.30, taper=(0.20, 0.85), wobble=0.06)


def _draw_foothill_outcrop(pen, P, rng, wx, wy, rx, ry, h, scale_w=0.76):
    """Tiny low-slung rocky foothill nubbin nestled off one lower flank (exact reference charm!)."""
    W0 = pen.width * scale_w
    s3 = (wx + rng.uniform(-0.14, 0.14) * rx, wy + rng.uniform(-0.08, 0.12) * ry, h)
    fL = (wx - rx, wy - ry * rng.uniform(0.05, 0.25), 0.0)
    fR = (wx + rx, wy - ry * rng.uniform(0.05, 0.25), 0.0)
    fM = (wx + rng.uniform(-0.22, 0.22) * rx, wy - ry * rng.uniform(0.82, 1.05), 0.0)
    _, rL = _build_3d_ridge(P, rng, s3, fL, n_pts=6, exp_range=(1.15, 1.48), shoulder=False, meander=0.08)
    _, rR = _build_3d_ridge(P, rng, s3, fR, n_pts=6, exp_range=(1.15, 1.48), shoulder=False, meander=0.08)
    _, rM = _build_3d_ridge(P, rng, s3, fM, n_pts=6, exp_range=(1.10, 1.42), shoulder=False, meander=0.10)
    crest = rL[::-1] + rR[1:]
    apron = list(chaikin([rL[-1], (rM[-1][0], rM[-1][1] + 1.5), rR[-1]], 2))
    apron[0], apron[-1] = rL[-1], rR[-1]
    poly = crest + apron[-2:0:-1]
    pen.cv.erase_soft_apron(poly, apron, feather=6.0)
    _hatch_ridge_shadow(pen, rng, rM, poly, rng.uniform(40.0, 52.0), rng.uniform(2.0, 2.5),
                        W0 * 0.48, max_len=rx * 0.55, taper_down=0.5, gap_prob=0.10, crest_accent=False)
    _stroke_broken_ridge(pen, rng, rM, W0 * 0.62, breaks=0)
    pen.stroke(rL[::-1], W0 * 0.66, taper=(0.72, 0.06), wobble=0.12)
    pen.stroke(rR, W0 * 0.74, taper=(0.06, 0.72), wobble=0.12)


def _draw_cartographic_mountain(pen, P, rng, wx, wy, rx_l, rx_r, ry, h,
                                exp_flank=(1.42, 1.82), blunt=0.0, crag=False,
                                shoulder_spur=True, foothill_nub=False,
                                hatch_reach=0.46, scale_w=1.0):
    """Draw one true 3/4 bird's-eye sweeping cartographic mountain with broken spine & airy clustered hatching.

    Designed to interlock organically when scattered en masse with Wonderdraft's Mountain Brush:
    - Broad, graceful concave flanks (~28-44° slopes) rather than narrow spiky triangles
    - Organic meandering broken watershed spine + classic hooked summit crag-notch on the lit face
    - Clustered ridge-hugging diagonal shading with generous sunlit parchment breathing room
    - Softly feathered lower foothills + delicate low-angle scree/terrace lines
    """
    W0 = pen.width * scale_w
    rx_avg = 0.5 * (rx_l + rx_r)
    sx = wx + rng.uniform(-0.14, 0.14) * rx_avg
    sy = wy + rng.uniform(-0.04, 0.12) * ry

    # Optional chiseled/blunt summit crest ridge spanning [sL -> sR]
    if blunt > 0.02:
        bw = rx_avg * blunt
        tilt = rng.uniform(-0.04, 0.04) * h
        sL = (sx - bw * 0.55, sy + rng.uniform(-1.5, 2.0), h + tilt)
        sR = (sx + bw * 0.55, sy + rng.uniform(-1.5, 2.0), h - tilt)
        n_top = 5
        top3 = []
        for i in range(n_top):
            t = i / (n_top - 1)
            tx = sL[0] + (sR[0] - sL[0]) * t
            ty = sL[1] + (sR[1] - sL[1]) * t + rng.normal(0, 1.0) * math.sin(math.pi * t)
            tz = sL[2] + (sR[2] - sL[2]) * t + rng.uniform(-1.5, 2.0) * math.sin(math.pi * t)
            top3.append((tx, ty, tz))
        top2 = [P(*p) for p in top3]
        top_idx = int(rng.integers(1, n_top - 1))
        spine_start = top3[top_idx]
        top_east = top2[top_idx:]
    else:
        sL = sR = spine_start = (sx, sy, h)
        top2 = [P(*sL)]
        top_east = []

    # Level horizontal wing angles (symmetric around due-west 180° and due-east 0° so mountains never tilt!)
    ang_L = math.radians(rng.uniform(174.0, 190.0))
    ang_R = math.radians(rng.uniform(-10.0, 6.0))
    # Main watershed spine sweeps gently down-front-left (balanced lit west vs shaded east relief!)
    bias_m = -6.0 if sx > wx else 4.0
    ang_M = math.radians(rng.uniform(-114.0, -90.0) + bias_m)

    foot_L = (wx + math.cos(ang_L) * rx_l, wy + math.sin(ang_L) * ry, 0.0)
    foot_R = (wx + math.cos(ang_R) * rx_r, wy + math.sin(ang_R) * ry, 0.0)
    foot_M = (wx + math.cos(ang_M) * rx_avg * rng.uniform(0.54, 0.82),
              wy + math.sin(ang_M) * ry * rng.uniform(0.84, 1.04), 0.0)

    n_seg = 10 if crag else 11
    rL_3, rL_2 = _build_3d_ridge(P, rng, sL, foot_L, n_pts=n_seg, exp_range=exp_flank,
                                 shoulder=True, crag=crag, meander=0.10, step_boost=1.05)
    rR_3, rR_2 = _build_3d_ridge(P, rng, sR, foot_R, n_pts=n_seg, exp_range=exp_flank,
                                 shoulder=True, crag=crag, meander=0.10, step_boost=1.0)
    rM_3, rM_2 = _build_3d_ridge(P, rng, spine_start, foot_M, n_pts=n_seg,
                                 exp_range=(exp_flank[0] * 0.92, exp_flank[1] * 0.95),
                                 shoulder=True, crag=crag, meander=0.16, step_boost=1.12)

    # Secondary SE buttress ridge dropping steep down-SE so east hatching fans cleanly off it (never parallel!)
    rSE_2 = None
    se_bi = None
    foot_SE_2 = None
    if shoulder_spur and rng.random() < (0.80 if crag else 0.66):
        bi = int(rng.integers(2, max(3, (len(rM_3) * 4) // 9)))
        b3 = rM_3[bi]
        ang_SE = math.radians(rng.uniform(-70.0, -48.0))
        fSE = (wx + math.cos(ang_SE) * rx_r * rng.uniform(0.56, 0.78),
               wy + math.sin(ang_SE) * ry * rng.uniform(0.82, 1.02), 0.0)
        _, cand_SE = _build_3d_ridge(P, rng, b3, fSE, n_pts=8, exp_range=(exp_flank[0] * 0.92, exp_flank[1] * 0.96),
                                     shoulder=True, crag=crag, meander=0.09, step_boost=0.88)
        se_ang = math.degrees(math.atan2(cand_SE[-1][1] - cand_SE[0][1], max(1.0, cand_SE[-1][0] - cand_SE[0][0])))
        if rM_2[-1][0] + 10.0 < cand_SE[-1][0] < rR_2[-1][0] - 14.0 and se_ang > 50.0:
            rSE_2 = cand_SE
            se_bi = bi
            foot_SE_2 = rSE_2[-1]

    # Secondary SW shoulder spur sweeping down-left across the sunlit flank
    rSW_2 = None
    foot_SW_2 = None
    if shoulder_spur and rng.random() < (0.74 if rx_l > rx_r * 1.08 else 0.54):
        src = rL_3 if (rx_l > rx_r * 1.10 and rng.random() < 0.55) else rM_3
        bi = int(rng.integers(2, max(3, len(src) // 2)))
        b3 = src[bi]
        ang_SW = math.radians(rng.uniform(-156.0, -128.0))
        fSW = (wx + math.cos(ang_SW) * rx_l * rng.uniform(0.78, 0.98),
               wy + math.sin(ang_SW) * ry * rng.uniform(0.76, 0.98), 0.0)
        _, rSW_2 = _build_3d_ridge(P, rng, b3, fSW, n_pts=7, exp_range=(exp_flank[0] * 0.90, exp_flank[1] * 0.96),
                                   shoulder=True, crag=crag, meander=0.10, step_boost=0.85)
        if rL_2[-1][0] + 8.0 < rSW_2[-1][0] < rM_2[-1][0] - 8.0 and (rSW_2[-1][1] - rSW_2[0][1]) > abs(rSW_2[-1][0] - rSW_2[0][0]) * 0.22:
            foot_SW_2 = rSW_2[-1]
        else:
            rSW_2 = None

    # 2. Projected silhouette & generous soft-feathered curved foothill mask
    crest = rL_2[::-1] + (top2[1:-1] if len(top2) > 2 else []) + rR_2
    apron_anchors = [rL_2[-1]]
    if foot_SW_2 is not None:
        apron_anchors.append((foot_SW_2[0], foot_SW_2[1] + 2.0))
    apron_anchors.append((rM_2[-1][0], rM_2[-1][1] + 2.5))
    if foot_SE_2 is not None:
        apron_anchors.append((foot_SE_2[0], foot_SE_2[1] + 2.0))
    apron_anchors.append(rR_2[-1])
    apron_anchors.sort(key=lambda p: p[0])
    apron_smooth = list(chaikin(apron_anchors, 2))
    apron_smooth[0], apron_smooth[-1] = rL_2[-1], rR_2[-1]

    poly = crest + apron_smooth[-2:0:-1]
    # Generous feathered lower apron so stacked mountains melt softly into terrain with zero hard floor!
    feather_px = max(11.0, min(20.0, ry * 0.34)) if not crag else max(8.5, min(14.0, ry * 0.26))
    pen.cv.erase_soft_apron(poly, apron_smooth, feather=feather_px)

    # 3. Crisp, airy, non-overlapping parallel woodcut shading along shadowed east/SE facets!
    slope_deg = math.degrees(math.atan2(h * P.ce, max(1.0, rx_r)))
    base_ang = float(np.clip(slope_deg + rng.uniform(4.0, 11.0), 32.0, 52.0 if crag else 45.0))
    sp = rng.uniform(2.15, 2.55) if crag else rng.uniform(2.40, 2.85)
    w_hatch = W0 * (0.50 if crag else 0.44)

    east_span = max(18.0, rR_2[-1][0] - rM_2[0][0])
    if rSE_2 is not None:
        # Partition east face into two non-overlapping facets along rSE_2 (zero double-ink mud!)
        poly_mid = list(rM_2[se_bi:]) + [(rM_2[-1][0], rM_2[-1][1] + 2.5), (rSE_2[-1][0], rSE_2[-1][1] + 2.0)] + list(rSE_2[::-1])
        spine_outer = list(rM_2[:se_bi + 1]) + list(rSE_2[1:])
        poly_outer = spine_outer[::-1] + (top_east[1:-1] if len(top_east) > 2 else []) + list(rR_2) + [(rSE_2[-1][0], rSE_2[-1][1] + 2.0)]

        se_ang = math.degrees(math.atan2(rSE_2[-1][1] - rSE_2[0][1], max(1.0, rSE_2[-1][0] - rSE_2[0][0])))
        outer_ang = min(base_ang, max(28.0, se_ang - rng.uniform(18.0, 24.0)))
        # Upper summit + SE buttress facet sweeping cleanly across the outer east slope
        _hatch_ridge_shadow(
            pen, rng, spine_outer, poly_outer, outer_ang, sp, w_hatch,
            max_len=east_span * (hatch_reach * (1.46 if crag else 1.42)),
            taper_down=0.38 if crag else 0.44,
            gap_prob=0.08 if crag else 0.11,
            crest_accent=True, crag=crag,
            start_t=0.02, end_t=0.86 if crag else 0.82,
        )
        # Lower-central shadow gully tucked between rM_2 and rSE_2
        mid_span = max(12.0, rSE_2[-1][0] - rM_2[se_bi][0])
        _hatch_ridge_shadow(
            pen, rng, rM_2[se_bi:], poly_mid, base_ang + rng.uniform(4.0, 9.0), sp * 1.04, w_hatch * 0.94,
            max_len=mid_span * (1.02 if crag else 0.94),
            taper_down=0.44, gap_prob=0.10, crest_accent=False,
            crag=crag, start_t=0.04, end_t=0.82,
        )
    else:
        poly_east = list(rM_2[::-1]) + (top_east[1:-1] if len(top_east) > 2 else []) + list(rR_2) + [(rM_2[-1][0], rM_2[-1][1] + 2.5)]
        _hatch_ridge_shadow(
            pen, rng, rM_2, poly_east, base_ang, sp, w_hatch,
            max_len=east_span * (hatch_reach * (1.42 if crag else 1.38)),
            taper_down=0.40 if crag else 0.48,
            gap_prob=0.08 if crag else 0.11,
            crest_accent=True, crag=crag,
            start_t=0.02, end_t=0.86 if crag else 0.82,
        )

    # Compact shadow tuft tucked strictly on the east underside of the SW shoulder spur
    if rSW_2 is not None and len(rSW_2) >= 3:
        poly_sw = list(rSW_2[::-1]) + list(rM_2) + [(rSW_2[-1][0], rSW_2[-1][1] + 2.0)]
        _hatch_ridge_shadow(
            pen, rng, rSW_2, poly_sw, base_ang + rng.uniform(6.0, 12.0), sp * 1.10, w_hatch * 0.78,
            max_len=rx_avg * (0.20 if crag else 0.16),
            taper_down=0.58, gap_prob=0.20, crest_accent=False,
            crag=crag, start_t=0.08, end_t=0.74,
        )

    # 4. Iconic subtle cartographic rock-notch & lit-flank contour creases (matches reference airiness!)
    # Small crisp '7'-shaped crag facet hook just below the summit on the sunlit west flank
    if rng.random() < 0.78 and len(rL_2) >= 4:
        p_pk = np.asarray(rL_2[0], float)
        p_lw = np.asarray(rL_2[2], float)
        p_mw = np.asarray(rM_2[2], float)
        hq = p_pk * 0.45 + p_lw * 0.30 + p_mw * 0.25 + np.array([rng.uniform(-2.5, 2.5), rng.uniform(2.0, 5.5)])
        hl = rng.uniform(3.2, 6.2)
        hook = [(hq[0] - hl * 0.65, hq[1] + hl * 0.22), (hq[0], hq[1]), (hq[0] + hl * 0.18, hq[1] + hl * 0.78)]
        pen.stroke(hook, W0 * 0.42, taper=(0.35, 0.65), wobble=0.06)

    # 2-4 delicate slope-following rock folds on the sunlit west face
    l_xs, l_ys = [p[0] for p in rL_2], [p[1] for p in rL_2]
    m_xs, m_ys = [p[0] for p in rM_2], [p[1] for p in rM_2]
    w_ymin, w_ymax = rL_2[0][1], rM_2[-1][1]
    for _ in range(int(rng.integers(3, 6) if crag else rng.integers(2, 4))):
        fy = rng.uniform(0.18, 0.74)
        wy_q = w_ymin + (w_ymax - w_ymin) * fy
        xl = float(np.interp(wy_q, l_ys, l_xs))
        xr = float(np.interp(wy_q, m_ys, m_xs))
        if xr - xl < 11.0:
            continue
        x0 = rng.uniform(xl + 5.0, xr - 4.0)
        wl = rng.uniform(8.0, 18.0)
        ang = rng.uniform(2.38, 2.82)
        p2 = (x0 + math.cos(ang) * wl, wy_q + math.sin(ang) * wl)
        for sa, sb in _clip_line_poly(poly, (x0, wy_q), p2):
            if float(np.hypot(*(sb - sa))) > 3.2:
                mid = (sa + sb) * 0.5 + np.array([0.8, 0.6])
                pen.stroke([sa, mid, sb], W0 * (0.42 if crag else 0.34),
                           taper=(0.35, 0.85), smooth=1, wobble=0.10)

    # Low-angle horizontal foothill rock-terrace accents along the sweeping outer skirts!
    if not crag and rng.random() < 0.72:
        _draw_foothill_strata(pen, rng, rL_2[-2], -1, poly, W0, scale=scale_w)
    if not crag and rng.random() < 0.50:
        _draw_foothill_strata(pen, rng, rR_2[-2], 1, poly, W0, scale=scale_w)

    # 5. Delicate hand-inked broken ridge spines & softly tapering flank silhouettes
    if rSW_2 is not None:
        _stroke_broken_ridge(pen, rng, rSW_2, W0 * (0.64 if crag else 0.54), breaks=1)
    if rSE_2 is not None:
        _stroke_broken_ridge(pen, rng, rSE_2, W0 * (0.70 if crag else 0.58), breaks=1)

    _stroke_broken_ridge(pen, rng, rM_2, W0 * (0.78 if crag else 0.64),
                         breaks=0 if crag else 1, end_taper=0.92)
    if len(top2) >= 2:
        pen.stroke(top2, W0 * 0.78, taper=(0.06, 0.06), wobble=0.14)
    # Taper both outer flanks down to delicate wisps so overlapping ranges merge seamlessly!
    pen.stroke(rL_2[::-1], W0 * (0.76 if crag else 0.68), taper=(0.76, 0.05), wobble=0.14)
    pen.stroke(rR_2, W0 * (0.84 if crag else 0.74), taper=(0.05, 0.76), wobble=0.15)

    # Optional tiny detached rocky foothill outcrop nestled beside the long flank
    if foothill_nub:
        sgn = -1 if rx_l >= rx_r else 1
        nx = wx + sgn * (rx_l if sgn < 0 else rx_r) * rng.uniform(0.72, 0.96)
        ny = wy - ry * rng.uniform(0.28, 0.58)
        _draw_foothill_outcrop(pen, P, rng, nx, ny, rng.uniform(18, 28), rng.uniform(12, 18),
                               rng.uniform(7, 13), scale_w=scale_w * 0.74)


def _draw_table_mountain_3d(pen, P, rng, wx, wy, rx_l, rx_r, ry, h,
                            top_frac_x=0.38, top_frac_y=0.07, tilt_h=0.0, scale_w=1.0):
    """Level 3/4 weathered flat-topped table mountain / rocky mesa ridge (never a volcano crater!)."""
    W0 = pen.width * scale_w
    rx_avg = 0.5 * (rx_l + rx_r)
    tx = rx_avg * top_frac_x
    ty = ry * top_frac_y
    sx = wx + rng.uniform(-0.08, 0.08) * rx_avg
    sy = wy + rng.uniform(0.02, 0.10) * ry

    # Craggy, slightly stepped horizontal rock-table crest (flat uneven shelf, NOT a round crater rim!)
    n_b = int(rng.integers(5, 8))
    n_f = int(rng.integers(5, 7))
    z_l = h + tilt_h * 0.5
    z_r = h - tilt_h * 0.5
    # Optional weathered step-down on one edge of the flat plateau so it reads as natural stratified rock
    step_side = int(rng.choice([-1, 0, 1]))
    step_drop = rng.uniform(2.5, 5.0) if step_side != 0 else 0.0

    def _plateau_z(t):
        z = z_l + (z_r - z_l) * t
        if step_side < 0 and t < 0.28:
            z -= step_drop * (1.0 - t / 0.28)
        elif step_side > 0 and t > 0.72:
            z -= step_drop * ((t - 0.72) / 0.28)
        return z

    back_rim_3, front_rim_3 = [], []
    for i in range(n_b):
        t = i / (n_b - 1)
        x = sx - tx + 2.0 * tx * t
        y = sy + ty * rng.uniform(0.35, 0.85)
        z = _plateau_z(t) + (rng.normal(0.0, h * 0.022) if 0 < i < n_b - 1 else 0.0)
        back_rim_3.append((x, y, max(h * 0.76, z)))
    front_rim_3.append(back_rim_3[0])
    for i in range(1, n_f - 1):
        t = i / (n_f - 1)
        x = sx - tx + 2.0 * tx * t + rng.normal(0.0, tx * 0.06)
        # Jagged, faceted cliff lip — stays close to straight E-W rather than bowing into an oval
        y = sy - ty * rng.uniform(0.45, 1.05)
        z = _plateau_z(t) + rng.normal(0.0, h * 0.020)
        front_rim_3.append((x, y, max(h * 0.74, z)))
    front_rim_3.append(back_rim_3[-1])

    back_rim_2 = [P(*p) for p in back_rim_3]
    front_rim_2 = [P(*p) for p in front_rim_3]

    i_M = n_f // 2
    i_SE = min(n_f - 2, (3 * n_f) // 4) if n_f >= 5 else None
    i_SW = max(1, n_f // 4) if n_f >= 5 else None

    # Level horizontal sweeping concave skirts fanning symmetrically down to both flanks
    ang_L = math.radians(rng.uniform(174.0, 190.0))
    ang_R = math.radians(rng.uniform(-10.0, 6.0))
    ang_M = math.radians(rng.uniform(-112.0, -88.0))
    foot_L = (wx + math.cos(ang_L) * rx_l, wy + math.sin(ang_L) * ry, 0.0)
    foot_R = (wx + math.cos(ang_R) * rx_r, wy + math.sin(ang_R) * ry, 0.0)
    foot_M = (wx + math.cos(ang_M) * rx_avg * rng.uniform(0.54, 0.80),
              wy + math.sin(ang_M) * ry * rng.uniform(0.84, 1.04), 0.0)

    _, rL_2 = _build_3d_ridge(P, rng, back_rim_3[0], foot_L, n_pts=10, exp_range=(1.44, 1.82),
                              shoulder=True, meander=0.09, step_boost=1.08)
    _, rR_2 = _build_3d_ridge(P, rng, back_rim_3[-1], foot_R, n_pts=10, exp_range=(1.44, 1.82),
                              shoulder=True, meander=0.09, step_boost=1.05)
    _, rM_2 = _build_3d_ridge(P, rng, front_rim_3[i_M], foot_M, n_pts=9, exp_range=(1.38, 1.74),
                              shoulder=True, meander=0.13, step_boost=1.0)

    rSE_2 = None
    if i_SE is not None and i_SE != i_M and i_SE != n_f - 1:
        ang_SE = math.radians(rng.uniform(-70.0, -46.0))
        fSE = (wx + math.cos(ang_SE) * rx_r * rng.uniform(0.58, 0.80),
               wy + math.sin(ang_SE) * ry * rng.uniform(0.82, 1.02), 0.0)
        _, cand_SE = _build_3d_ridge(P, rng, front_rim_3[i_SE], fSE, n_pts=8, exp_range=(1.38, 1.74),
                                     shoulder=True, meander=0.09, step_boost=0.88)
        se_ang = math.degrees(math.atan2(cand_SE[-1][1] - cand_SE[0][1], max(1.0, cand_SE[-1][0] - cand_SE[0][0])))
        if rM_2[-1][0] + 10.0 < cand_SE[-1][0] < rR_2[-1][0] - 14.0 and se_ang > 50.0:
            rSE_2 = cand_SE
    rSW_2 = None
    if i_SW is not None and i_SW != i_M and i_SW != 0:
        ang_SW = math.radians(rng.uniform(-154.0, -126.0))
        fSW = (wx + math.cos(ang_SW) * rx_l * rng.uniform(0.78, 0.96),
               wy + math.sin(ang_SW) * ry * rng.uniform(0.78, 0.98), 0.0)
        _, cand_SW = _build_3d_ridge(P, rng, front_rim_3[i_SW], fSW, n_pts=7, exp_range=(1.38, 1.74),
                                     shoulder=True, meander=0.09, step_boost=0.85)
        if rL_2[-1][0] + 8.0 < cand_SW[-1][0] < rM_2[-1][0] - 8.0:
            rSW_2 = cand_SW

    crest = rL_2[::-1] + back_rim_2[1:-1] + rR_2

    apron_anchors = [rL_2[-1]]
    if rSW_2 is not None:
        apron_anchors.append((rSW_2[-1][0], rSW_2[-1][1] + 2.0))
    apron_anchors.append((rM_2[-1][0], rM_2[-1][1] + 2.5))
    if rSE_2 is not None:
        apron_anchors.append((rSE_2[-1][0], rSE_2[-1][1] + 2.0))
    apron_anchors.append(rR_2[-1])
    apron_anchors.sort(key=lambda p: p[0])
    apron_smooth = list(chaikin(apron_anchors, 2))
    apron_smooth[0], apron_smooth[-1] = rL_2[-1], rR_2[-1]

    poly = crest + apron_smooth[-2:0:-1]
    pen.cv.erase_soft_apron(poly, apron_smooth, feather=max(12.0, min(19.0, ry * 0.34)))

    # Crisp non-overlapping east-facet & cliff-drop hatching (clear level 3D table relief!)
    slope_deg = math.degrees(math.atan2(h * P.ce, max(1.0, rx_r)))
    base_ang = float(np.clip(slope_deg + rng.uniform(6.0, 14.0), 34.0, 48.0))
    sp = rng.uniform(2.35, 2.80)
    w_h = W0 * 0.44
    east_span = max(18.0, rR_2[-1][0] - rM_2[0][0])

    if rSE_2 is not None:
        poly_mid = list(rM_2) + [(rM_2[-1][0], rM_2[-1][1] + 2.5), (rSE_2[-1][0], rSE_2[-1][1] + 2.0)] + list(rSE_2[::-1]) + list(front_rim_2[i_SE - 1:i_M:-1])
        poly_outer = list(rSE_2[::-1]) + list(front_rim_2[i_SE + 1:-1]) + list(rR_2) + [(rSE_2[-1][0], rSE_2[-1][1] + 2.0)]
        mid_span = max(14.0, rSE_2[-1][0] - rM_2[0][0])
        se_span = max(14.0, rR_2[-1][0] - rSE_2[0][0])
        se_ang = math.degrees(math.atan2(rSE_2[-1][1] - rSE_2[0][1], max(1.0, rSE_2[-1][0] - rSE_2[0][0])))
        outer_ang = min(base_ang, max(28.0, se_ang - rng.uniform(18.0, 24.0)))
        _hatch_ridge_shadow(pen, rng, rM_2, poly_mid, base_ang + rng.uniform(4.0, 9.0), sp, w_h,
                            max_len=mid_span * 0.92, taper_down=0.44, gap_prob=0.10,
                            crest_accent=False, start_t=0.02, end_t=0.80)
        _hatch_ridge_shadow(pen, rng, rSE_2, poly_outer, outer_ang, sp * 0.98, w_h * 0.98,
                            max_len=se_span * 0.96, taper_down=0.42, gap_prob=0.09,
                            crest_accent=True, start_t=0.02, end_t=0.82)
    else:
        poly_east = list(rM_2[::-1]) + list(front_rim_2[i_M + 1:-1]) + list(rR_2) + [(rM_2[-1][0], rM_2[-1][1] + 2.5)]
        _hatch_ridge_shadow(pen, rng, rM_2, poly_east, base_ang, sp, w_h,
                            max_len=east_span * 0.78, taper_down=0.44, gap_prob=0.10,
                            crest_accent=True, start_t=0.02, end_t=0.80)

    # Crisp short cliff-drop ticks tucked right beneath the shadowed east half of the flat table lip
    east_rim = front_rim_2[i_M:]
    cliff_poly = list(east_rim) + list(rR_2) + [(rM_2[-1][0], rM_2[-1][1] + 2.5)] + list(rM_2[::-1])
    _hatch_ridge_shadow(pen, rng, east_rim, cliff_poly, rng.uniform(68.0, 80.0), sp * 1.05, w_h * 0.82,
                        max_len=min(12.0, h * 0.28), taper_down=0.22, gap_prob=0.12,
                        crest_accent=False, start_t=0.06, end_t=0.92)

    # Subtle horizontal rock-shelf creases across the sunlit west face + foothill strata
    if rng.random() < 0.70:
        _draw_foothill_strata(pen, rng, rL_2[-2], -1, poly, W0, scale=scale_w)
    if rng.random() < 0.45:
        _draw_foothill_strata(pen, rng, rR_2[-2], 1, poly, W0, scale=scale_w)

    # Delicate broken ridge spines & weathered table rim (keep front rim broken/subtle so it reads as rock cliff, not a crater!)
    if rSW_2 is not None:
        _stroke_broken_ridge(pen, rng, rSW_2, W0 * 0.48, breaks=1)
    if rSE_2 is not None:
        _stroke_broken_ridge(pen, rng, rSE_2, W0 * 0.54, breaks=1)
    _stroke_broken_ridge(pen, rng, rM_2, W0 * 0.60, breaks=1, end_taper=0.92)
    # Emphasize the shadowed east front rim lip, taper lightly across the sunlit west shelf
    pen.stroke(front_rim_2[i_M:], W0 * 0.60, taper=(0.18, 0.14), wobble=0.14)
    if i_M >= 2:
        pen.stroke(front_rim_2[1:i_M + 1], W0 * 0.36, taper=(0.65, 0.25), wobble=0.12)
    pen.stroke(back_rim_2, W0 * 0.74, taper=(0.06, 0.06), wobble=0.15)
    pen.stroke(rL_2[::-1], W0 * 0.68, taper=(0.76, 0.05), wobble=0.14)
    pen.stroke(rR_2, W0 * 0.74, taper=(0.05, 0.76), wobble=0.15)


def _draw_hill_3d(pen, P, rng, wx, wy, rx, ry, h, scale_w=1.0):
    """Draw one gentle, softly rounded level 3/4 bird's-eye rolling downland hill with sweeping SE hachures."""
    W0 = pen.width * scale_w
    phi = math.radians(rng.uniform(-6.0, 6.0))
    cp, sp = math.cos(phi), math.sin(phi)

    def local_3d(u, v, z):
        return (wx + u * cp - v * sp, wy + u * sp + v * cp, z)

    n_ring = 64
    thetas = np.linspace(0.0, TAU, n_ring, endpoint=False)
    ring_3 = [local_3d(rx * math.cos(t), ry * math.sin(t), 0.0) for t in thetas]
    ring_2 = [P(*p) for p in ring_3]
    i_left = int(np.argmin([p[0] for p in ring_2]))
    i_right = int(np.argmax([p[0] for p in ring_2]))

    path_fwd, idx = [], i_left
    while True:
        path_fwd.append(idx)
        if idx == i_right:
            break
        idx = (idx + 1) % n_ring
    path_bwd, idx = [], i_left
    while True:
        path_bwd.append(idx)
        if idx == i_right:
            break
        idx = (idx - 1) % n_ring
    y_fwd = float(np.mean([ring_2[i][1] for i in path_fwd]))
    y_bwd = float(np.mean([ring_2[i][1] for i in path_bwd]))
    front_arc = path_fwd if y_fwd >= y_bwd else path_bwd
    apron_2 = [ring_2[i] for i in front_arc]

    pL_3, pR_3 = ring_3[i_left], ring_3[i_right]
    pk_t = rng.uniform(0.36, 0.58)
    has_sub = rng.random() < 0.52
    sub_t = rng.uniform(0.68, 0.80) if pk_t < 0.48 else rng.uniform(0.20, 0.34)
    sub_a = h * rng.uniform(0.18, 0.30) if has_sub else 0.0
    brow_bend = rng.uniform(-0.14, 0.14) * ry

    n_c = 24
    ts = np.linspace(0.0, 1.0, n_c)
    crest_3 = []
    for t in ts:
        u = (pk_t - t) / max(1e-6, pk_t) if t <= pk_t else (t - pk_t) / max(1e-6, 1.0 - pk_t)
        u = min(1.0, max(0.0, u))
        hz = h * (0.5 * (1.0 + math.cos(math.pi * (u ** 1.08)))) ** 0.94
        if has_sub:
            hz += sub_a * math.exp(-((t - sub_t) / 0.14) ** 2) * math.sin(math.pi * t)
        cx3 = pL_3[0] + (pR_3[0] - pL_3[0]) * t
        cy3 = pL_3[1] + (pR_3[1] - pL_3[1]) * t + brow_bend * math.sin(math.pi * t)
        crest_3.append((cx3, cy3, max(0.0, hz)))
    crest_2 = list(chaikin([P(*p) for p in crest_3], 1))
    crest_2[0], crest_2[-1] = ring_2[i_left], ring_2[i_right]

    poly = crest_2 + apron_2[-2:0:-1]
    pen.cv.erase_soft_apron(poly, apron_2, feather=max(8.0, ry * 0.32))

    pk_i = int(np.argmin([p[1] for p in crest_2]))
    east_crest = crest_2[pk_i:]
    n_ec = len(east_crest)
    for k in range(1, n_ec - 2, 2):
        frac = k / max(1, n_ec - 2)
        bx, by = east_crest[k]
        bell = math.sin(math.pi * min(1.0, max(0.08, frac))) ** 0.55
        base_len = (h * 0.72 + ry * 0.42) * bell * rng.uniform(0.68, 1.16)
        if (k // 2) % 2 == 0:
            base_len *= 1.15
        if base_len < 3.5:
            continue
        st_drop = rng.uniform(1.0, 3.4) * (1.0 - 0.45 * frac)
        ang = math.radians(rng.uniform(62.0, 74.0) - 6.0 * frac)
        p0 = (bx + rng.uniform(-1.0, 1.0), by + st_drop)
        p2 = (p0[0] + math.cos(ang) * base_len, p0[1] + math.sin(ang) * base_len)
        for sa, sb in _clip_line_poly(poly, p0, p2):
            sl = float(np.hypot(*(sb - sa)))
            if sl < 2.2:
                continue
            mid = (sa + sb) * 0.5 + np.array([sl * 0.12, -sl * 0.05])
            pen.stroke([sa, mid, sb], W0 * (0.44 * (1.06 - 0.28 * frac)),
                       taper=(0.24, 0.88), smooth=1, wobble=0.08)

    for k in range(2, n_ec - 3, 3):
        frac = k / max(1, n_ec - 2)
        if rng.random() < 0.35:
            continue
        bx, by = east_crest[k]
        off_y = rng.uniform(5.0, max(6.5, h * 0.45 + ry * 0.12)) * math.sin(math.pi * frac)
        p0 = (bx + off_y * 0.32, by + off_y)
        sl_l = rng.uniform(5.5, 11.5) * (1.05 - 0.40 * frac)
        ang = math.radians(rng.uniform(58.0, 70.0))
        p2 = (p0[0] + math.cos(ang) * sl_l, p0[1] + math.sin(ang) * sl_l)
        for sa, sb in _clip_line_poly(poly, p0, p2):
            if float(np.hypot(*(sb - sa))) > 2.4:
                mid = (sa + sb) * 0.5 + np.array([0.8, -0.3])
                pen.stroke([sa, mid, sb], W0 * 0.34, taper=(0.35, 0.88), smooth=1, wobble=0.07)

    if rng.random() < 0.55 and pk_i > 4:
        ci = int(rng.integers(2, max(3, pk_i - 1)))
        bx, by = crest_2[ci]
        p0 = (bx + rng.uniform(1.0, 4.0), by + rng.uniform(4.5, max(5.5, h * 0.35)))
        wl = rng.uniform(9.0, 17.0)
        p2 = (p0[0] - wl * 0.86, p0[1] + wl * 0.36)
        for sa, sb in _clip_line_poly(poly, p0, p2):
            if float(np.hypot(*(sb - sa))) > 3.5:
                mid = (sa + sb) * 0.5 + np.array([0.0, -0.7])
                pen.stroke([sa, mid, sb], W0 * 0.32, taper=(0.45, 0.85), smooth=1, wobble=0.08)

    pen.stroke(crest_2[:pk_i + 1], W0 * 0.66, taper=(0.82, 0.06), wobble=0.13)
    pen.stroke(crest_2[pk_i:], W0 * 0.76, taper=(0.06, 0.82), wobble=0.14)


# --------------------------------------------------------------------------- Wonderdraft Mountain Brush pool generators

def varied_mountain(pen, P, rng):
    """High-variety cartographic mountain pool: sweeping asymmetric ridges, steep peaks, flat-tops & low massifs."""
    P = _level_proj(P)
    kind = str(rng.choice(
        ["sweeping_ridge", "flat_top", "steep_peak", "low_broad", "solo_peak"],
        p=[0.30, 0.22, 0.18, 0.16, 0.14],
    ))
    if kind == "sweeping_ridge":
        # Wide sweeping concave cartographic peak with balanced asymmetric valley spur
        long_left = rng.random() < 0.50
        rx_long = rng.uniform(92, 124)
        rx_short = rng.uniform(62, 86)
        _draw_cartographic_mountain(
            pen, P, rng, 0.0, 0.0,
            rx_long if long_left else rx_short,
            rx_short if long_left else rx_long,
            rng.uniform(44, 58), rng.uniform(44, 66),
            exp_flank=(1.44, 1.84),
            blunt=(rng.uniform(0.08, 0.15) if rng.random() < 0.35 else 0.0),
            crag=False, shoulder_spur=True,
            foothill_nub=(rng.random() < 0.55), hatch_reach=0.52,
        )
    elif kind == "steep_peak":
        # Dramatic steep concave alpine horn / solitary peak flaring gracefully into valley skirts
        rx_l = rng.uniform(66, 88)
        rx_r = rng.uniform(62, 84)
        _draw_cartographic_mountain(
            pen, P, rng, 0.0, 0.0, rx_l, rx_r,
            rng.uniform(42, 54), rng.uniform(62, 84),
            exp_flank=(1.46, 1.84), blunt=0.0, crag=False,
            shoulder_spur=True, foothill_nub=(rng.random() < 0.35),
            hatch_reach=0.56,
        )
    elif kind == "flat_top":
        # Level weathered flat-topped / beveled rocky table mountain with sweeping concave skirts
        long_left = rng.random() < 0.50
        rx_l = rng.uniform(86, 112) if long_left else rng.uniform(68, 92)
        rx_r = rng.uniform(68, 92) if long_left else rng.uniform(86, 112)
        _draw_table_mountain_3d(
            pen, P, rng, 0.0, 0.0, rx_l, rx_r,
            rng.uniform(44, 56), rng.uniform(36, 54),
            top_frac_x=rng.uniform(0.32, 0.46), top_frac_y=rng.uniform(0.05, 0.09),
            tilt_h=rng.uniform(-2.0, 2.0),
        )
    elif kind == "low_broad":
        # Low, wide sweeping foothill ridge / weathered mountain pass with broad blunt crest
        _draw_cartographic_mountain(
            pen, P, rng, 0.0, 0.0,
            rng.uniform(76, 106), rng.uniform(72, 102),
            rng.uniform(40, 52), rng.uniform(22, 36),
            exp_flank=(1.18, 1.50), blunt=rng.uniform(0.14, 0.26),
            crag=False, shoulder_spur=(rng.random() < 0.70),
            foothill_nub=(rng.random() < 0.50), hatch_reach=0.50,
        )
    else:
        # Compact solitary peak / minor foothill horn (fills gaps naturally between larger mountains!)
        rx = rng.uniform(46, 66)
        _draw_cartographic_mountain(
            pen, P, rng, 0.0, 0.0,
            rx * rng.uniform(0.90, 1.14), rx * rng.uniform(0.90, 1.14),
            rng.uniform(30, 42), rng.uniform(28, 44),
            exp_flank=(1.32, 1.66),
            blunt=(rng.uniform(0.08, 0.14) if rng.random() < 0.30 else 0.0),
            crag=False, shoulder_spur=(rng.random() < 0.55),
            foothill_nub=False, hatch_reach=0.52, scale_w=0.90,
        )


def mountain(pen, P, rng):
    """Wonderdraft Mountain Brush pool: classic sweeping level 3/4 alpine peaks & shouldered massifs."""
    P = _level_proj(P)
    r = rng.random()
    if r < 0.62:
        # Noble sweeping single alpine mountain (interlocks cleanly into grand ranges without clutter!)
        long_left = rng.random() < 0.50
        rx_a = rng.uniform(86, 114)
        rx_b = rng.uniform(66, 88)
        _draw_cartographic_mountain(
            pen, P, rng, 0.0, 0.0,
            rx_a if long_left else rx_b, rx_b if long_left else rx_a,
            rng.uniform(44, 56), rng.uniform(52, 74),
            exp_flank=(1.42, 1.78), blunt=0.0, crag=False,
            shoulder_spur=True, foothill_nub=(rng.random() < 0.45),
            hatch_reach=0.56,
        )
    else:
        # Well-spaced twin alpine massif (one majestic peak + one lower broad shoulder peak)
        side = int(rng.choice([-1, 1]))
        rx1, ry1, h1 = rng.uniform(74, 92), rng.uniform(42, 52), rng.uniform(54, 72)
        rx2, ry2, h2 = rx1 * rng.uniform(0.76, 0.90), ry1 * rng.uniform(0.76, 0.90), h1 * rng.uniform(0.64, 0.82)
        peaks = [
            (side * rx1 * rng.uniform(0.54, 0.72), ry1 * rng.uniform(0.28, 0.50), rx2, ry2, h2, 0.90),
            (-side * rx1 * rng.uniform(0.18, 0.32), -ry1 * rng.uniform(0.12, 0.24), rx1, ry1, h1, 1.0),
        ]
        for wx, wy, rx, ry, h, sw in sorted(peaks, key=lambda it: -P.depth(it[0], it[1])):
            _draw_cartographic_mountain(
                pen, P, rng, wx, wy, rx * rng.uniform(0.92, 1.08), rx * rng.uniform(0.92, 1.08),
                ry, h, exp_flank=(1.40, 1.74), blunt=0.0, crag=False,
                shoulder_spur=True, foothill_nub=False, hatch_reach=0.54, scale_w=sw,
            )


def crag(pen, P, rng):
    """Wonderdraft Mountain Brush pool: steep serrated level 3/4 granite spires & dramatic fractured crags."""
    P = _level_proj(P)
    r = rng.random()
    if r < 0.58:
        rx = rng.uniform(56, 74)
        _draw_cartographic_mountain(
            pen, P, rng, 0.0, 0.0,
            rx * rng.uniform(0.90, 1.10), rx * rng.uniform(0.90, 1.10),
            rng.uniform(38, 50), rng.uniform(68, 94),
            exp_flank=(1.22, 1.52), blunt=0.0, crag=True,
            shoulder_spur=True, foothill_nub=(rng.random() < 0.35),
            hatch_reach=0.62,
        )
    else:
        side = int(rng.choice([-1, 1]))
        rx1, ry1, h1 = rng.uniform(52, 66), rng.uniform(36, 46), rng.uniform(64, 88)
        rx2, ry2, h2 = rx1 * rng.uniform(0.74, 0.88), ry1 * rng.uniform(0.74, 0.88), h1 * rng.uniform(0.66, 0.86)
        peaks = [
            (side * rx1 * rng.uniform(0.52, 0.72), ry1 * rng.uniform(0.28, 0.50), rx2, ry2, h2, 0.90),
            (-side * rx1 * rng.uniform(0.18, 0.32), -ry1 * rng.uniform(0.12, 0.24), rx1, ry1, h1, 1.0),
        ]
        for wx, wy, rx, ry, h, sw in sorted(peaks, key=lambda it: -P.depth(it[0], it[1])):
            _draw_cartographic_mountain(
                pen, P, rng, wx, wy, rx * rng.uniform(0.94, 1.06), rx * rng.uniform(0.94, 1.06), ry, h,
                exp_flank=(1.20, 1.50), blunt=0.0, crag=True,
                shoulder_spur=True, foothill_nub=False, hatch_reach=0.60, scale_w=sw,
            )


def hill(pen, P, rng):
    """Wonderdraft Mountain Brush pool: gentle level 3/4 bird's-eye rolling hills & overlapping highland downs."""
    P = _level_proj(P)
    r = rng.random()
    if r < 0.45:
        _draw_hill_3d(pen, P, rng, 0.0, 0.0, rng.uniform(52, 70), rng.uniform(30, 42), rng.uniform(15, 24))
    else:
        side = int(rng.choice([-1, 1]))
        rx1, ry1, h1 = rng.uniform(44, 58), rng.uniform(28, 38), rng.uniform(14, 22)
        rx2, ry2, h2 = rx1 * rng.uniform(0.78, 0.96), ry1 * rng.uniform(0.78, 0.96), h1 * rng.uniform(0.76, 0.96)
        hills = [
            (side * rx1 * rng.uniform(0.46, 0.68), ry1 * rng.uniform(0.42, 0.72), rx2, ry2, h2, 0.92),
            (-side * rx1 * rng.uniform(0.18, 0.34), -ry1 * rng.uniform(0.16, 0.32), rx1, ry1, h1, 1.0),
        ]
        for wx, wy, rx, ry, h, sw in sorted(hills, key=lambda it: -P.depth(it[0], it[1])):
            _draw_hill_3d(pen, P, rng, wx, wy, rx, ry, h, scale_w=sw)
