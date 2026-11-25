"""Wonderdraft Mountain Brush pools: 3/4 bird's-eye hand-inked alpine mountains, jagged crags, and rolling hills.

Built in true 3D world space and projected through the shared 3/4 orthographic camera `Proj` (matching every
settlement, castle, temple, lair, and cliff in Inkwright — viewed from above and south-east, never flat front
elevations):
- Radiating 3D ridge arêtes (craggy W/NW & E/NE flank crests, meandering forward-descending main watershed spine,
  and angled SE & SW buttress spurs) that form genuine bird's-eye ridge networks and diamond ground footprints
  with zero flat horizontal baseline or flaring ski-jump tails
- Clean faceted woodcut parallel diagonal hatching across every shadowed east/SE facet, with rich dark under-crest
  contrast, organically staggered foothill terminations, and crisp luminous NW faces
- Curved 3D foothill masks (`erase_soft_apron`) so overlapping peaks mask rear ridges cleanly while fading
  seamlessly into the terrain beneath
- Three distinct Mountain Brush pools: noble multi-ridged Alpine Mountains, serrated fractured Crags, and gentle
  rolling 3/4 bird's-eye Hills.
"""
import math

import numpy as np

from .ink import TAU, chaikin


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


def _build_3d_ridge(P, rng, p0, p1, n_pts=10, exp_range=(1.15, 1.44), shoulder=True,
                    crag=False, meander=0.11, step_boost=1.0):
    """Build one craggy 3D mountain ridge from (x0, y0, h0) down to (x1, y1, h1) and return (pts_3d, pts_2d)."""
    x0, y0, h0 = p0
    x1, y1, h1 = p1
    dx, dy = x1 - x0, y1 - y0
    dist = max(1e-6, math.hypot(dx, dy))
    nx, ny = -dy / dist, dx / dist
    exp_p = rng.uniform(*exp_range)
    m_amp = dist * rng.uniform(-meander, meander)
    m_phase = rng.uniform(0.0, math.pi)

    # 1-2 geological rock steps / craggy sub-shoulders so ridges read as carved mountain rock (not smooth prisms!)
    bumps = []
    if shoulder and h0 - h1 > 8.0:
        n_b = 2 if (crag or rng.random() < 0.62) else 1
        for bi in range(n_b):
            bf = rng.uniform(0.22, 0.52) if bi == 0 else rng.uniform(0.54, 0.78)
            bw = rng.uniform(0.08, 0.13) if crag else rng.uniform(0.10, 0.16)
            ba = (h0 - h1) * rng.uniform(0.08, 0.17 if crag else 0.14) * step_boost
            bumps.append((bf, bw, ba))

    pts3 = []
    ts = np.linspace(0.0, 1.0, n_pts)
    for i, t in enumerate(ts):
        lat = m_amp * math.sin(math.pi * t + m_phase * 0.35) * math.sin(math.pi * t)
        if 0 < i < n_pts - 1:
            lat += rng.normal(0.0, dist * (0.042 if crag else 0.028)) * math.sin(math.pi * t)
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
            # Crisp eroded rock notches along both alpine and crag ridges!
            wh += rng.normal(0.0, (h0 - h1) * (0.048 if crag else 0.028)) * (1.0 - t) ** 0.55
            wh = max(h1, min(h0 * (1.02 if crag else 1.005), wh))
        pts3.append((float(wx), float(wy), float(wh)))

    pts2 = [P(wx, wy, wh) for wx, wy, wh in pts3]
    # Keep alpine crests subtly angular & craggy (avoid over-smoothing into paper-tent curves!)
    return pts3, pts2


def _hatch_facet(pen, rng, facet_poly, clip_poly, angle_deg, spacing, w_base,
                 reach=(0.56, 0.94), crest_dark=True, crag=False, start_jitter=0.0):
    """Fill one shadowed east/SE mountain facet with bold, rhythmic parallel woodcut diagonal hatching.

    Strokes originate along the watershed ridge (left of `facet_poly`) and sweep down-right along the
    slope fall-line with organically staggered lower lengths so shadows breathe naturally into the valley
    instead of terminating on a sharp geometric line.
    """
    poly = np.asarray(facet_poly, float)
    if len(poly) < 3:
        return
    a = math.radians(angle_deg)
    d = np.array([math.cos(a), math.sin(a)])
    n = np.array([-d[1], d[0]])
    pn = poly @ n
    lo_n, hi_n = float(pn.min()), float(pn.max())
    if hi_n - lo_n < spacing * 0.8:
        return
    ymin, ymax = float(poly[:, 1].min()), float(poly[:, 1].max())
    yr = max(1.0, ymax - ymin)

    c = lo_n + spacing * rng.uniform(0.35, 0.75)
    idx = 0
    while c < hi_n - spacing * 0.25:
        xs = []
        m = len(poly)
        for i in range(m):
            p, q = poly[i], poly[(i + 1) % m]
            dp, dq = float(p @ n - c), float(q @ n - c)
            if (dp < 0.0) != (dq < 0.0):
                xs.append(float((p + (q - p) * (dp / (dp - dq))) @ d))
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            s0, s1 = xs[k], xs[k + 1]
            L = s1 - s0
            if L < 2.2:
                continue
            st_off = rng.uniform(0.0, min(start_jitter, L * 0.25)) if start_jitter > 0 else 0.0
            p_start = d * (s0 + st_off) + n * c
            fy = float(np.clip((p_start[1] - ymin) / yr, 0.0, 1.0))
            # Staggered organic foothill reach (eliminates rigid straight lower cutoff!)
            if idx % 3 == 0:
                cov = rng.uniform(max(0.72, reach[0]), reach[1])
            elif idx % 3 == 1:
                cov = rng.uniform(reach[0] * 0.65, reach[1] * 0.86)
            else:
                cov = rng.uniform(reach[0] * 0.36, reach[0] * 0.78)
            # Fade gently shorter toward the lower foothill skirt so shading dissipates cleanly into terrain
            cov *= (1.0 - 0.30 * (fy ** 1.35))
            rem = max(0.0, L - st_off)
            if rem * cov < 1.8:
                continue
            p_end = d * (s0 + st_off + rem * cov) + n * (c + rng.uniform(-0.24, 0.24))
            for sa, sb in _clip_line_poly(clip_poly, p_start, p_end):
                sl = float(np.hypot(*(sb - sa)))
                if sl < 1.8:
                    continue
                mid = (sa + sb) * 0.5 + np.array([-d[1], d[0]]) * rng.uniform(-0.35, 0.55)
                w_cur = w_base * (1.14 - 0.36 * fy) * rng.uniform(0.90, 1.12)
                pen.stroke([sa, mid, sb], w_cur, taper=(0.08, 0.88), smooth=1,
                           wobble=0.10 if crag else 0.08, pressure=0.10)
            # Crisp dark under-crest cast shadow tucked right beside the upper ridge spine
            if crest_dark and fy < 0.58 and L > 4.5 and (crag or idx % 2 == 0):
                sh_l = min(L * 0.42, rng.uniform(4.5, 11.5) * (1.12 - 0.58 * fy))
                q_end = d * (s0 + sh_l) + n * (c + spacing * 0.44)
                for sa, sb in _clip_line_poly(clip_poly, p_start + n * (spacing * 0.44), q_end):
                    if float(np.hypot(*(sb - sa))) > 1.6:
                        pen.stroke([sa, sb], w_base * (0.94 if crag else 0.82),
                                   taper=(0.05, 0.88), wobble=0.07)
        c += spacing * rng.uniform(0.88, 1.14)
        idx += 1


def _draw_peak_3d(pen, P, rng, wx, wy, rx, ry, h, style="alpine",
                  add_sub_peaks=True, scale_w=1.0):
    """Draw one true 3/4 bird's-eye 3D mountain massif or jagged crag spire projected through `P`.

    Every massif uses radiating 3D ridges (craggy W/NW & E/NE flank crests, meandering forward-descending
    watershed spine, and diagonal SE & SW buttress spurs) with woodcut diagonal east-facet hatching.
    """
    W0 = pen.width * scale_w
    crag = (style == "crag")

    sx = wx + rng.uniform(-0.14, 0.10) * rx
    sy = wy + rng.uniform(-0.08, 0.14) * ry
    summit_3d = (sx, sy, h)

    def ground_pt(ang_deg, r_scale=1.0):
        a = math.radians(ang_deg)
        return (wx + math.cos(a) * rx * r_scale, wy + math.sin(a) * ry * r_scale, 0.0)

    # Broad sweeping alpine flanks; steeper tighter crags
    ang_L = rng.uniform(176.0, 206.0)
    ang_R = rng.uniform(-16.0, 14.0)
    # Main front watershed spine drifts slightly SW or SE (avoids robotic straight-down tent corner!)
    ang_M = rng.uniform(-108.0, -76.0) if rng.random() < 0.72 else rng.uniform(-74.0, -56.0)

    foot_L = ground_pt(ang_L, rng.uniform(0.95, 1.10))
    foot_R = ground_pt(ang_R, rng.uniform(0.95, 1.10))
    foot_M = ground_pt(ang_M, rng.uniform(0.94, 1.12))

    exp_flank = (1.02, 1.30) if crag else (1.16, 1.46)
    exp_front = (1.00, 1.28) if crag else (1.12, 1.40)
    n_seg = 9 if crag else 10

    rL_3, rL_2 = _build_3d_ridge(P, rng, summit_3d, foot_L, n_pts=n_seg, exp_range=exp_flank,
                                 shoulder=True, crag=crag, meander=0.11)
    rR_3, rR_2 = _build_3d_ridge(P, rng, summit_3d, foot_R, n_pts=n_seg, exp_range=exp_flank,
                                 shoulder=True, crag=crag, meander=0.11)
    rM_3, rM_2 = _build_3d_ridge(P, rng, summit_3d, foot_M, n_pts=n_seg, exp_range=exp_front,
                                 shoulder=True, crag=crag, meander=0.14, step_boost=1.15)

    # Secondary SE front-right buttress spur (carves a crisp 2nd shadowed east facet!)
    has_SE = rng.random() < (0.90 if crag else 0.88)
    rSE_2 = None
    foot_SE_2 = None
    if has_SE:
        bi = int(rng.integers(1, max(2, len(rM_3) // 2)))
        b3 = rM_3[bi]
        ang_SE = rng.uniform(-46.0, -20.0)
        fSE = ground_pt(ang_SE, rng.uniform(0.88, 1.08))
        _, rSE_2 = _build_3d_ridge(P, rng, b3, fSE, n_pts=7, exp_range=(1.04, 1.34),
                                   shoulder=True, crag=crag, meander=0.10, step_boost=0.95)
        foot_SE_2 = rSE_2[-1]

    # Secondary SW front-left buttress spur — always descends steeply down-left (never a flat horizontal bar!)
    has_SW = rng.random() < (0.82 if crag else 0.82)
    rSW_2 = None
    foot_SW_2 = None
    if has_SW:
        src = rM_3 if rng.random() < 0.70 else rL_3
        bi = int(rng.integers(2, max(3, len(src) // 2)))
        b3 = src[bi]
        ang_SW = rng.uniform(-148.0, -122.0)
        fSW = ground_pt(ang_SW, rng.uniform(0.92, 1.12))
        _, rSW_2 = _build_3d_ridge(P, rng, b3, fSW, n_pts=7, exp_range=(1.04, 1.34),
                                   shoulder=True, crag=crag, meander=0.10, step_boost=0.90)
        # Reject only if projected slope would look nearly horizontal
        sdx = abs(rSW_2[-1][0] - rSW_2[0][0])
        sdy = rSW_2[-1][1] - rSW_2[0][1]
        if sdy > sdx * 0.35:
            foot_SW_2 = rSW_2[-1]
        else:
            rSW_2 = None

    # 2. Construct the 3D projected silhouette & curved front foothill apron
    crest = rL_2[::-1] + rR_2[1:]

    apron_anchors = [rL_2[-1]]
    if foot_SW_2 is not None and rL_2[-1][0] < foot_SW_2[0] < rM_2[-1][0]:
        apron_anchors.append((foot_SW_2[0], foot_SW_2[1] + 2.0))
    apron_anchors.append((rM_2[-1][0], rM_2[-1][1] + 2.5))
    if foot_SE_2 is not None and rM_2[-1][0] < foot_SE_2[0] < rR_2[-1][0]:
        apron_anchors.append((foot_SE_2[0], foot_SE_2[1] + 2.0))
    apron_anchors.append(rR_2[-1])
    apron_anchors.sort(key=lambda p: p[0])
    apron_smooth = list(chaikin(apron_anchors, 2))
    apron_smooth[0] = rL_2[-1]
    apron_smooth[-1] = rR_2[-1]

    poly = crest + apron_smooth[-2:0:-1]
    pen.cv.erase_soft_apron(poly, apron_smooth, feather=6.5 if crag else 7.5)

    # 3. Bold, unified parallel woodcut hatching across every shadowed east/SE facet!
    base_ang = rng.uniform(52.0, 62.0) if crag else rng.uniform(43.0, 52.0)
    sp_main = rng.uniform(1.85, 2.22) if crag else rng.uniform(2.05, 2.50)
    w_hatch = W0 * (0.64 if crag else 0.56)

    if rSE_2 is not None:
        b_pt = rSE_2[0]
        dists = [math.hypot(p[0] - b_pt[0], p[1] - b_pt[1]) for p in rM_2]
        b_idx = int(np.argmin(dists))
        inner_facet = rM_2[b_idx:] + rSE_2[::-1]
        if len(inner_facet) >= 3:
            _hatch_facet(pen, rng, inner_facet, poly, base_ang + rng.uniform(2.0, 6.0),
                         sp_main * 0.96, w_hatch * 1.04, reach=(0.64, 0.94),
                         crest_dark=True, crag=crag)
        outer_facet = rM_2[:b_idx + 1] + rSE_2[1:] + rR_2[::-1]
        if len(outer_facet) >= 3:
            _hatch_facet(pen, rng, outer_facet, poly, base_ang - rng.uniform(4.0, 9.0),
                         sp_main * 1.06, w_hatch * 0.90, reach=(0.56, 0.92),
                         crest_dark=True, crag=crag)
    else:
        east_facet = rM_2 + rR_2[::-1]
        if len(east_facet) >= 3:
            _hatch_facet(pen, rng, east_facet, poly, base_ang, sp_main, w_hatch,
                         reach=(0.60, 0.94), crest_dark=True, crag=crag)

    # Tucked shadow wedge immediately east of the protruding SW shoulder ridge (3/4 aerial dimensionality!)
    if rSW_2 is not None and len(rSW_2) >= 3:
        wedge_w = rx * (0.24 if crag else 0.22)
        right_edge = []
        n_sw = len(rSW_2)
        for i, (px, py) in enumerate(rSW_2):
            f = i / max(1, n_sw - 1)
            off = wedge_w * (0.45 + 0.55 * math.sin(math.pi * f)) * (1.0 - 0.35 * f)
            right_edge.append((px + off, py + off * 0.55))
        sw_facet = list(rSW_2) + right_edge[::-1]
        _hatch_facet(pen, rng, sw_facet, poly, base_ang + rng.uniform(4.0, 10.0),
                     sp_main * 1.10, w_hatch * 0.78, reach=(0.52, 0.90),
                     crest_dark=False, crag=crag)

    # Subtle rock-fracture ledges on crags only (lighter weight so they read as strata, never random scratches)
    if crag:
        e_min_x, e_max_x = rM_2[0][0], rR_2[-1][0]
        e_min_y, e_max_y = rM_2[0][1], rR_2[-1][1]
        for _ in range(int(rng.integers(2, 5))):
            fy = rng.uniform(0.20, 0.68)
            cy = e_min_y + (e_max_y - e_min_y) * fy
            cx0 = rng.uniform(e_min_x + 2.0, e_max_x - 10.0)
            fl = rng.uniform(7.0, 15.0)
            pa, pb = (cx0, cy), (cx0 + fl, cy + rng.uniform(-2.5, 4.0))
            for sa, sb in _clip_line_poly(poly, pa, pb):
                if float(np.hypot(*(sb - sa))) > 3.5:
                    pen.stroke([sa, sb], W0 * 0.44, taper=(0.20, 0.82), wobble=0.12)

    # 4. Crisp luminous sunlit WEST/NW face with delicate slope-following contour creases
    n_west = int(rng.integers(4, 7)) if crag else int(rng.integers(2, 5))
    w_ymin, w_ymax = rL_2[0][1], rM_2[-1][1]
    l_xs, l_ys = [p[0] for p in rL_2], [p[1] for p in rL_2]
    m_xs, m_ys = [p[0] for p in rM_2], [p[1] for p in rM_2]
    for _ in range(n_west):
        fy = rng.uniform(0.14, 0.66)
        wy_q = w_ymin + (w_ymax - w_ymin) * fy
        xl = float(np.interp(wy_q, l_ys, l_xs))
        xr = float(np.interp(wy_q, m_ys, m_xs))
        if xr - xl < 10.0:
            continue
        x0 = rng.uniform(xl + 4.0, xr - 3.5)
        wl = rng.uniform(7.0, 15.0) if crag else rng.uniform(9.0, 18.0)
        ang = rng.uniform(2.15, 2.55) if crag else rng.uniform(2.28, 2.64)
        p2 = (x0 + math.cos(ang) * wl, wy_q + math.sin(ang) * wl)
        for sa, sb in _clip_line_poly(poly, (x0, wy_q), p2):
            if float(np.hypot(*(sb - sa))) > 3.0:
                mid = (sa + sb) * 0.5 + np.array([0.8, 0.7])
                pen.stroke([sa, mid, sb], W0 * (0.46 if crag else 0.36),
                           taper=(0.28, 0.84), smooth=1, wobble=0.10)

    # 5. Hand-inked primary ridge spines & open fading flank contours (ZERO horizontal tails!)
    if rSW_2 is not None:
        pen.stroke(rSW_2, W0 * (0.74 if crag else 0.66), taper=(0.06, 0.78), wobble=0.14)
    if rSE_2 is not None:
        pen.stroke(rSE_2, W0 * (0.80 if crag else 0.72), taper=(0.06, 0.78), wobble=0.14)

    pen.stroke(rM_2, W0 * (0.90 if crag else 0.82), taper=(0.04, 0.82), wobble=0.16)
    pen.stroke(rL_2[::-1], W0 * (0.82 if crag else 0.74), taper=(0.56, 0.04), wobble=0.14)
    pen.stroke(rR_2, W0 * (0.94 if crag else 0.86), taper=(0.04, 0.58), wobble=0.16)

    # Flanking subsidiary shoulder foothill peak tucked into grand single massifs so every stamp reads richly!
    if add_sub_peaks:
        n_sub = 2 if (not crag and rng.random() < 0.48) else 1
        sides = [-1, 1] if n_sub == 2 else [int(rng.choice([-1, 1]))]
        subs = []
        for sgn in sides:
            fx = wx + sgn * rx * rng.uniform(0.46, 0.66)
            fy = wy - ry * rng.uniform(0.28, 0.54)
            subs.append((fx, fy, rx * rng.uniform(0.46, 0.60), ry * rng.uniform(0.46, 0.60),
                         h * rng.uniform(0.40, 0.54)))
        for fx, fy, srx, sry, sh in sorted(subs, key=lambda it: -P.depth(it[0], it[1])):
            _draw_peak_3d(pen, P, rng, fx, fy, srx, sry, sh, style=style,
                          add_sub_peaks=False, scale_w=scale_w * 0.86)


def _draw_hill_3d(pen, P, rng, wx, wy, rx, ry, h, scale_w=1.0):
    """Draw one gentle, softly rounded 3/4 bird's-eye rolling downland hill with sweeping SE hachures.

    Computes the exact projected left-to-right horizon span of the 3D elliptical dome so there are
    never vertical edge cutoffs, and feathers the rounded eastern shadow organically without hard seams.
    """
    W0 = pen.width * scale_w
    # Gentle orientation variation around the 3/4 E-W ridge axis (so dome always projects cleanly left->right)
    phi = math.radians(rng.uniform(-16.0, 18.0))
    cp, sp = math.cos(phi), math.sin(phi)

    def local_3d(u, v, z):
        return (wx + u * cp - v * sp, wy + u * sp + v * cp, z)

    # Sample the full 3D ground ellipse perimeter and find exact screen-space leftmost & rightmost points!
    n_ring = 64
    thetas = np.linspace(0.0, TAU, n_ring, endpoint=False)
    ring_3 = [local_3d(rx * math.cos(t), ry * math.sin(t), 0.0) for t in thetas]
    ring_2 = [P(*p) for p in ring_3]
    i_left = int(np.argmin([p[0] for p in ring_2]))
    i_right = int(np.argmax([p[0] for p in ring_2]))

    # Walk the front half of the ground ellipse (largest screen Y = closest to viewer) from i_left to i_right
    path_fwd = []
    idx = i_left
    while True:
        path_fwd.append(idx)
        if idx == i_right:
            break
        idx = (idx + 1) % n_ring
    path_bwd = []
    idx = i_left
    while True:
        path_bwd.append(idx)
        if idx == i_right:
            break
        idx = (idx - 1) % n_ring
    # Whichever arc has larger mean screen Y is the front (viewer-facing) ground skirt!
    y_fwd = float(np.mean([ring_2[i][1] for i in path_fwd]))
    y_bwd = float(np.mean([ring_2[i][1] for i in path_bwd]))
    front_arc = path_fwd if y_fwd >= y_bwd else path_bwd
    apron_2 = [ring_2[i] for i in front_arc]

    # Build the smooth asymmetric 3D hill brow running from exact left silhouette foot -> summit -> right foot!
    pL_3 = ring_3[i_left]
    pR_3 = ring_3[i_right]
    # Crown sits slightly off-center along the chord and gently offset north/south for organic 3/4 roundness
    pk_t = rng.uniform(0.36, 0.58)
    has_sub = rng.random() < 0.52
    sub_t = rng.uniform(0.68, 0.80) if pk_t < 0.48 else rng.uniform(0.20, 0.34)
    sub_a = h * rng.uniform(0.18, 0.30) if has_sub else 0.0
    brow_bend = rng.uniform(-0.14, 0.14) * ry

    n_c = 24
    ts = np.linspace(0.0, 1.0, n_c)
    crest_3 = []
    for t in ts:
        # Smooth rounded cosine hill dome profile (gentle, broad rolling hill!)
        u = (pk_t - t) / max(1e-6, pk_t) if t <= pk_t else (t - pk_t) / max(1e-6, 1.0 - pk_t)
        u = min(1.0, max(0.0, u))
        hz = h * (0.5 * (1.0 + math.cos(math.pi * (u ** 1.08)))) ** 0.94
        if has_sub:
            hz += sub_a * math.exp(-((t - sub_t) / 0.14) ** 2) * math.sin(math.pi * t)
        # Interpolate 3D ground position along the chord + slight curved bow away from viewer
        cx3 = pL_3[0] + (pR_3[0] - pL_3[0]) * t
        cy3 = pL_3[1] + (pR_3[1] - pL_3[1]) * t + brow_bend * math.sin(math.pi * t)
        crest_3.append((cx3, cy3, max(0.0, hz)))
    crest_2 = list(chaikin([P(*p) for p in crest_3], 1))
    crest_2[0] = ring_2[i_left]
    crest_2[-1] = ring_2[i_right]

    # Full closed hill dome mask with smooth feathered curved front apron!
    poly = crest_2 + apron_2[-2:0:-1]
    pen.cv.erase_soft_apron(poly, apron_2, feather=max(6.0, ry * 0.28))

    # Highest point along the projected crest
    pk_i = int(np.argmin([p[1] for p in crest_2]))

    # Soft, sweeping curved SE-shoulder hachures curving gently downward into the rounded eastern dome!
    east_crest = crest_2[pk_i:]
    n_ec = len(east_crest)
    for k in range(1, n_ec - 2, 2):
        frac = k / max(1, n_ec - 2)
        bx, by = east_crest[k]
        # Sweeping organic lengths: fullest across the broad mid-shoulder, tapering softly near crown & toe
        bell = math.sin(math.pi * min(1.0, max(0.08, frac))) ** 0.55
        base_len = (h * 0.72 + ry * 0.42) * bell * rng.uniform(0.68, 1.16)
        if (k // 2) % 2 == 0:
            base_len *= 1.15
        if base_len < 3.5:
            continue
        # Soft staggered start tucked just below the sunlit-to-shaded brow rollover
        st_drop = rng.uniform(1.0, 3.4) * (1.0 - 0.45 * frac)
        # Steeper downward roll-off (~62–76°) so hachures curve into the rounded hillside (not along the rim!)
        ang = math.radians(rng.uniform(62.0, 74.0) - 6.0 * frac)
        p0 = (bx + rng.uniform(-1.0, 1.0), by + st_drop)
        p2 = (p0[0] + math.cos(ang) * base_len, p0[1] + math.sin(ang) * base_len)
        for sa, sb in _clip_line_poly(poly, p0, p2):
            sl = float(np.hypot(*(sb - sa)))
            if sl < 2.2:
                continue
            # Gently outward-curving convex contour arc matching a rounded grassy hillside dome!
            mid = (sa + sb) * 0.5 + np.array([sl * 0.12, -sl * 0.05])
            pen.stroke([sa, mid, sb], W0 * (0.44 * (1.06 - 0.28 * frac)),
                       taper=(0.24, 0.88), smooth=1, wobble=0.08)

    # Airy second lower-flank soft hachure tier staggered gently down the SE belly
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

    # Occasional delicate sunlit-west grass contour sweep (subtle 3D volume on lit brow)
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

    # Soft, airy hilltop brow stroke (fades effortlessly at both ends — zero bubble outline!)
    pen.stroke(crest_2[:pk_i + 1], W0 * 0.66, taper=(0.82, 0.06), wobble=0.13)
    pen.stroke(crest_2[pk_i:], W0 * 0.76, taper=(0.06, 0.82), wobble=0.14)


# --------------------------------------------------------------------------- Wonderdraft Mountain Brush pool generators

def mountain(pen, P, rng):
    """Wonderdraft Mountain Brush pool: 3/4 bird's-eye multi-ridged alpine massifs & interlocking ridges."""
    r = rng.random()
    if r < 0.32:
        # Grand multi-ridged alpine massif with 1-2 tucked shoulder foothills
        _draw_peak_3d(pen, P, rng, 0.0, 0.0, rng.uniform(62, 80), rng.uniform(48, 62),
                      rng.uniform(52, 72), style="alpine", add_sub_peaks=True)
    elif r < 0.72:
        # Interlocking diagonal 2-peak alpine massif (staggered in true 3D ground depth `(x, y)`!)
        side = int(rng.choice([-1, 1]))
        rx1, ry1, h1 = rng.uniform(56, 70), rng.uniform(42, 54), rng.uniform(48, 66)
        rx2, ry2, h2 = rx1 * rng.uniform(0.74, 0.92), ry1 * rng.uniform(0.74, 0.92), h1 * rng.uniform(0.70, 0.92)
        peaks = [
            (side * rx1 * rng.uniform(0.46, 0.70), ry1 * rng.uniform(0.32, 0.60), rx2, ry2, h2, 0.92),
            (-side * rx1 * rng.uniform(0.18, 0.34), -ry1 * rng.uniform(0.14, 0.30), rx1, ry1, h1, 1.0),
        ]
        for wx, wy, rx, ry, h, sw in sorted(peaks, key=lambda it: -P.depth(it[0], it[1])):
            _draw_peak_3d(pen, P, rng, wx, wy, rx, ry, h, style="alpine", add_sub_peaks=False, scale_w=sw)
    else:
        # Compact 3/4 triangular 3-peak alpine cluster (interlocks organically into sprawling mountain ranges!)
        rx0, ry0, h0 = rng.uniform(50, 62), rng.uniform(38, 48), rng.uniform(46, 62)
        peaks = [
            (rng.uniform(-12, 12), ry0 * rng.uniform(0.40, 0.64), rx0 * 1.06, ry0 * 1.02, h0 * rng.uniform(1.02, 1.18), 0.95),
            (-rx0 * rng.uniform(0.52, 0.74), rng.uniform(-8, 12), rx0 * rng.uniform(0.76, 0.90),
             ry0 * rng.uniform(0.74, 0.88), h0 * rng.uniform(0.72, 0.90), 0.92),
            (rx0 * rng.uniform(0.36, 0.60), -ry0 * rng.uniform(0.26, 0.50), rx0 * rng.uniform(0.80, 0.94),
             ry0 * rng.uniform(0.76, 0.90), h0 * rng.uniform(0.70, 0.88), 1.0),
        ]
        if rng.random() < 0.5:
            peaks = [(-wx, wy, rx, ry, h, sw) for wx, wy, rx, ry, h, sw in peaks]
        for wx, wy, rx, ry, h, sw in sorted(peaks, key=lambda it: -P.depth(it[0], it[1])):
            _draw_peak_3d(pen, P, rng, wx, wy, rx, ry, h, style="alpine", add_sub_peaks=False, scale_w=sw)


def crag(pen, P, rng):
    """Wonderdraft Mountain Brush pool: steep serrated 3/4 bird's-eye granite spires & fractured crags."""
    r = rng.random()
    if r < 0.30:
        # Towering fractured granite peak with a flanking crag tooth
        _draw_peak_3d(pen, P, rng, 0.0, 0.0, rng.uniform(48, 62), rng.uniform(38, 50),
                      rng.uniform(72, 96), style="crag", add_sub_peaks=(rng.random() < 0.72))
    elif r < 0.70:
        # Staggered twin serrated crag massif in 3D ground depth
        side = int(rng.choice([-1, 1]))
        rx1, ry1, h1 = rng.uniform(44, 58), rng.uniform(36, 46), rng.uniform(66, 90)
        rx2, ry2, h2 = rx1 * rng.uniform(0.72, 0.90), ry1 * rng.uniform(0.72, 0.90), h1 * rng.uniform(0.70, 0.94)
        peaks = [
            (side * rx1 * rng.uniform(0.46, 0.66), ry1 * rng.uniform(0.32, 0.58), rx2, ry2, h2, 0.92),
            (-side * rx1 * rng.uniform(0.18, 0.34), -ry1 * rng.uniform(0.14, 0.30), rx1, ry1, h1, 1.0),
        ]
        for wx, wy, rx, ry, h, sw in sorted(peaks, key=lambda it: -P.depth(it[0], it[1])):
            _draw_peak_3d(pen, P, rng, wx, wy, rx, ry, h, style="crag", add_sub_peaks=False, scale_w=sw)
    else:
        # Jagged 3-tooth 3/4 crag cluster
        rx0, ry0, h0 = rng.uniform(42, 52), rng.uniform(32, 42), rng.uniform(60, 82)
        peaks = [
            (rng.uniform(-10, 10), ry0 * rng.uniform(0.40, 0.64), rx0 * 1.04, ry0 * 1.00, h0 * rng.uniform(1.04, 1.18), 0.95),
            (-rx0 * rng.uniform(0.50, 0.70), rng.uniform(-6, 10), rx0 * rng.uniform(0.74, 0.88),
             ry0 * rng.uniform(0.74, 0.88), h0 * rng.uniform(0.70, 0.90), 0.92),
            (rx0 * rng.uniform(0.36, 0.56), -ry0 * rng.uniform(0.26, 0.48), rx0 * rng.uniform(0.78, 0.94),
             ry0 * rng.uniform(0.76, 0.90), h0 * rng.uniform(0.68, 0.88), 1.0),
        ]
        if rng.random() < 0.5:
            peaks = [(-wx, wy, rx, ry, h, sw) for wx, wy, rx, ry, h, sw in peaks]
        for wx, wy, rx, ry, h, sw in sorted(peaks, key=lambda it: -P.depth(it[0], it[1])):
            _draw_peak_3d(pen, P, rng, wx, wy, rx, ry, h, style="crag", add_sub_peaks=False, scale_w=sw)


def hill(pen, P, rng):
    """Wonderdraft Mountain Brush pool: gentle 3/4 bird's-eye rolling hills & overlapping highland downs."""
    r = rng.random()
    if r < 0.34:
        # Single broad rolling downland hill
        _draw_hill_3d(pen, P, rng, 0.0, 0.0, rng.uniform(50, 68), rng.uniform(30, 42), rng.uniform(15, 24))
    elif r < 0.74:
        # Twin staggered 3D hillocks
        side = int(rng.choice([-1, 1]))
        rx1, ry1, h1 = rng.uniform(44, 58), rng.uniform(28, 38), rng.uniform(14, 22)
        rx2, ry2, h2 = rx1 * rng.uniform(0.78, 0.96), ry1 * rng.uniform(0.78, 0.96), h1 * rng.uniform(0.76, 0.96)
        hills = [
            (side * rx1 * rng.uniform(0.42, 0.64), ry1 * rng.uniform(0.48, 0.82), rx2, ry2, h2, 0.92),
            (-side * rx1 * rng.uniform(0.18, 0.34), -ry1 * rng.uniform(0.18, 0.38), rx1, ry1, h1, 1.0),
        ]
        for wx, wy, rx, ry, h, sw in sorted(hills, key=lambda it: -P.depth(it[0], it[1])):
            _draw_hill_3d(pen, P, rng, wx, wy, rx, ry, h, scale_w=sw)
    else:
        # Compact 3-hillock 3/4 rolling downs cluster
        rx0, ry0, h0 = rng.uniform(40, 52), rng.uniform(26, 35), rng.uniform(13, 21)
        hills = [
            (rng.uniform(-10, 10), ry0 * rng.uniform(0.58, 0.88), rx0 * 1.04, ry0 * 1.02, h0 * rng.uniform(0.95, 1.12), 0.92),
            (-rx0 * rng.uniform(0.48, 0.68), rng.uniform(-6, 10), rx0 * rng.uniform(0.78, 0.92),
             ry0 * rng.uniform(0.78, 0.92), h0 * rng.uniform(0.78, 0.96), 0.94),
            (rx0 * rng.uniform(0.36, 0.56), -ry0 * rng.uniform(0.35, 0.62), rx0 * rng.uniform(0.80, 0.96),
             ry0 * rng.uniform(0.80, 0.94), h0 * rng.uniform(0.76, 0.96), 1.0),
        ]
        if rng.random() < 0.5:
            hills = [(-wx, wy, rx, ry, h, sw) for wx, wy, rx, ry, h, sw in hills]
        for wx, wy, rx, ry, h, sw in sorted(hills, key=lambda it: -P.depth(it[0], it[1])):
            _draw_hill_3d(pen, P, rng, wx, wy, rx, ry, h, scale_w=sw)
