"""'Godly' ruins: temples, stone circles, colossi, a god's sword, obelisks."""
import math

import numpy as np

from .ink import TAU, chaikin, clip_halfplane, ellipse_pts
from .lairs import crack, hill_pts, rock
from .ruins import damage_profile, draw_wall
from .solids import box, broken_cylinder, cylinder, hull, lying_cylinder, pyramid


def _scatter(pen, P, rng, cx, cy, rx, ry, n=(3, 8), kinds=("tuft", "tuft", "rubble", "stone"),
             avoid_boxes=(), avoid_circles=()):
    """Scatter ground tufts, stones & rubble strictly on the visible front arc, outside all footprints."""
    items = []
    placed = list(avoid_circles)
    for _ in range(rng.integers(*n)):
        kind = str(rng.choice(kinds))
        rad = rng.uniform(5.0, 9.0) if kind == "rubble" else (rng.uniform(1.6, 3.0) if kind == "stone" else 3.0)
        pad = rad + 3.0
        for _ in range(24):
            t = rng.uniform(math.pi + 0.28, TAU - 0.28) + P.az
            d = rng.uniform(1.02, 1.44)
            x, y = cx + rx * d * math.cos(t), cy + ry * d * math.sin(t)
            if any(b[0] - pad < x < b[2] + pad and b[1] - pad < y < b[3] + pad for b in avoid_boxes):
                continue
            if any(math.hypot(x - ox, y - oy) < orc + pad for ox, oy, orc in placed):
                continue
            placed.append((x, y, rad * 0.85))
            items.append((P.depth(x, y), kind, (x, y, rad)))
            break
    for _, kind, (x, y, rad) in sorted(items, key=lambda it: -it[0]):
        sx, sy = P(x, y)
        if kind == "tuft":
            pen.tuft(sx, sy, rng.uniform(3.5, 6))
        elif kind == "rubble":
            pen.rubble(sx, sy, rad, rad * 0.45)
        else:
            pen.stone(sx, sy, rad)


def _fluted_drum(pen, P, xa, xb, y, r, h0=0.0):
    """Fallen fluted column drum lying east-west on the platform or ground."""
    W0 = pen.width
    lying_cylinder(pen, P, xa, xb, y, r, h0)
    # Add longitudinal classical column flutes along the visible side
    for f in (0.32, 0.58, 0.80):
        ang = -math.pi * 0.65 + f * math.pi * 0.85
        p0 = P(xa + 1.0, y + r * math.cos(ang), h0 + r + r * math.sin(ang))
        p1 = P(xb - 1.0, y + r * math.cos(ang), h0 + r + r * math.sin(ang))
        pen.stroke([p0, p1], W0 * 0.36, taper=(0.15, 0.15), wobble=0.05)


# --------------------------------------------------------------------------- temple

def _draw_stylobate(pen, P, rng, x0, y0, x1, y1, h0, steps, sh, pave=True):
    """Multi-tiered classical stepped marble stylobate with optional cracked flagstones; returns top z."""
    W0 = pen.width
    for i in range(steps):
        ins = i * 2.8
        box(pen, P, x0 + ins, y0 + ins, x1 - ins, y1 - ins, h0 + i * sh, h0 + (i + 1) * sh)
    top = h0 + steps * sh
    m = steps * 2.8 + 4.5
    if pave and (x1 - x0) > 2 * m + 16 and (y1 - y0) > 2 * m + 16:
        nx_p = max(3, int((x1 - x0 - 2 * m) / 24) + 1)
        ny_p = max(3, int((y1 - y0 - 2 * m) / 22) + 1)
        for gx_f in np.linspace(x0 + m, x1 - m, nx_p)[1:-1]:
            if rng.random() < 0.72:
                pen.stroke([P(gx_f, y0 + m, top), P(gx_f, y1 - m, top)], W0 * 0.30, taper=(0.25, 0.25), wobble=0.14)
        for gy_f in np.linspace(y0 + m, y1 - m, ny_p)[1:-1]:
            if rng.random() < 0.72:
                pen.stroke([P(x0 + m, gy_f, top), P(x1 - m, gy_f, top)], W0 * 0.30, taper=(0.25, 0.25), wobble=0.14)
        if rng.random() < 0.65:
            crack(pen, *P(rng.uniform(x0 + m, x1 - m), y0 + steps * 2.8, top),
                  math.pi * 0.5 + rng.normal(0, 0.3), rng.uniform(5, 9), W0 * 0.44, branch=False)
    return top


def _draw_stairs(pen, P, x0, y0, x1, y1, h0, h1, facing="south", n=4):
    """Broad marble flight of steps ascending from h0 to h1 along south (y0->y1) or east (x1->x0)."""
    if h1 <= h0 + 0.5:
        return
    dh = (h1 - h0) / n
    # Stack steps upward from lowest widest tread (k=0) to highest recessed tread (k=n-1)
    if facing == "south":
        dy = (y1 - y0) / n
        for k in range(n):
            ya = y0 + k * dy
            box(pen, P, x0, ya, x1, y1, h0 + k * dh, h0 + (k + 1) * dh, w=pen.width * 0.72)
    else:  # east-facing stairs descending toward +x (x0 is high/west, x1 is low/east)
        dx = (x1 - x0) / n
        for k in range(n):
            xb = x1 - k * dx
            box(pen, P, x0, y0, xb, y1, h0 + k * dh, h0 + (k + 1) * dh, w=pen.width * 0.72)


def _draw_doric_col(pen, P, rng, x, y, base_h, cr, state, h):
    """Fluted classical Doric column (intact with echinus+abacus capital, broken fluted shaft, or stump)."""
    W0 = pen.width
    cylinder(pen, P, x, y, cr * 1.36, base_h, base_h + 2.0)  # molded base
    if state == "intact":
        C = cylinder(pen, P, x, y, cr, base_h + 2.0, base_h + h - 1.5, top=False)
        for k in (-0.65, -0.22, 0.22, 0.65):
            t = P.az + 1.5 * math.pi + k
            pen.stroke([C(t, base_h + 3.0), C(t, base_h + h - 2.2)], W0 * 0.35, taper=(0.1, 0.1), wobble=0)
        cylinder(pen, P, x, y, cr * 1.28, base_h + h - 1.5, base_h + h)  # echinus
        box(pen, P, x - cr * 1.48, y - cr * 1.48, x + cr * 1.48, y + cr * 1.48, base_h + h, base_h + h + 2.6)  # abacus
        if rng.random() < 0.14:
            t0 = P.az + 1.5 * math.pi + rng.uniform(-0.4, 0.4)
            pts = [C(t0 + math.sin(k * 0.9) * 0.14, base_h + 2 + k * 3.5) for k in range(int(rng.integers(4, 8)))]
            pen.stroke(pts, W0 * 0.5, taper=(0.05, 0.65), smooth=1)
            for q in pts[1::2]:
                pen.dab(q[0] + rng.choice([-1.4, 1.4]), q[1], 1.3, w=W0 * 0.60)
    else:
        C = broken_cylinder(pen, P, x, y, cr, base_h + 2.0, base_h + h, jag=min(5.0, h * 0.42))
        if h > 8:
            for k in (-0.35, 0.25):
                t = P.az + 1.5 * math.pi + k
                pen.stroke([C(t, base_h + 3.0), C(t, base_h + h * 0.72)], W0 * 0.34, taper=(0.1, 0.35), wobble=0)


def _draw_entablature_x(pen, P, rng, xa, xb, y, hb, cr, ped=False, snapped=False):
    """East-west classical entablature architrave + molded cornice + optional south-facing gable pediment."""
    W0 = pen.width
    if snapped and not ped:
        xb = xa + (xb - xa) * rng.uniform(0.55, 0.85)
    box(pen, P, xa - cr * 1.45, y - cr * 1.3, xb + cr * 1.45, y + cr * 1.3, hb, hb + 4.4)
    box(pen, P, xa - cr * 1.65, y - cr * 1.5, xb + cr * 1.65, y + cr * 1.5, hb + 4.4, hb + 6.0)
    if ped:
        xm = (xa + xb) * 0.5
        hp = min(15.5, (xb - xa) * 0.22)
        p_south = [P(xa - cr * 1.5, y - cr * 1.4, hb + 6.0),
                   P(xb + cr * 1.5, y - cr * 1.4, hb + 6.0),
                   P(xm, y - cr * 1.4, hb + 6.0 + hp)]
        p_roof = [P(xm, y - cr * 1.4, hb + 6.0 + hp),
                  P(xb + cr * 1.5, y - cr * 1.4, hb + 6.0),
                  P(xb + cr * 1.5, y + cr * 1.6, hb + 6.0),
                  P(xm, y + cr * 1.6, hb + 6.0 + hp)]
        pen.cv.erase(p_roof)
        pen.hatch(p_roof, 76, 1.6, W0 * 0.55, coverage=(0.7, 1.0))
        pen.closed(p_roof, W0 * 0.8)
        pen.cv.erase(p_south)
        p_in = [P(xa + 1.8, y - cr * 1.4, hb + 7.2),
                P(xb - 1.8, y - cr * 1.4, hb + 7.2),
                P(xm, y - cr * 1.4, hb + 5.0 + hp * 0.85)]
        pen.hatch(p_in, 84, 2.2, W0 * 0.36, coverage=(0.4, 0.8))
        pen.closed(p_in, W0 * 0.52, wobble=0.1)
        pen.closed(p_south, W0 * 0.9, wobble=0.12)


def _draw_entablature_y(pen, P, rng, x, ya, yb, hb, cr, ped=False, roof_d=14.0):
    """North-south classical entablature + optional Parthenon-style east gable pediment with pitched tiled roof!"""
    W0 = pen.width
    box(pen, P, x - cr * 1.35, ya - cr * 1.45, x + cr * 1.35, yb + cr * 1.45, hb, hb + 4.4)
    box(pen, P, x - cr * 1.55, ya - cr * 1.65, x + cr * 1.55, yb + cr * 1.65, hb + 4.4, hb + 6.0)
    if ped:
        ym = (ya + yb) * 0.5
        hp = min(17.5, (yb - ya) * 0.24)
        xf = x + cr * 1.45
        xb = xf - roof_d
        # Back (north) pitched roof slope visible over the ridge at el=34°!
        p_roof_n = [P(xb, ym, hb + 6.0 + hp), P(xf, ym, hb + 6.0 + hp),
                    P(xf, yb + cr * 1.5, hb + 6.0), P(xb, yb + cr * 1.5, hb + 6.0)]
        pen.cv.erase(p_roof_n)
        pen.hatch(p_roof_n, 16, 2.2, W0 * 0.42, coverage=(0.35, 0.75))
        pen.closed(p_roof_n, W0 * 0.80, wobble=0.10)
        # Lit south-facing pitched roof slope extending west behind the east gable
        p_roof_s = [P(xb, ya - cr * 1.5, hb + 6.0), P(xf, ya - cr * 1.5, hb + 6.0),
                    P(xf, ym, hb + 6.0 + hp), P(xb, ym, hb + 6.0 + hp)]
        pen.cv.erase(p_roof_s)
        # Classical ribbed roof tile lines running down the south slope
        n_ribs = max(3, int(roof_d / 3.8))
        for rx in np.linspace(xb + 2.0, xf - 2.0, n_ribs):
            pen.stroke([P(rx, ya - cr * 1.3, hb + 6.4), P(rx, ym, hb + 5.8 + hp)],
                       W0 * 0.42, taper=(0.15, 0.15), wobble=0.08)
        pen.closed(p_roof_s, W0 * 0.85, wobble=0.12)
        # Shadowed east-facing triangular tympanum gable
        p_east = [P(xf, ya - cr * 1.5, hb + 6.0), P(xf, yb + cr * 1.5, hb + 6.0), P(xf, ym, hb + 6.0 + hp)]
        pen.cv.erase(p_east)
        pen.hatch(p_east, 82, 1.85, W0 * 0.52, coverage=(0.65, 1.0))
        p_in = [P(xf, ya + 2.0, hb + 7.4), P(xf, yb - 2.0, hb + 7.4), P(xf, ym, hb + 4.8 + hp * 0.86)]
        pen.hatch(p_in, 82, 1.45, W0 * 0.58, coverage=(0.75, 1.0))
        pen.closed(p_in, W0 * 0.52, wobble=0.08)
        pen.closed(p_east, W0 * 0.92, wobble=0.12)


def _draw_altar(pen, P, rng, cx, cy, base_h, scale=1.0):
    """Stepped carved marble sanctuary altar / cult statue pedestal with sacred glyph."""
    w1, d1 = 10.5 * scale, 6.8 * scale
    w2, d2 = 7.8 * scale, 4.8 * scale
    box(pen, P, cx - w1, cy - d1, cx + w1, cy + d1, base_h, base_h + 3.0 * scale)
    box(pen, P, cx - w2, cy - d2, cx + w2, cy + d2, base_h + 3.0 * scale, base_h + 8.8 * scale)
    _glyph(pen, *P(cx, cy - d2, base_h + 5.8 * scale), 3.3 * scale, str(rng.choice(GLYPHS)))


def _draw_cella_block(pen, P, rng, x0, y0, x1, y1, base_h, H_wall, sev, rooms=2,
                      door_south=True, door_east=True, altar_room=0, inner_cols=False, slice_axis="x"):
    """Multi-room ruined classical ashlar cella/naos (strictly back-to-front; slices west->east or north->south)."""
    W, D = x1 - x0, y1 - y0
    if W < 22 or D < 18:
        return
    t = 3.2

    def Q(x, y, h=0.0):
        return P(x, y, base_h + h)

    # Keep interior sanctuary walls slightly breached so rooms, doorways & altars read unmistakably from 3/4 view!
    sev_in = min(0.70, max(0.24, sev * 0.85))
    sev_front = min(0.82, max(0.38, sev * 1.05))
    p_back = damage_profile(rng, W, H_wall, 0, False, sev_in)
    p_left = damage_profile(rng, D, H_wall, 0, False, sev_in)

    # 1. Back (north) & Left (west) outer cella walls (inner faces visible)
    draw_wall(pen, Q, (x0, y1 - t), (x1, y1 - t), p_back, (0, t), shade="light")
    draw_wall(pen, Q, (x0 + t, y0), (x0 + t, y1), p_left, (-t, 0), shade="dark")

    # 2. Slice interior into 1..3 chambers (either west->east OR north->south, strictly back-to-front!)
    if slice_axis == "y":
        ys = np.linspace(y1, y0, rooms + 1)  # north (far) -> south (near)
        for r_idx in range(rooms):
            yb_r, ya_r = float(ys[r_idx]), float(ys[r_idx + 1])
            rmx, rmy = (x0 + x1) * 0.5, (ya_r + yb_r) * 0.5
            rd = yb_r - ya_r
            if r_idx == altar_room:
                sc = min(1.15, max(0.68, min(W, rd) / 34.0))
                if inner_cols and W > 34 and rd > 32:
                    _draw_doric_col(pen, P, rng, rmx - W * 0.22, rmy + rd * 0.20, base_h, 2.5,
                                    "broken" if rng.random() < 0.6 else "intact", H_wall * rng.uniform(0.45, 0.90))
                _draw_altar(pen, P, rng, rmx, rmy, base_h, scale=sc)
                if inner_cols and W > 34 and rd > 32:
                    _draw_doric_col(pen, P, rng, rmx + W * 0.22, rmy - rd * 0.20, base_h, 2.5,
                                    "broken", H_wall * rng.uniform(0.35, 0.72))
            else:
                if W > 22 and rd > 22 and rng.random() < 0.62:
                    _draw_altar(pen, P, rng, rmx, rmy, base_h, scale=0.62)
                elif rng.random() < 0.75:
                    sx, sy = P(rmx, rmy, base_h)
                    pen.rubble(sx, sy, rng.uniform(6, 11), rng.uniform(3, 5))
            # East-west interior cross-wall separating this northern chamber from the next southern chamber
            if r_idx < rooms - 1:
                ym = ya_r
                dh_p = H_wall * rng.uniform(0.76, 0.94)
                wp_len = W - 2 * t
                wm = wp_len * 0.5
                dw = min(9.5, wp_len * 0.28)
                dh_door = min(dh_p * 0.62, 13.0)
                p_part = damage_profile(rng, wp_len, dh_p, 0, False, sev_in * 0.85, protect=(wm - dw * 1.1, wm + dw * 1.1))
                draw_wall(pen, Q, (x0 + t, ym - t * 0.5), (x1 - t, ym - t * 0.5), p_part, (0, t),
                          shade=None, openings=[(wm - dw * 0.5, wm + dw * 0.5, 0.0, dh_door)])
        xs = np.array([x0, x1])
    else:
        xs = np.linspace(x0, x1, rooms + 1)
        for r_idx in range(rooms):
            ra, rb = float(xs[r_idx]), float(xs[r_idx + 1])
            rmx, rmy = (ra + rb) * 0.5, (y0 + y1) * 0.5
            rw = rb - ra
            if r_idx == altar_room:
                sc = min(1.15, max(0.68, min(rw, D) / 34.0))
                if inner_cols and rw > 34 and D > 32:
                    for ic_y in (rmy + D * 0.23, rmy - D * 0.23):
                        if ic_y > rmy:
                            _draw_doric_col(pen, P, rng, rmx - rw * 0.22, ic_y, base_h, 2.5,
                                            "broken" if rng.random() < 0.6 else "intact", H_wall * rng.uniform(0.45, 0.92))
                _draw_altar(pen, P, rng, rmx, rmy, base_h, scale=sc)
                if inner_cols and rw > 34 and D > 32:
                    _draw_doric_col(pen, P, rng, rmx + rw * 0.22, rmy - D * 0.22, base_h, 2.5,
                                    "broken", H_wall * rng.uniform(0.35, 0.72))
            else:
                if rw > 22 and D > 22 and rng.random() < 0.62:
                    _draw_altar(pen, P, rng, rmx, rmy, base_h, scale=0.62)
                elif rng.random() < 0.75:
                    sx, sy = P(rmx, rmy, base_h)
                    pen.rubble(sx, sy, rng.uniform(6, 11), rng.uniform(3, 5))

            if r_idx < rooms - 1:
                xm = rb
                dh_p = H_wall * rng.uniform(0.80, 0.98)
                dp_len = D - 2 * t
                dm = dp_len * 0.5
                dw = min(8.5, dp_len * 0.28)
                dh_door = min(dh_p * 0.62, 13.0)
                p_part = damage_profile(rng, dp_len, dh_p, 0, False, sev_in * 0.80, protect=(dm - dw * 1.1, dm + dw * 1.1))
                draw_wall(pen, Q, (xm + t * 0.5, y0 + t), (xm + t * 0.5, y1 - t), p_part, (-t, 0),
                          shade="dark", openings=[(dm - dw * 0.5, dm + dw * 0.5, 0.0, dh_door)])

    # 3. East (right) outer cella wall
    ops_e = []
    prot_e = None
    if door_east and D > 24:
        dw = min(10.5, D * 0.26)
        dh_d = min(H_wall * 0.70, 16.5)
        ops_e = [(D * 0.5 - dw * 0.5, D * 0.5 + dw * 0.5, 0.0, dh_d)]
        prot_e = (D * 0.5 - dw * 0.8, D * 0.5 + dw * 0.8) if rng.random() < 0.75 else None
    p_right = damage_profile(rng, D, H_wall, 0, False, sev_front * 0.9, protect=prot_e)
    draw_wall(pen, Q, (x1, y0), (x1, y1), p_right, (-t, 0), shade="dark", openings=ops_e)

    # 4. South (front) outer cella wall (lowered/breached so interior chambers & partitions stay visible!)
    ops_s = []
    if door_south and W > 28:
        if slice_axis == "y":
            dw = min(10.5, W * 0.28)
            ops_s.append((W * 0.5 - dw * 0.5, W * 0.5 + dw * 0.5, 0.0, min(H_wall * 0.68, 16.0)))
        else:
            for r_idx in range(rooms):
                if r_idx == 0 or (rooms >= 2 and rng.random() < 0.55):
                    ra, rb = float(xs[r_idx]), float(xs[r_idx + 1])
                    sm = (ra + rb) * 0.5 - x0
                    dw = min(9.5, (rb - ra) * 0.28)
                    ops_s.append((sm - dw * 0.5, sm + dw * 0.5, 0.0, min(H_wall * 0.65, 15.0)))
    p_front = damage_profile(rng, W, H_wall * rng.uniform(0.82, 0.98), 0, False, sev_front)
    draw_wall(pen, Q, (x0, y0), (x1, y0), p_front, (0, t), shade=None, openings=ops_s)


def _draw_peristyle_hall(pen, P, rng, x0, y0, x1, y1, base_h, cr, Hc, sev,
                         rooms=2, ped_mode="auto", open_front=False, inner_cols=False):
    """Full classical Greek temple hall on top of an already-drawn stylobate: outer peristyle + multi-room cella."""
    W, D = x1 - x0, y1 - y0
    m = 5.5
    nx = max(3, int((W - 2 * m) / rng.uniform(14.5, 18.0)))
    ny = max(2, int((D - 2 * m) / rng.uniform(15.0, 18.5)))
    gx = np.linspace(x0 + m, x1 - m, nx + 1)
    gy = np.linspace(y0 + m, y1 - m, ny + 1)
    dx_bay = (gx[-1] - gx[0]) / nx
    dy_bay = (gy[-1] - gy[0]) / ny

    if ped_mode == "auto":
        ped_mode = str(rng.choice(["east", "south", "north", "none"], p=[0.36, 0.28, 0.16, 0.20]))

    # Choose preserved intact portico span to support a classical gable pediment
    portico_span = None  # (side, idx0, idx1)
    if ped_mode == "east" and ny >= 2:
        portico_span = ("east", 0, ny)
    elif ped_mode in ("south", "north") and nx >= 3:
        if D > W * 1.05:
            # Deep N-S temple: full-width south or north gable pediment across the entire front facade!
            portico_span = (ped_mode, 0, nx)
        else:
            plen = min(nx, int(rng.integers(3, min(5, nx) + 1)))
            p0 = int(rng.integers(0, nx - plen + 1))
            portico_span = (ped_mode, p0, p0 + plen)

    cols = {}
    for i, x in enumerate(gx):
        for j, y in enumerate(gy):
            if 0 < i < nx and 0 < j < ny:
                continue
            if open_front and j == 0 and 0 < i < nx:
                continue
            forced = False
            if portico_span is not None:
                pside, a0, a1 = portico_span
                if pside == "east" and i == nx and a0 <= j <= a1:
                    forced = True
                elif pside == "south" and j == 0 and a0 <= i <= a1:
                    forced = True
                elif pside == "north" and j == ny and a0 <= i <= a1:
                    forced = True
            r = rng.random()
            state = "intact" if (forced or r > sev) else ("broken" if r > sev * 0.44 else "stump")
            h = Hc if state == "intact" else (Hc * rng.uniform(0.24, 0.76) if state == "broken" else rng.uniform(2.5, 6.5))
            cols[(i, j)] = (float(x), float(y), state, float(h))

    hb = base_h + Hc + 2.6

    def _run_beams_x(row_j, side_name):
        i = 0
        while i < nx:
            a, b = cols.get((i, row_j)), cols.get((i + 1, row_j))
            in_p = (portico_span is not None and portico_span[0] == side_name and portico_span[1] <= i < portico_span[2])
            if a and b and a[2] == "intact" and b[2] == "intact" and (in_p or rng.random() < 0.78):
                j_end = i + 1
                while j_end < nx and (j_end + 1, row_j) in cols and cols[(j_end + 1, row_j)][2] == "intact" and (
                    (portico_span is not None and portico_span[0] == side_name and j_end < portico_span[2]) or rng.random() < 0.75
                ):
                    j_end += 1
                # Avoid corner double-beam clash if an east pediment owns the east corner
                if ped_mode == "east" and j_end == nx and (j_end - i) >= 2:
                    j_end -= 1
                if j_end > i:
                    is_ped = (j_end - i >= 2) and (in_p or (ped_mode == side_name and rng.random() < 0.6))
                    _draw_entablature_x(pen, P, rng, gx[i], gx[j_end], gy[row_j], hb, cr,
                                        ped=is_ped, snapped=(not is_ped and rng.random() < 0.25))
                i = max(i + 1, j_end)
            else:
                i += 1

    # STRICT BACK-TO-FRONT DRAW ORDER:
    # 1. Back (north) colonnade + north architrave/pediment
    for i in range(nx + 1):
        if (i, ny) in cols:
            x, y, st, h = cols[(i, ny)]
            _draw_doric_col(pen, P, rng, x, y, base_h, cr, st, h)
    _run_beams_x(ny, "north")

    # 2. West outer colonnade (north to south)
    for j in range(ny - 1, 0, -1):
        if (0, j) in cols:
            x, y, st, h = cols[(0, j)]
            _draw_doric_col(pen, P, rng, x, y, base_h, cr, st, h)

    # 3. Inner Multi-Room Walled Cella (inset by ~1 column bay inside the outer peristyle!)
    cx0 = gx[0] + max(10.5, dx_bay * 0.85)
    cx1 = gx[-1] - max(10.5, dx_bay * 0.85)
    cy0 = gy[0] + max(10.0, dy_bay * 0.78)
    cy1 = gy[-1] - max(10.0, dy_bay * 0.78)
    if rooms > 0 and (cx1 - cx0) >= 24 and (cy1 - cy0) >= 22:
        s_axis = "y" if (cy1 - cy0) > (cx1 - cx0) * 1.15 else "x"
        span_main = (cy1 - cy0) if s_axis == "y" else (cx1 - cx0)
        actual_rooms = rooms if span_main >= rooms * 24 else max(1, int(span_main / 24))
        H_wall = Hc * rng.uniform(0.62, 0.82)
        altar_room = 0 if actual_rooms == 1 else (1 if actual_rooms == 3 else int(rng.choice([0, 1])))
        _draw_cella_block(pen, P, rng, cx0, cy0, cx1, cy1, base_h, H_wall, sev,
                          rooms=actual_rooms, door_south=(ped_mode != "east" or s_axis == "y"),
                          door_east=(s_axis != "y"), altar_room=altar_room, inner_cols=inner_cols,
                          slice_axis=s_axis)
    else:
        _draw_altar(pen, P, rng, (x0 + x1) * 0.5, (y0 + y1) * 0.5, base_h, scale=1.0)

    # 4. East outer colonnade (north to south, middle columns)
    for j in range(ny - 1, 0, -1):
        if (nx, j) in cols:
            x, y, st, h = cols[(nx, j)]
            _draw_doric_col(pen, P, rng, x, y, base_h, cr, st, h)

    # 5. Front (south) colonnade (west to east) + south architrave/pediment
    for i in range(nx + 1):
        if (i, 0) in cols:
            x, y, st, h = cols[(i, 0)]
            _draw_doric_col(pen, P, rng, x, y, base_h, cr, st, h)
    _run_beams_x(0, "south")

    # 6. East-end Parthenon architrave & triangular pediment spanning the east colonnade!
    if ped_mode == "east" and all((nx, j) in cols and cols[(nx, j)][2] == "intact" for j in range(ny + 1)):
        _draw_entablature_y(pen, P, rng, gx[nx], gy[0], gy[ny], hb, cr, ped=True,
                            roof_d=min(20.0, dx_bay * 1.05))
    else:
        # Occasional short intact east architrave beam between adjacent intact east columns
        j = 0
        while j < ny:
            a, b = cols.get((nx, j)), cols.get((nx, j + 1))
            if a and b and a[2] == "intact" and b[2] == "intact" and rng.random() < 0.55:
                j_end = j + 1
                while j_end < ny and cols.get((nx, j_end + 1), (0, 0, "", 0))[2] == "intact" and rng.random() < 0.65:
                    j_end += 1
                _draw_entablature_y(pen, P, rng, gx[nx], gy[j], gy[j_end], hb, cr, ped=False)
                j = j_end
            else:
                j += 1


def _draw_ring_beam(pen, P, xa, ya, xb, yb, hb, cr):
    """Curved-segment classical stone architrave bridging two adjacent columns around a circular Tholos."""
    W0 = pen.width
    a, b = np.array([xa, ya], float), np.array([xb, yb], float)
    u = b - a
    L = float(np.hypot(*u))
    if L < 2.0:
        return
    u /= L
    n_out = np.array([-u[1], u[0]])
    if float(n_out @ ((a + b) * 0.5)) < 0:
        n_out = -n_out
    hw = cr * 1.28
    p0, p1 = a - u * (cr * 0.8), b + u * (cr * 0.8)
    # 3D stone lintel prism: outer visible vertical face + top cornice slab
    h_top = hb + 5.6
    out_face = [P(*(p0 + n_out * hw), hb), P(*(p1 + n_out * hw), hb),
                P(*(p1 + n_out * hw), h_top), P(*(p0 + n_out * hw), h_top)]
    in_face = [P(*(p0 - n_out * hw), hb), P(*(p1 - n_out * hw), hb),
               P(*(p1 - n_out * hw), h_top), P(*(p0 - n_out * hw), h_top)]
    top_face = [P(*(p0 + n_out * hw), h_top), P(*(p1 + n_out * hw), h_top),
                P(*(p1 - n_out * hw), h_top), P(*(p0 - n_out * hw), h_top)]
    # Check whether outer face or inner face faces the south-east viewer (+sin az, -cos az)
    v_xy = np.array([P.sa, -P.ca])
    vis_face = out_face if float(n_out @ v_xy) >= 0 else in_face
    vis_n = n_out if float(n_out @ v_xy) >= 0 else -n_out
    pen.cv.erase(vis_face)
    if vis_n[0] > 0.35:
        pen.hatch(vis_face, 82, 1.7, W0 * 0.55, coverage=(0.7, 1.0))
    pen.closed(vis_face, W0 * 0.78, wobble=0.10)
    # Cornice step line
    pen.stroke([P(*(p0 + vis_n * hw), hb + 4.1), P(*(p1 + vis_n * hw), hb + 4.1)], W0 * 0.42, taper=(0.05, 0.05))
    pen.cv.erase(top_face)
    pen.closed(top_face, W0 * 0.78, wobble=0.10)


def _draw_tholos_rotunda(pen, P, rng, cr, Hc, sev):
    """Ancient Greek Circular Tholos / Rotunda Sanctuary: stepped round stylobate, ring colonnade & inner cella drum."""
    W0 = pen.width
    R = rng.uniform(62, 74)
    steps, sh = 3, rng.uniform(2.8, 3.3)
    for i in range(steps):
        cylinder(pen, P, 0.0, 0.0, R - i * 3.4, i * sh, (i + 1) * sh)
    top = steps * sh
    # Broad south marble stairs ascending onto the circular stylobate
    _draw_stairs(pen, P, -18.0, -R - 3.5, 18.0, -R + steps * 3.4, 0.0, top, facing="south", n=3)
    # Subtle radial marble floor paving lines
    r_pave = R - steps * 3.4 - 2.0
    for t in np.linspace(0, TAU, 12, endpoint=False):
        if rng.random() < 0.72:
            pen.stroke([P(math.cos(t) * r_pave * 0.45, math.sin(t) * r_pave * 0.45, top),
                        P(math.cos(t) * r_pave, math.sin(t) * r_pave, top)], W0 * 0.30, taper=(0.3, 0.3))

    # Outer ring of 10-14 fluted Doric columns around rc
    n_cols = int(rng.choice([10, 12, 14]))
    rc = R - steps * 3.4 - 6.8
    cols = []
    ang0 = -math.pi * 0.5 + math.pi / n_cols  # leaves a clear framed axial doorway facing south!
    for i in range(n_cols):
        t = ang0 + i * (TAU / n_cols)
        x, y = rc * math.cos(t), rc * math.sin(t)
        r_v = rng.random()
        # Preserve a dramatic intact colonnade arc on the rear/flanks so surviving entablature crowns the rotunda!
        in_arc = (0 <= i <= n_cols // 2) and rng.random() < 0.55
        st = "intact" if (in_arc or r_v > sev * 0.88) else ("broken" if r_v > sev * 0.42 else "stump")
        h = Hc if st == "intact" else (Hc * rng.uniform(0.28, 0.76) if st == "broken" else rng.uniform(3.0, 7.0))
        cols.append((i, t, x, y, st, h))

    # Inner circular ruined cella drum parameters
    ri = R * 0.48
    t_wall = 3.2
    H_wall = Hc * rng.uniform(0.58, 0.74)
    n_d = 72
    th_d = np.linspace(0, TAU, n_d, endpoint=False)
    phi = rng.uniform(0, TAU)
    wave = 0.5 + 0.5 * np.cos(th_d - phi)
    hs = np.clip(np.round((H_wall * (0.36 + 0.64 * (wave ** 1.1))) / 3.4) * 3.4, 5.0, H_wall)

    def hat(tt):
        return float(np.interp(tt % TAU, np.append(th_d, TAU), np.append(hs, hs[0])))

    def C_drum(tt, rad, hh):
        return P(rad * math.cos(tt), rad * math.sin(tt), top + hh)

    a = P.az
    hb = top + Hc + 2.6

    # 1. Back outer columns + curved back architrave spans
    back_cols = [c for c in cols if P.depth(c[2], c[3]) >= 0.0]
    for _, _, x, y, st, h in sorted(back_cols, key=lambda c: -P.depth(c[2], c[3])):
        _draw_doric_col(pen, P, rng, x, y, top, cr * 0.94, st, h)
    for i in range(n_cols):
        c0, c1 = cols[i], cols[(i + 1) % n_cols]
        if (P.depth(c0[2], c0[3]) + P.depth(c1[2], c1[3])) * 0.5 >= 0.0:
            if c0[4] == "intact" and c1[4] == "intact" and rng.random() < 0.80:
                _draw_ring_beam(pen, P, c0[2], c0[3], c1[2], c1[3], hb, cr * 0.94)

    # 2. Back half of inner circular ruined cella drum
    back_t = np.linspace(a, a + math.pi, 36)
    ob_top = [C_drum(tt, ri, hat(tt)) for tt in back_t]
    ib_top = [C_drum(tt, ri - t_wall, hat(tt)) for tt in back_t]
    ib_fl = [C_drum(tt, ri - t_wall, 0.0) for tt in back_t[::-1]]
    pen.cv.erase(ib_top + ib_fl)
    pen.cv.erase(ob_top + ib_top[::-1])
    pen.hatch(ib_top + ib_fl, 80, 1.85, W0 * 0.62, coverage=(0.68, 1.0))
    pen.stroke(ob_top, W0 * 0.78, taper=(0.02, 0.02), wobble=0.14)
    pen.stroke(ib_top, W0 * 0.72, taper=(0.02, 0.02), wobble=0.14)

    # 3. Grand central sanctuary altar + subtle interior rubble inside the rotunda
    _draw_altar(pen, P, rng, 0.0, 0.0, top, scale=1.18)
    if rng.random() < 0.65:
        pen.rubble(*P(rng.uniform(-ri * 0.35, ri * 0.35), rng.uniform(-ri * 0.35, 0.0), top), 7.5, 3.2)

    # 4. Front half of inner circular ruined cella drum (with arched south sanctuary entrance!)
    front_t = np.linspace(a + math.pi, a + TAU, 36)
    of_top = [C_drum(tt, ri, hat(tt)) for tt in front_t]
    if_top = [C_drum(tt, ri - t_wall, hat(tt)) for tt in front_t]
    f_body = [C_drum(tt, ri, 0.0) for tt in front_t] + of_top[::-1]
    pen.cv.erase(of_top + if_top[::-1])
    pen.cv.erase(f_body)
    pen.stroke(if_top, W0 * 0.68, taper=(0.02, 0.02), wobble=0.12)
    shade_t = np.linspace(a + TAU - 1.05, a + TAU, 18)
    pen.hatch([C_drum(tt, ri, 0.0) for tt in shade_t] + [C_drum(tt, ri, hat(tt)) for tt in shade_t[::-1]],
              82, 1.7, W0 * 0.68, coverage=(0.72, 1.0))
    # South sanctuary doorway cut into front cella drum
    td_c = -math.pi * 0.5
    dt_d = 5.0 / ri
    if hat(td_c) > 13.0:
        dh_d = min(hat(td_c) - 3.0, 15.0)
        pen.blot([C_drum(td_c - dt_d, ri, 0.0), C_drum(td_c + dt_d, ri, 0.0),
                  C_drum(td_c + dt_d, ri, dh_d), C_drum(td_c - dt_d, ri, dh_d)], jitter=0.06)
        pen.stroke([C_drum(td_c - dt_d * 1.25, ri, dh_d + 0.8), C_drum(td_c + dt_d * 1.25, ri, dh_d + 0.8)],
                   W0 * 0.60, taper=(0.05, 0.05))
    pen.stroke(of_top, W0 * 0.96, taper=(0.02, 0.02), wobble=0.16)
    pen.stroke([C_drum(tt, ri, 0.0) for tt in front_t], W0 * 1.02, taper=(0.04, 0.04))

    # 5. Front outer columns + curved front architrave spans
    front_cols = [c for c in cols if P.depth(c[2], c[3]) < 0.0]
    for _, _, x, y, st, h in sorted(front_cols, key=lambda c: -P.depth(c[2], c[3])):
        _draw_doric_col(pen, P, rng, x, y, top, cr * 0.94, st, h)
    for i in range(n_cols):
        c0, c1 = cols[i], cols[(i + 1) % n_cols]
        if (P.depth(c0[2], c0[3]) + P.depth(c1[2], c1[3])) * 0.5 < 0.0:
            if c0[4] == "intact" and c1[4] == "intact" and rng.random() < 0.78:
                _draw_ring_beam(pen, P, c0[2], c0[3], c1[2], c1[3], hb, cr * 0.94)

    # Fallen fluted drums & scattered marble rubble strictly outside the circular base & south stairs
    drum_boxes = []
    for _ in range(rng.integers(2, 4)):
        dl = rng.uniform(10, 15)
        for _ in range(18):
            fx = rng.uniform(-R * 0.80, R * 0.65)
            fy = -R - rng.uniform(11, 20)
            if -22 < fx + dl and fx < 22 and fy > -R - 14:
                continue
            dbox = (fx - 2, fy - cr - 2, fx + dl + 2, fy + cr + 2)
            if any(not (dbox[2] < b[0] or dbox[0] > b[2] or dbox[3] < b[1] or dbox[1] > b[3]) for b in drum_boxes):
                continue
            drum_boxes.append(dbox)
            _fluted_drum(pen, P, fx, fx + dl, fy, cr * 0.92, 0.0)
            break
    _scatter(pen, P, rng, 0, 0, R * 1.12, R * 1.12, n=(4, 9),
             avoid_boxes=[(-22, -R - 10, 22, -R + 4)] + drum_boxes,
             avoid_circles=[(0, 0, R + 5.0)])


def temple(pen, P, rng):
    """Ancient Greek temple ruins in varied footprints: wide E-W Parthenon, deep N-S hall, circular Tholos, terraced forecourt & L-wings."""
    archetype = str(rng.choice(
        ["grand_ew", "grand_ns", "tholos_rotunda", "forecourt_complex", "multi_wing"],
        p=[0.22, 0.22, 0.22, 0.18, 0.16],
    ))
    cr = rng.uniform(3.3, 4.1)
    Hc = rng.uniform(36, 46)
    sev = rng.uniform(0.24, 0.66)

    if archetype == "tholos_rotunda":
        _draw_tholos_rotunda(pen, P, rng, cr, Hc, sev)
        return

    if archetype in ("grand_ew", "grand_ns"):
        # Monumental Parthenon-scale Greek temple: either wide E-W OR deep N-S (opposite orientation)!
        if archetype == "grand_ew":
            W = rng.uniform(164, 200)
            D = rng.uniform(98, 122)
        else:
            W = rng.uniform(94, 118)
            D = rng.uniform(158, 194)
        x0, y0 = -W * 0.5, -D * 0.5
        steps, sh = 3, rng.uniform(2.8, 3.4)
        top = _draw_stylobate(pen, P, rng, x0, y0, x0 + W, y0 + D, 0.0, steps, sh)
        # Grand marble entry staircase: south on deep N-S temples, south or east on wide E-W temples
        stair_east = (archetype == "grand_ew") and (rng.random() < 0.52)
        if stair_east:
            _draw_stairs(pen, P, x0 + W - steps * 2.8, -16, x0 + W + 4.5, 16, 0.0, top, facing="east", n=4)
        else:
            _draw_stairs(pen, P, -22, y0 - 4.5, 22, y0 + steps * 2.8, 0.0, top, facing="south", n=4)
        ins = steps * 2.8
        rooms = int(rng.choice([2, 2, 3]))
        if archetype == "grand_ns":
            ped_mode = str(rng.choice(["south", "south", "north"]))
        else:
            ped_mode = "east" if stair_east else str(rng.choice(["south", "east", "north"]))
        _draw_peristyle_hall(pen, P, rng, x0 + ins, y0 + ins, x0 + W - ins, y0 + D - ins, top, cr, Hc, sev,
                             rooms=rooms, ped_mode=ped_mode, inner_cols=True)
        avoid = [(x0 - 2, y0 - 2, x0 + W + 2, y0 + D + 2)]
        if stair_east:
            avoid.append((x0 + W - steps * 2.8, -18, x0 + W + 7, 18))
        else:
            avoid.append((-24, y0 - 7, 24, y0 + steps * 2.8))
        for _ in range(rng.integers(2, 4)):
            dl = rng.uniform(10, 16)
            for _ in range(18):
                fx = rng.uniform(x0 + 6, x0 + W - 20)
                fy = y0 - rng.uniform(11, 20)
                dbox = (fx - 2, fy - cr - 2, fx + dl + 2, fy + cr + 2)
                if any(not (dbox[2] < b[0] or dbox[0] > b[2] or dbox[3] < b[1] or dbox[1] > b[3]) for b in avoid):
                    continue
                avoid.append(dbox)
                _fluted_drum(pen, P, fx, fx + dl, fy, cr, 0.0)
                break
        _scatter(pen, P, rng, 0, 0, W * 0.56, D * 0.56, n=(4, 9), avoid_boxes=avoid)

    elif archetype == "forecourt_complex":
        # Processional North-South Two-Terrace Sanctuary: deep N-S footprint climbing from south forecourt to raised temple!
        W = rng.uniform(122, 152)
        D_main = rng.uniform(82, 100)
        D_court = rng.uniform(68, 84)
        x0 = -W * 0.5
        y_join = -4.0
        top_hi = _draw_stylobate(pen, P, rng, x0, y_join, x0 + W, y_join + D_main, 0.0, 4, 3.0)
        Wc = W * rng.uniform(0.92, 1.04)
        xc0 = -Wc * 0.5
        top_lo = _draw_stylobate(pen, P, rng, xc0, y_join - D_court, xc0 + Wc, y_join + 2.0, 0.0, 2, 2.8)
        _draw_stairs(pen, P, -20, y_join - 9.5, 20, y_join + 8.5, top_lo, top_hi, facing="south", n=4)
        _draw_stairs(pen, P, -18, y_join - D_court - 3.5, 18, y_join - D_court + 5.6, 0.0, top_lo, facing="south", n=3)

        ins_hi = 4 * 2.8
        _draw_peristyle_hall(pen, P, rng, x0 + ins_hi, y_join + ins_hi * 0.6, x0 + W - ins_hi, y_join + D_main - ins_hi,
                             top_hi, cr * 0.95, Hc * 0.96, sev, rooms=2,
                             ped_mode=str(rng.choice(["south", "south", "east"])), inner_cols=False)

        ins_lo = 2 * 2.8 + 4.5
        c_xa, c_xb = xc0 + ins_lo, xc0 + Wc - ins_lo
        c_ya, c_yb = y_join - D_court + ins_lo, y_join - 8.5
        Hc_low = Hc * 0.72
        has_treasury = rng.random() < 0.60
        if has_treasury:
            tw = min(42.0, (c_xb - c_xa) * 0.34)
            _draw_cella_block(pen, P, rng, c_xa, c_ya + 4.0, c_xa + tw, c_yb - 2.0, top_lo,
                              Hc_low * 0.80, sev, rooms=1, door_south=False, door_east=True, altar_room=0)
        else:
            for y_c in np.linspace(c_yb - 4, c_ya + 4, 3):
                st = "intact" if rng.random() > sev else "broken"
                _draw_doric_col(pen, P, rng, c_xa + 3.0, y_c, top_lo, cr * 0.88, st,
                                Hc_low if st == "intact" else Hc_low * rng.uniform(0.3, 0.7))
        alt_x = (c_xa + (44.0 if has_treasury else 10.0) + c_xb) * 0.5
        _draw_altar(pen, P, rng, alt_x, (c_ya + c_yb) * 0.5 - 2.0, top_lo, scale=1.15)
        e_ys = np.linspace(c_yb - 4, c_ya + 4, 3)
        e_sts = ["intact" if rng.random() > sev * 0.85 else "broken" for _ in e_ys]
        for y_c, st in zip(e_ys, e_sts):
            _draw_doric_col(pen, P, rng, c_xb - 3.0, y_c, top_lo, cr * 0.88, st,
                            Hc_low if st == "intact" else Hc_low * rng.uniform(0.3, 0.75))
        if all(st == "intact" for st in e_sts):
            _draw_entablature_y(pen, P, rng, c_xb - 3.0, e_ys[-1], e_ys[0], top_lo + Hc_low + 2.6, cr * 0.88, ped=False)
        for fx_c in (c_xa + 3.0, -22.0, 22.0, c_xb - 3.0):
            if has_treasury and fx_c < c_xa + 40.0:
                continue
            st = "intact" if rng.random() > sev else "broken"
            _draw_doric_col(pen, P, rng, fx_c, c_ya, top_lo, cr * 0.88, st,
                            Hc_low if st == "intact" else Hc_low * rng.uniform(0.25, 0.72))

        avoid = [
            (x0 - 2, y_join - 2, x0 + W + 2, y_join + D_main + 2),
            (xc0 - 2, y_join - D_court - 2, xc0 + Wc + 2, y_join + 4),
            (-20, y_join - D_court - 6, 20, y_join - D_court + 6),
        ]
        for _ in range(rng.integers(1, 3)):
            dl = rng.uniform(9, 15)
            for _ in range(18):
                fx = rng.uniform(xc0 + 6, xc0 + Wc - 20)
                fy = y_join - D_court - rng.uniform(11, 19)
                dbox = (fx - 2, fy - cr - 2, fx + dl + 2, fy + cr + 2)
                if any(not (dbox[2] < b[0] or dbox[0] > b[2] or dbox[3] < b[1] or dbox[1] > b[3]) for b in avoid):
                    continue
                avoid.append(dbox)
                _fluted_drum(pen, P, fx, fx + dl, fy, cr * 0.9, 0.0)
                break
        _scatter(pen, P, rng, 0, 0, W * 0.58, (D_main + D_court) * 0.55, n=(4, 9), avoid_boxes=avoid)

    else:
        # L-Shaped Multi-Wing Acropolis Sanctuary (balanced square/asymmetric bounding box!)
        ns_main = rng.random() < 0.45
        if ns_main:
            # Tall N-S Main Hall on the west + Projecting East Shrine Wing!
            W1, D1 = rng.uniform(86, 104), rng.uniform(146, 174)
            W2, D2 = rng.uniform(56, 72), rng.uniform(66, 82)
            mx0, my0 = -W1 * 0.5 - W2 * 0.25, -D1 * 0.5
            mx1, my1 = mx0 + W1, my0 + D1
            wx0, wx1 = mx1 - 6.0, mx1 + W2
            wy0, wy1 = my0 + 8.0, my0 + 8.0 + D2
            steps, sh = 3, rng.uniform(2.7, 3.2)
            top1 = _draw_stylobate(pen, P, rng, mx0, my0, mx1, my1, 0.0, steps, sh)
            top2 = _draw_stylobate(pen, P, rng, wx0, wy0, wx1, wy1, 0.0, 2, sh)
            _draw_stairs(pen, P, (mx0 + mx1) * 0.5 - 16, my0 - 4.0, (mx0 + mx1) * 0.5 + 16, my0 + steps * 2.8,
                         0.0, top1, facing="south", n=3)
            ins1 = steps * 2.8
            _draw_peristyle_hall(pen, P, rng, mx0 + ins1, my0 + ins1, mx1 - ins1, my1 - ins1, top1, cr, Hc, sev,
                                 rooms=2, ped_mode="south", inner_cols=True)
            ins2 = 2 * 2.8 + 3.5
            Hc2, cr2 = Hc * rng.uniform(0.76, 0.88), cr * 0.90
            _draw_cella_block(pen, P, rng, wx0 + 8.0, wy0 + ins2 + 12.0, wx1 - ins2 - 4.0, wy1 - ins2, top2,
                              Hc2 * 0.82, sev * 0.9, rooms=1, door_south=True, door_east=True, altar_room=0)
            pxs = np.linspace(wx0 + 10.0, wx1 - ins2, 4)
            p_sts = []
            keep_p = rng.random() < 0.72
            for px in pxs:
                st = "intact" if (keep_p or rng.random() > sev) else "broken"
                p_sts.append(st)
                _draw_doric_col(pen, P, rng, float(px), wy0 + ins2, top2, cr2, st,
                                Hc2 if st == "intact" else Hc2 * rng.uniform(0.3, 0.75))
            if all(st == "intact" for st in p_sts):
                _draw_entablature_x(pen, P, rng, float(pxs[0]), float(pxs[-1]), wy0 + ins2,
                                    top2 + Hc2 + 2.6, cr2, ped=(rng.random() < 0.75))
            avoid = [
                (mx0 - 2, my0 - 2, mx1 + 2, my1 + 2),
                (wx0 - 2, wy0 - 2, wx1 + 2, wy1 + 2),
                ((mx0 + mx1) * 0.5 - 18, my0 - 6, (mx0 + mx1) * 0.5 + 18, my0 + steps * 2.8),
            ]
            for _ in range(rng.integers(2, 4)):
                dl = rng.uniform(9, 14)
                for _ in range(18):
                    fx = rng.uniform(mx0 + 6, wx1 - 18)
                    fy = my0 - rng.uniform(11, 19)
                    dbox = (fx - 2, fy - cr - 2, fx + dl + 2, fy + cr + 2)
                    if any(not (dbox[2] < b[0] or dbox[0] > b[2] or dbox[3] < b[1] or dbox[1] > b[3]) for b in avoid):
                        continue
                    avoid.append(dbox)
                    _fluted_drum(pen, P, fx, fx + dl, fy, cr, 0.0)
                    break
            _scatter(pen, P, rng, (mx0 + wx1) * 0.5, (my0 + my1) * 0.5,
                     (W1 + W2) * 0.54, D1 * 0.56, n=(4, 9), avoid_boxes=avoid)
        else:
            # Classic E-W Main Hall + South-East or South-West Projecting Wing
            W1, D1 = rng.uniform(136, 166), rng.uniform(84, 102)
            W2, D2 = rng.uniform(66, 84), rng.uniform(56, 70)
            wing_side = str(rng.choice(["se", "sw"]))
            mx0, my0 = -W1 * 0.5, -14.0
            mx1, my1 = mx0 + W1, my0 + D1
            if wing_side == "se":
                wx0, wx1 = mx1 - W2, mx1 + 6.0
            else:
                wx0, wx1 = mx0 - 6.0, mx0 + W2
            wy0, wy1 = my0 - D2, my0 + 6.0
            steps, sh = 3, rng.uniform(2.7, 3.2)
            top1 = _draw_stylobate(pen, P, rng, mx0, my0, mx1, my1, 0.0, steps, sh)
            top2 = _draw_stylobate(pen, P, rng, wx0, wy0, wx1, wy1, 0.0, 2, sh)
            sx_mid = (mx0 + 28.0) if wing_side == "se" else (mx1 - 28.0)
            _draw_stairs(pen, P, sx_mid - 16, my0 - 4.0, sx_mid + 16, my0 + steps * 2.8, 0.0, top1, facing="south", n=3)
            ins1 = steps * 2.8
            _draw_peristyle_hall(pen, P, rng, mx0 + ins1, my0 + ins1, mx1 - ins1, my1 - ins1, top1, cr, Hc, sev,
                                 rooms=2, ped_mode=str(rng.choice(["east", "south", "north"])), inner_cols=True)
            ins2 = 2 * 2.8 + 3.5
            Hc2, cr2 = Hc * rng.uniform(0.76, 0.88), cr * 0.90
            ch_y0, ch_y1 = wy0 + ins2 + 16.0, wy1 - 5.0
            _draw_cella_block(pen, P, rng, wx0 + ins2 + 4.0, ch_y0, wx1 - ins2 - 4.0, ch_y1, top2,
                              Hc2 * 0.82, sev * 0.9, rooms=1, door_south=True, door_east=(wing_side == "se"), altar_room=0)
            pxs = np.linspace(wx0 + ins2, wx1 - ins2, 4)
            p_sts = []
            keep_portico = rng.random() < 0.72
            for px in pxs:
                st = "intact" if (keep_portico or rng.random() > sev) else "broken"
                p_sts.append(st)
                _draw_doric_col(pen, P, rng, float(px), wy0 + ins2, top2, cr2, st,
                                Hc2 if st == "intact" else Hc2 * rng.uniform(0.3, 0.75))
            if all(st == "intact" for st in p_sts):
                _draw_entablature_x(pen, P, rng, float(pxs[0]), float(pxs[-1]), wy0 + ins2,
                                    top2 + Hc2 + 2.6, cr2, ped=(rng.random() < 0.75))
            avoid = [
                (mx0 - 2, my0 - 2, mx1 + 2, my1 + 2),
                (wx0 - 2, wy0 - 2, wx1 + 2, wy1 + 2),
                (sx_mid - 18, my0 - 6, sx_mid + 18, my0 + steps * 2.8),
            ]
            for _ in range(rng.integers(2, 4)):
                dl = rng.uniform(9, 14)
                for _ in range(18):
                    fx = rng.uniform(mx0 + 6, mx1 - 20)
                    fy = wy0 - rng.uniform(11, 19)
                    dbox = (fx - 2, fy - cr - 2, fx + dl + 2, fy + cr + 2)
                    if any(not (dbox[2] < b[0] or dbox[0] > b[2] or dbox[3] < b[1] or dbox[1] > b[3]) for b in avoid):
                        continue
                    avoid.append(dbox)
                    _fluted_drum(pen, P, fx, fx + dl, fy, cr, 0.0)
                    break
            _scatter(pen, P, rng, (mx0 + mx1) * 0.5, (wy0 + my1) * 0.5,
                     W1 * 0.58, (D1 + D2) * 0.54, n=(4, 9), avoid_boxes=avoid)


# --------------------------------------------------------------------------- stone circle

def _monolith(pen, P, rng, x, y, w, d, h, lean=(0.0, 0.0), glyph=False):
    """Rough-hewn tapered ancient standing megalith with faceted stone ridge & weathered crown."""
    W0 = pen.width
    lx, ly = lean
    # Asymmetric rough-hewn top bevel + faceted side planes
    sl_l = h * rng.uniform(0.84, 1.0)
    sl_r = h * rng.uniform(0.82, 1.0)
    pk = h * rng.uniform(1.0, 1.10)
    tw_l = w * rng.uniform(0.32, 0.44)
    tw_r = w * rng.uniform(0.30, 0.42)
    td = d * rng.uniform(0.34, 0.46)

    # Front (south) face and shadowed right (east) face in 3/4 view
    south = [
        P(x - w * 0.5, y - d * 0.5, 0),
        P(x + w * 0.5, y - d * 0.5, 0),
        P(x + tw_r + lx, y - td + ly, sl_r),
        P(x + rng.uniform(-1.5, 1.5) + lx, y - td + ly, pk),
        P(x - tw_l + lx, y - td + ly, sl_l),
    ]
    east = [
        P(x + w * 0.5, y - d * 0.5, 0),
        P(x + w * 0.48, y + d * 0.5, 0),
        P(x + tw_r * 0.9 + lx, y + td + ly, sl_r * rng.uniform(0.92, 1.02)),
        P(x + tw_r + lx, y - td + ly, sl_r),
    ]
    top_cap = [
        south[4], south[3], south[2], east[2],
        P(x - tw_l * 0.9 + lx, y + td + ly, sl_l * 0.96),
    ]
    for f in (top_cap, east, south):
        pen.cv.erase(f)
    pen.hatch(east, 82, 1.55, W0 * 0.68, coverage=(0.75, 1.0))
    # Subtle facet shadow line & vertical fissure on lit front face
    pen.stroke([P(x + rng.normal(0, 1.2) + lx * 0.5, y - d * 0.48, h * 0.72),
                P(x + rng.normal(0, 1.5), y - d * 0.5, h * 0.18)],
               W0 * 0.46, taper=(0.3, 0.6), wobble=0.22)
    if glyph:
        _glyph(pen, *P(x + lx * 0.45, y - d * 0.48, h * 0.48), 4.0, str(rng.choice(GLYPHS)))
    if rng.random() < 0.45:
        crack(pen, *P(x + rng.uniform(-w * 0.2, w * 0.2), y - d * 0.5, h * rng.uniform(0.3, 0.7)),
              math.pi * 0.5 + rng.normal(0, 0.4), rng.uniform(5, 10), W0 * 0.44, branch=False)
    for f in (top_cap, east, south):
        pen.closed(f, W0 * 0.85, overshoot=0.25, wobble=0.22)


def stone_circle(pen, P, rng):
    """Ancient megalithic henge: rough-hewn standing monoliths, trilithon gates, ritual altar & worn ring track."""
    W0 = pen.width
    Rc = rng.uniform(56, 76)
    n = int(rng.integers(8, 12))

    # Faint worn ritual earth circle connecting the standing stones
    ring_ts = np.linspace(0, TAU, 48)
    for dr in (-4.5, 4.5):
        pts = [P((Rc + dr) * math.cos(t), (Rc + dr) * 0.88 * math.sin(t)) for t in ring_ts]
        i = 0
        while i < len(pts) - 3:
            j = min(len(pts), i + int(rng.integers(5, 12)))
            if rng.random() < 0.55:
                pen.stroke(pts[i:j], W0 * 0.36, taper=(0.35, 0.35), wobble=0.35)
            i = j + int(rng.integers(2, 5))

    items = []
    ts = np.linspace(0, TAU, n, endpoint=False) + rng.uniform(0, TAU)
    for idx, t in enumerate(ts):
        x, y = Rc * math.cos(t) + rng.normal(0, 2.2), Rc * 0.88 * math.sin(t) + rng.normal(0, 2.2)
        r = rng.random()
        if r < 0.12:
            continue
        items.append((P.depth(x, y), "fallen" if r < 0.28 else "stone", (x, y, idx)))

    # 1 to 2 grand trilithon archways framing the back / flanks of the henge
    n_tri = int(rng.choice([1, 1, 2]))
    for k in range(n_tri):
        t = P.az + math.pi * 0.5 + (rng.uniform(-0.5, 0.5) if n_tri == 1 else (-0.55 + k * 1.1) + rng.normal(0, 0.15))
        x, y = Rc * 0.58 * math.cos(t), Rc * 0.52 * math.sin(t)
        items.append((P.depth(x, y), "trilithon", (x, y, k)))

    items.append((P.depth(0, 0), "altar", (0, 0, 0)))

    occ = []
    for _, kind, (x, y, idx) in sorted(items, key=lambda it: -it[0]):
        if kind == "stone":
            w, d, h = rng.uniform(9, 13.5), rng.uniform(5.0, 7.2), rng.uniform(25, 42)
            _monolith(pen, P, rng, x, y, w, d, h,
                      lean=(rng.normal(0, 3.0), rng.normal(0, 1.5)), glyph=(rng.random() < 0.35))
            occ.append((x, y, max(w, d) * 0.72))
            if rng.random() < 0.55:
                pen.tuft(*P(x - w * 0.45, y - d * 0.5 - 2.2), rng.uniform(3.5, 5.5))
        elif kind == "fallen":
            L, w = rng.uniform(22, 32), rng.uniform(8, 11.5)
            h_f = rng.uniform(4.5, 6.5)
            box(pen, P, x - L / 2, y - w / 2, x + L / 2, y + w / 2, 0, h_f, inset=1.2)
            crack(pen, *P(x, y - w * 0.4, h_f * 0.8), 0.3, rng.uniform(6, 12), W0 * 0.42, branch=False)
            occ.append((x, y, L * 0.56))
            pen.tuft(*P(x - L * 0.35, y - w * 0.65 - 1.5), rng.uniform(3.5, 5.2))
        elif kind == "trilithon":
            h = rng.uniform(38, 50)
            for dx in (-11, 11):
                _monolith(pen, P, rng, x + dx, y, 8.8, 6.4, h, lean=(rng.normal(0, 1.0), 0))
                occ.append((x + dx, y, 9.0))
            if rng.random() < 0.88:
                # Heavy rough-hewn lintel capstone spanning the uprights
                box(pen, P, x - 18, y - 4.2, x + 18, y + 4.2, h - 0.5, h + 6.5, inset=0.8)
                crack(pen, *P(x + rng.uniform(-6, 6), y - 4.0, h + 5.0),
                      math.pi * 0.5 + rng.normal(0, 0.3), 5.5, W0 * 0.45, branch=False)
        else:
            # Carved stepped ritual altar with recessed rune/offering ring
            box(pen, P, x - 16, y - 9.5, x + 16, y + 9.5, 0, 2.8, inset=0.8)
            box(pen, P, x - 13, y - 7.2, x + 13, y + 7.2, 2.8, 7.6, inset=0.6)
            occ.append((x, y, 17.5))
            cx, cy = P(x, y, 7.6)
            pen.closed(ellipse_pts(cx, cy, 5.8, 5.8 * P.se, 16), W0 * 0.45, wobble=0.1)
            if rng.random() < 0.55:
                pen.blot(ellipse_pts(cx + rng.normal(0, 1.5), cy, rng.uniform(2.0, 3.6), rng.uniform(1.0, 1.8), 10), jitter=0.3)
    _scatter(pen, P, rng, 0, 0, Rc * 1.15, Rc * 1.05, n=(4, 8), kinds=("tuft", "tuft", "stone"), avoid_circles=occ)


# --------------------------------------------------------------------------- colossi

def _cracks_on(pen, P, C, rng, h_lo, h_hi, n):
    if h_hi <= h_lo:
        return
    for _ in range(n):
        t = P.az + 1.5 * math.pi + rng.uniform(-0.6, 0.6)
        x, y = C(t, rng.uniform(h_lo, h_hi))
        crack(pen, x, y, math.pi / 2 + rng.normal(0, 0.5), rng.uniform(6, 14), pen.width * 0.52, branch=False)


def _sculpted_leg(pen, P, rng, lx, ly, base, h, has_tunic=False):
    """Anatomically sculpted colossal stone leg: sandalled foot, muscular calf, carved knee/greave, snapped thigh."""
    W0 = pen.width
    a = P.az

    # 1. Sculpted classical sandalled foot (angled sole, heel, rounded forefoot, articulated toes & ankle straps)
    box(pen, P, lx - 5.2, ly - 3.5, lx + 5.2, ly + 5.2, base, base + 4.8, inset=0.4)
    cylinder(pen, P, lx, ly - 5.8, 5.3, base, base + 4.0)
    # Articulated carved stone toes along the front edge
    for k in range(4):
        tx = lx - 3.6 + k * 2.4
        pen.stroke([P(tx, ly - 10.8, base + 2.0), P(tx, ly - 9.0, base + 3.9)], W0 * 0.46, taper=(0, 0.2), wobble=0)
    # Crisscrossed sandal leather straps rising up onto the ankle
    pen.beam(P(lx - 5.3, ly - 3.2, base + 4.2), P(lx + 5.3, ly - 3.2, base + 4.2), 1.1)
    pen.stroke([P(lx - 4.4, ly - 1.5, base + 4.6), P(lx + 4.2, ly - 3.5, base + 8.5)], W0 * 0.6, taper=(0.05, 0.05))
    pen.stroke([P(lx + 4.4, ly - 1.5, base + 4.6), P(lx - 4.2, ly - 3.5, base + 8.5)], W0 * 0.6, taper=(0.05, 0.05))

    # 2. Continuous anatomically lofted leg (ankle -> muscular calf -> sculpted knee -> stout thigh!)
    h0 = base + 4.2
    h_knee = 34.0
    h_max = max(18.0, h)
    # Radius profile vs height above ankle: narrow ankle -> muscular calf swell -> knee -> stout thigh
    zs_k = np.array([0.0, 6.0, 16.0, 25.0, 32.0, 39.0, 54.0, 78.0])
    rx_k = np.array([4.3, 4.6,  6.5,  6.8,  5.9,  6.6,  8.0,  9.2])
    ry_k = np.array([4.6, 5.0,  7.0,  7.1,  6.2,  6.9,  8.3,  9.4])

    jag = rng.uniform(4.5, 7.5)
    ph = rng.uniform(0, TAU, 2)

    def hat(t):
        return h0 + h_max - jag * (0.48 + 0.30 * math.sin(2 * t + ph[0]) + 0.22 * math.sin(5 * t + ph[1]))

    def Lpt(t, hh):
        dz = max(0.0, hh - h0)
        rx = float(np.interp(dz, zs_k, rx_k))
        ry = float(np.interp(dz, zs_k, ry_k))
        # subtle calf/knee anatomical forward shift
        vy = 0.8 * math.sin(min(1.0, dz / 34.0) * math.pi)
        return P(lx + rx * math.cos(t), ly + vy + ry * math.sin(t), hh)

    t_l, t_r = a + math.pi, a + TAU
    front = np.linspace(t_l, t_r, 26)
    n_z = 18
    left_s = [Lpt(t_l, h0 + (hat(t_l) - h0) * f) for f in np.linspace(0, 1, n_z)]
    right_s = [Lpt(t_r, h0 + (hat(t_r) - h0) * f) for f in np.linspace(0, 1, n_z)]
    brk_f = [Lpt(t, hat(t)) for t in front]
    brk_ring = [Lpt(t, hat(t)) for t in np.linspace(0, TAU, 36, endpoint=False)]
    body = [Lpt(t, h0) for t in front] + right_s[1:] + brk_f[::-1][1:-1] + left_s[:0:-1]

    pen.cv.erase(body)
    pen.cv.erase(brk_ring)

    # Sculptural east-side shadow hatching following the muscular calf & thigh curve!
    sh_ts = np.linspace(t_r - 1.05, t_r, 10)
    sh_o = [Lpt(t_r, h0 + (hat(t_r) - h0) * f) for f in np.linspace(0, 1, n_z)]
    sh_i = [Lpt(t_r - 1.05, h0 + (hat(t_r - 1.05) - h0) * f) for f in np.linspace(0, 1, n_z)]
    pen.hatch(sh_o + sh_i[::-1], 83, 1.5, W0 * 0.65, coverage=(0.75, 1.0))

    # Carved shin / greave ridge & sculpted kneecap detail
    t_f = a + 1.5 * math.pi
    if h_max > 22:
        pen.stroke([Lpt(t_f - 0.15, h0 + 5), Lpt(t_f - 0.1, h0 + min(h_max - 4, h_knee - 3))],
                   W0 * 0.44, taper=(0.3, 0.35), wobble=0.08)
    if h_max > h_knee + 4:
        # Carved kneecap / bronze-stone greave collar!
        k_arc = [Lpt(t, h0 + h_knee - 1.5 + 1.2 * math.cos((t - t_f) * 2.2)) for t in np.linspace(t_f - 0.75, t_f + 0.75, 12)]
        pen.stroke(k_arc, W0 * 0.62, taper=(0.25, 0.25), wobble=0.08)
        k_top = [Lpt(t, h0 + h_knee + 3.2 - 0.8 * math.cos((t - t_f) * 2.2)) for t in np.linspace(t_f - 0.55, t_f + 0.55, 10)]
        pen.stroke(k_top, W0 * 0.46, taper=(0.3, 0.3), wobble=0.06)

    # Optional carved armor pteruges / draped stone tunic hem around tall broken thighs!
    if has_tunic and h_max > 48:
        ht = h0 + min(h_max - 9, 49.0)
        hem = [Lpt(t, ht + 1.2 * math.sin(6 * t)) for t in np.linspace(t_l, t_r, 20)]
        top_t = [Lpt(t, min(hat(t) - 1.5, ht + 12.0)) for t in np.linspace(t_l, t_r, 20)]
        tunic = hem + top_t[::-1]
        pen.cv.erase(tunic)
        pen.hatch([Lpt(t, ht) for t in sh_ts] + [Lpt(t, min(hat(t) - 1.5, ht + 12.0)) for t in sh_ts[::-1]],
                  82, 1.5, W0 * 0.6, coverage=(0.7, 1.0))
        for tf in np.linspace(t_l + 0.35, t_r - 0.35, 5):
            pen.stroke([Lpt(tf, ht + 0.5), Lpt(tf, min(hat(tf) - 2.0, ht + 10.5))], W0 * 0.45, taper=(0.15, 0.4))
        pen.stroke(hem, W0 * 0.9, taper=(0.05, 0.05), wobble=0.12)

    # Weathered stone cracks (tightly angled so they stay strictly within the carved leg width)
    for _ in range(int(rng.integers(1, 3))):
        tc = t_f + rng.uniform(-0.35, 0.30)
        hc = rng.uniform(h0 + 10, max(h0 + 11, hat(tc) - 5))
        crack(pen, *Lpt(tc, hc), math.pi * 0.5 + rng.normal(0, 0.22), rng.uniform(4.5, 8.5), W0 * 0.45, branch=False)

    # Solid fractured stone top cross-section & crisp outlines
    pen.hatch(brk_ring, rng.uniform(12, 32), 2.3, W0 * 0.42, coverage=(0.35, 0.75))
    for _ in range(rng.integers(2, 4)):
        tb = rng.uniform(0, TAU)
        pen.stroke([P(lx, ly, hat(tb)), Lpt(tb, hat(tb))], W0 * 0.48, taper=(0.25, 0.25))
    pen.closed(brk_ring, W0 * 0.85, wobble=0.14)
    pen.stroke([Lpt(t, h0) for t in front], W0 * 1.0, taper=(0.05, 0.05))
    pen.stroke(chaikin(left_s, 1), W0 * 1.0, taper=(0.04, 0.04), wobble=0.12)
    pen.stroke(chaikin(right_s, 1), W0 * 1.0, taper=(0.04, 0.04), wobble=0.12)


def _fallen_colossus_head(pen, rng, x, y, tilt):
    """Ozymandias shattered colossal stone visage (3/4 classical sculpt) half-sunk into earth below the plinth."""
    W0 = pen.width
    ca, sa = math.cos(tilt), math.sin(tilt)

    def R(px, py):
        return (x + px * ca - py * sa, y + px * sa + py * ca)

    # Chiseled angular classical head silhouette: broad crowned temple, tapering cheekbones, sculpted jaw/beard
    ctrl = [
        R(-5.8, -10.8), R(0.0, -12.4), R(6.4, -10.4),
        R(9.2, -5.2), R(8.8, 1.8), R(6.0, 7.6),
        R(1.8, 10.4), R(-3.2, 9.6), R(-7.0, 5.4),
        R(-8.4, -1.2), R(-8.0, -6.8),
    ]
    poly = chaikin(ctrl + [ctrl[0]], 1)[:-1]
    pen.cv.erase(poly)

    # Strong side-plane & underside sculptural hatching (gives 3D carved marble depth instead of flat disc!)
    sh = clip_halfplane(poly, np.array([math.cos(tilt + 0.35), math.sin(tilt + 0.35)]),
                        x * math.cos(tilt + 0.35) + y * math.sin(tilt + 0.35) + 1.8)
    if len(sh) >= 3:
        pen.hatch(sh, 76, 1.45, W0 * 0.56, coverage=(0.72, 1.0))

    # Carved diadem / crown band with slight 3D brow bevel
    pen.stroke([R(-7.4, -6.0), R(0.2, -7.2), R(7.8, -5.4)], W0 * 0.72, taper=(0.1, 0.1), smooth=1, wobble=0.06)
    pen.stroke([R(-7.0, -4.4), R(0.2, -5.6), R(7.4, -3.8)], W0 * 0.48, taper=(0.15, 0.15), smooth=1, wobble=0.06)

    # Sculpted brow ridge, deep-set shadowed eye planes (not cartoon dots), chiseled nose & mouth
    pen.stroke([R(-5.2, -2.2), R(-0.6, -1.4), R(4.8, -2.0)], W0 * 0.65, taper=(0.15, 0.15), wobble=0.05)
    pen.stroke([R(-4.4, -0.6), R(-2.0, -0.5)], W0 * 0.72, taper=(0.25, 0.25), wobble=0)
    pen.stroke([R(1.6, -0.5), R(4.0, -0.6)], W0 * 0.72, taper=(0.25, 0.25), wobble=0)
    pen.stroke([R(-0.2, -1.4), R(-0.6, 3.0), R(1.2, 3.3)], W0 * 0.54, taper=(0.12, 0.22), wobble=0.05)
    pen.stroke([R(-2.2, 5.2), R(0.2, 4.9), R(2.4, 5.3)], W0 * 0.48, taper=(0.2, 0.2), wobble=0.05)
    # Carved marble curls / fluted beard along jaw
    for fx in (-3.6, -1.2, 1.2, 3.6):
        pen.stroke([R(fx, 6.2), R(fx * 0.85, 8.8)], W0 * 0.40, taper=(0.15, 0.55), wobble=0.06)
    # Dramatic diagonal stone split crack across temple & cheek
    crack(pen, *R(4.5, -9.2), tilt + 1.85, 11.5, W0 * 0.5, branch=False)
    pen.closed(poly, W0 * 0.92, wobble=0.14)
    # Half-buried ground crease & tufts tucking the lower edge into the soil!
    pen.stroke([(x - 10.5, y + 7.2), (x - 2.0, y + 9.2), (x + 8.5, y + 7.8)], W0 * 0.55, taper=(0.35, 0.35), smooth=1)
    pen.tuft(x - 8.5, y + 8.5, rng.uniform(3.2, 4.8))
    pen.tuft(x + 7.5, y + 8.2, rng.uniform(3.2, 4.8))


def statue_legs(pen, P, rng):
    """The Colossus: muscular sculpted legs standing on a grand weathered plinth with fallen monument fragments."""
    W0 = pen.width
    # Grand 2- or 3-tier monumental pedestal with inscription glyph & cracks
    box(pen, P, -33, -21, 33, 21, 0, 6)
    box(pen, P, -28, -17, 28, 17, 6, 14.5, inset=0.6)
    if rng.random() < 0.6:
        for gx in (-11, 0, 11):
            _glyph(pen, *P(gx, -17, 10.2), 3.4, str(rng.choice(GLYPHS)))
    if rng.random() < 0.65:
        crack(pen, *P(rng.uniform(-16, 16), -17, 14.0), math.pi * 0.5 + rng.normal(0, 0.35), rng.uniform(6, 11), W0 * 0.5, branch=False)

    # Heroic contrapposto stance: one sculpted leg stepped slightly forward, optional tunic hem on taller breaks
    spread = rng.uniform(9.5, 13.0)
    hs = sorted([rng.uniform(32, 74), rng.uniform(22, 58)], reverse=rng.random() < 0.5)
    tunic = rng.random() < 0.55
    _sculpted_leg(pen, P, rng, -spread, 4.0, 14.5, hs[0], has_tunic=tunic)
    _sculpted_leg(pen, P, rng, spread, -3.0, 14.5, hs[1], has_tunic=tunic)

    # Varied foreground monument debris: half-buried Ozymandias visage (~58%) and/or shattered torso block (~55%)
    avoid_boxes = [(-35.0, -23.0, 35.0, 23.0)]
    avoid_circles = []
    head_side = -1 if rng.random() < 0.55 else 1
    hx_w, hy_w = head_side * rng.uniform(16, 36), -rng.uniform(37, 47)
    has_head = rng.random() < 0.58
    if has_head:
        hx, hy = P(hx_w, hy_w)
        _fallen_colossus_head(pen, rng, hx, hy - 4.0, tilt=head_side * rng.uniform(0.32, 0.92))
        avoid_circles.append((hx_w, hy_w, 15.0))
    if (not has_head) or rng.random() < 0.55:
        fx = -head_side * rng.uniform(16, 38)
        fy = -rng.uniform(37, 47)
        if rng.random() < 0.55:
            # Shattered carved marble torso/plinth block lying half-buried
            L, w, h = rng.uniform(18, 26), rng.uniform(10, 14), rng.uniform(7.5, 11.5)
            box(pen, P, fx - L * 0.5, fy - w * 0.5, fx + L * 0.5, fy + w * 0.5, 0, h, inset=1.0)
            crack(pen, *P(fx, fy - w * 0.45, h * 0.85), math.pi * 0.5 + rng.normal(0, 0.3), h * 0.7, W0 * 0.46, branch=False)
            avoid_boxes.append((fx - L * 0.5 - 2, fy - w * 0.5 - 2, fx + L * 0.5 + 2, fy + w * 0.5 + 2))
        else:
            dr = rng.uniform(5.2, 7.4)
            dl = rng.uniform(8, 16)
            _fluted_drum(pen, P, fx - 10, fx + dl, fy, dr)
            avoid_boxes.append((fx - 12, fy - dr - 2, fx + dl + 2, fy + dr + 2))
    _scatter(pen, P, rng, 0, -10, 48, 36, n=(4, 9), avoid_boxes=avoid_boxes, avoid_circles=avoid_circles)


def _oct_prism(pen, P, X, Y, r, h0, h1, inset=0.0, shade=True, top=True, w=None):
    """8-sided stone pedestal tier aligned with the 3/4 view."""
    W0 = pen.width
    w = w or W0 * 0.85
    ang0 = math.pi / 8
    ts = [ang0 + k * TAU / 8 for k in range(8)]
    bot = [P(X + r * math.cos(t), Y + r * math.sin(t), h0) for t in ts]
    rt = max(1.0, r - inset)
    topf = [P(X + rt * math.cos(t), Y + rt * math.sin(t), h1) for t in ts]
    cam = (P.sa, -P.ca)
    vis = []
    for i in range(8):
        j = (i + 1) % 8
        tm = ts[i] + TAU / 16
        nx, ny = math.cos(tm), math.sin(tm)
        if nx * cam[0] + ny * cam[1] > 0.02:
            face = [bot[i], bot[j], topf[j], topf[i]]
            vis.append((nx * 0.92 + ny * 0.38, face))
    for _, f in vis:
        pen.cv.erase(f)
    if top:
        pen.cv.erase(topf)
    if shade:
        for eastness, f in vis:
            if eastness > 0.45:
                sp = 1.5 if eastness > 0.8 else 2.2
                pen.hatch(f, 82, sp, W0 * (0.7 if eastness > 0.8 else 0.55), coverage=(0.7, 1.0))
    for _, f in vis:
        pen.closed(f, w, overshoot=0.3, wobble=0.2)
    if top:
        pen.closed(topf, w, overshoot=0.3, wobble=0.2)
    return topf


def _statue_plinth(pen, P, rng, kind, pw, b0):
    """Draw a weathered square, octagonal, or round stepped pedestal; returns top height."""
    W0 = pen.width
    h1 = b0 * rng.uniform(0.34, 0.44)
    pw2 = pw * rng.uniform(0.78, 0.85)
    if kind == "square":
        box(pen, P, -pw, -pw, pw, pw, 0, h1)
        if rng.random() < 0.35:  # three-step grand pedestal
            hm = h1 + (b0 - h1) * 0.42
            pm = (pw + pw2) * 0.5
            box(pen, P, -pm, -pm, pm, pm, h1, hm)
            box(pen, P, -pw2, -pw2, pw2, pw2, hm, b0, inset=rng.uniform(0.0, 0.8))
        else:
            box(pen, P, -pw2, -pw2, pw2, pw2, h1, b0, inset=rng.uniform(0.0, 1.0))
        if rng.random() < 0.45:  # carved inscription/glyph on the south face
            mid_h = h1 + (b0 - h1) * 0.52
            if rng.random() < 0.55:
                _glyph(pen, *P(0, -pw2 + 0.3, mid_h), 4.0, str(rng.choice(GLYPHS)))
            else:
                for gx in (-pw2 * 0.38, 0.0, pw2 * 0.38):
                    _glyph(pen, *P(gx, -pw2 + 0.3, mid_h), 3.0, str(rng.choice(GLYPHS)))
        if rng.random() < 0.55:  # weathered stone crack across the front tier
            cx = rng.uniform(-pw2 * 0.6, pw2 * 0.6)
            x, y = P(cx, -pw2, b0 - 0.5)
            crack(pen, x, y, math.pi / 2 + rng.normal(0, 0.35), rng.uniform(5, 11), W0 * 0.5, branch=False)
    elif kind == "octagonal":
        _oct_prism(pen, P, 0, 0, pw * 1.1, 0, h1, inset=0.5)
        _oct_prism(pen, P, 0, 0, pw2 * 1.1, h1, b0, inset=rng.uniform(0.5, 1.4))
        if rng.random() < 0.5:
            x, y = P(rng.uniform(-pw2 * 0.4, pw2 * 0.4), -pw2, b0 - 0.5)
            crack(pen, x, y, math.pi / 2 + rng.normal(0, 0.35), rng.uniform(5, 10), W0 * 0.5, branch=False)
    else:  # circular drum pedestal
        cylinder(pen, P, 0, 0, pw * 1.06, 0, h1)
        C = cylinder(pen, P, 0, 0, pw2 * 1.05, h1, b0)
        if rng.random() < 0.5:  # subtle carved fluting on the drum
            for k in np.linspace(-0.9, 0.7, 5):
                t = P.az + 1.5 * math.pi + k
                pen.stroke([C(t, h1 + 1.0), C(t, b0 - 1.0)], W0 * 0.38, taper=(0.15, 0.15), wobble=0)
        if rng.random() < 0.5:
            _cracks_on(pen, P, C, rng, h1 + 1.5, b0 - 1.0, 1)


def _draw_sleeve(pen, sh, el, wr, drape_drop, side, outward=False, w_arm=4.4, w_wr=3.6):
    """Anatomically draped bell-sleeve connecting shoulder -> elbow -> wrist without shoulder spikes."""
    W0 = pen.width
    sh, el, wr = np.asarray(sh, float), np.asarray(el, float), np.asarray(wr, float)
    s = 1.0 if side >= 0 else -1.0

    if not outward:
        # Arm hangs from shoulder down to elbow and sweeps inward across chest/waist
        p_sh_top = sh + np.array([-s * 1.0, -1.2])
        p_delt = sh + np.array([s * (w_arm * 0.55), 3.0])
        p_el_out = el + np.array([s * (w_arm * 0.65), 1.0])
        p_bell = np.array([el[0] * 0.46 + wr[0] * 0.54, max(el[1], wr[1]) + drape_drop])
        p_wr_b = wr + np.array([0.0, w_wr * 0.52])

        p_wr_t = wr + np.array([0.0, -w_wr * 0.48])
        p_fore_t = (el + wr) * 0.5 + np.array([0.0, -w_wr * 0.44])
        p_el_in = el + np.array([-s * (w_arm * 0.35), -w_arm * 0.38])
        p_pit = sh + np.array([-s * (w_arm * 0.55), 4.2])

        outer = chaikin([p_sh_top, p_delt, p_el_out, p_bell, p_wr_b], 2)
        inner = chaikin([p_wr_t, p_fore_t, p_el_in, p_pit], 2)
        poly = np.vstack([outer, inner])
    else:
        # Arm extends outward/upward (holding staff or raised aloft), bell-sleeve drapes beneath!
        d_f = wr - el
        Lf = max(1e-6, float(np.hypot(*d_f)))
        u_f = d_f / Lf
        n_up = np.array([-u_f[1], u_f[0]])
        if n_up[1] > 0:
            n_up = -n_up

        p_sh_top = sh + np.array([-s * 0.8, -1.4])
        p_mid_u = (sh + el) * 0.5 + np.array([s * 1.2, -1.6])
        p_el_t = el + n_up * (w_arm * 0.46)
        p_wr_t = wr + n_up * (w_wr * 0.48)

        p_wr_b = wr - n_up * (w_wr * 0.54)
        p_bell = np.array([el[0] * 0.56 + wr[0] * 0.44, max(el[1], wr[1]) + drape_drop])
        p_el_b = el + np.array([-s * 1.2, drape_drop * 0.62])
        p_pit = sh + np.array([-s * (w_arm * 0.45), 5.0])

        outer = chaikin([p_sh_top, p_mid_u, p_el_t, p_wr_t], 2)
        inner = chaikin([p_wr_b, p_bell, p_el_b, p_pit], 2)
        poly = np.vstack([outer, inner])

    pen.cv.erase(poly)

    # Clean, balanced underside/right-flank shadow hatching
    down_sh = clip_halfplane(poly, np.array([0.25 if s > 0 else -0.1, 0.96]),
                             float((el[0] * 0.5 + wr[0] * 0.5) * (0.25 if s > 0 else -0.1)
                                   + (el[1] * 0.5 + wr[1] * 0.5) * 0.96 + 1.2))
    if len(down_sh) >= 3:
        pen.hatch(down_sh, 78 if s > 0 else 100, 1.75, W0 * 0.46, coverage=(0.55, 0.95))
    if s > 0:
        r_sh = clip_halfplane(poly, np.array([0.96, 0.25]),
                              float(max(sh[0], el[0]) * 0.96 + el[1] * 0.25 - w_arm * 0.55))
        if len(r_sh) >= 3:
            pen.hatch(r_sh, 82, 1.75, W0 * 0.48, coverage=(0.6, 0.95))

    # One graceful fabric fold crease from the elbow bend into the hanging bell
    c_end = p_bell * 0.72 + el * 0.28
    pen.stroke([el + np.array([0.0, -0.5]), (el + c_end) * 0.5 + np.array([s * 0.5, 0.5]), c_end],
               W0 * 0.44, taper=(0.25, 0.65), smooth=1, wobble=0.1)

    # Sleeve cuff lip & outer drapery contours (tapered smoothly into shoulder)
    pen.stroke([p_wr_t, p_wr_b], W0 * 0.78, taper=(0.08, 0.08), wobble=0.08)
    pen.stroke(outer, W0 * 0.9, taper=(0.28, 0.04), wobble=0.15)
    pen.stroke(inner, W0 * 0.9, taper=(0.04, 0.28), wobble=0.15)


def _draw_hand(pen, pos, r=2.1, ang=0.0, kind="cup"):
    """Compact sculpted stone hand (no coin/button stripes!)."""
    W0 = pen.width
    x, y = float(pos[0]), float(pos[1])
    ca, sa = math.cos(ang), math.sin(ang)

    def R(px, py):
        return (x + px * ca - py * sa, y + px * sa + py * ca)

    if kind == "grip":
        # Compact curled fist wrapping cleanly across a staff/hilt
        pts = ellipse_pts(0, 0, r * 1.15, r * 0.82, 14)
        rot = np.stack([x + pts[:, 0] * ca - pts[:, 1] * sa, y + pts[:, 0] * sa + pts[:, 1] * ca], 1)
        pen.cv.erase(rot)
        sh = clip_halfplane(rot, np.array([0.4, 0.9]), x * 0.4 + y * 0.9 + r * 0.15)
        if len(sh) >= 3:
            pen.hatch(sh, 78, 1.3, W0 * 0.38, coverage=(0.65, 1.0))
        pen.closed(rot, W0 * 0.65, wobble=0.06)
    elif kind == "prayer":
        # Pointed steepled hands pressed together in prayer
        pts = chaikin([
            R(0.0, -r * 1.65), R(r * 0.88, -r * 0.35), R(r * 0.78, r * 0.95),
            R(0.0, r * 1.25), R(-r * 0.78, r * 0.95), R(-r * 0.88, -r * 0.35),
        ] + [R(0.0, -r * 1.65)], 2)[:-1]
        pen.cv.erase(pts)
        sh = clip_halfplane(pts, np.array([0.95, 0.3]), x * 0.95 + y * 0.3 + 0.15)
        if len(sh) >= 3:
            pen.hatch(sh, 80, 1.3, W0 * 0.38, coverage=(0.6, 1.0))
        pen.stroke([R(0.0, -r * 1.35), R(0.0, r * 1.05)], W0 * 0.42, taper=(0.2, 0.2), wobble=0)
        pen.closed(pts, W0 * 0.68, wobble=0.06)
    else:
        # Soft sculpted cupped palm/fingers
        pts = chaikin([
            R(-r * 1.0, -r * 0.35), R(-r * 0.2, -r * 0.85), R(r * 0.95, -r * 0.45),
            R(r * 1.05, r * 0.35), R(r * 0.2, r * 0.85), R(-r * 0.9, r * 0.55),
        ] + [R(-r * 1.0, -r * 0.35)], 1)[:-1]
        pen.cv.erase(pts)
        sh = clip_halfplane(pts, np.array([0.55, 0.83]), x * 0.55 + y * 0.83 + r * 0.15)
        if len(sh) >= 3:
            pen.hatch(sh, 76, 1.35, W0 * 0.38, coverage=(0.65, 1.0))
        pen.closed(pts, W0 * 0.65, wobble=0.06)


def _draw_hood(pen, rng, cx, cy, hw, hh, sh_l, sh_r, cowl_front, style="gothic", beard=False, crown=False, tilt=0.0):
    """Sculpted 3/4 stone hood & draped cowl merging seamlessly over the shoulders."""
    W0 = pen.width
    peak_h = hh * (1.20 if style == "gothic" else 1.00 if style == "monk" else 1.06)
    peak_x = cx + hw * (0.18 if style == "gothic" else 0.06) + tilt * 2.2
    cowl_spread = 1.12 if style == "veiled" else 1.0

    # Broad, natural draped cowl flowing cleanly onto both shoulders
    ctrl = [
        (sh_l[0], sh_l[1]),
        (cx - hw * 0.95 * cowl_spread, cy + hh * 0.48),
        (cx - hw * 1.00, cy - hh * 0.14),       # overhanging front-left brow
        (cx - hw * 0.62, cy - hh * 0.82),
        (peak_x, cy - peak_h),                  # hood crown / gothic peak
        (cx + hw * 0.78, cy - hh * 0.68),
        (cx + hw * 1.02, cy - hh * 0.02),       # back-right curve of hood
        (cx + hw * 0.98 * cowl_spread, cy + hh * 0.54),
        (sh_r[0], sh_r[1]),
        (cowl_front[0] + hw * 0.58, cowl_front[1] - 0.8),
        (cowl_front[0], cowl_front[1]),         # wide gentle cowl collar sweep over chest
        (cowl_front[0] - hw * 0.58, cowl_front[1] - 1.0),
    ]
    hood_poly = chaikin(ctrl + [ctrl[0]], 2)[:-1]
    pen.cv.erase(hood_poly)

    # Crisp, restrained shadow hatching on the back-right 35% of the hood
    sh_poly = clip_halfplane(hood_poly, np.array([0.96, 0.28]), cx * 0.96 + cy * 0.28 + hw * 0.24)
    if len(sh_poly) >= 3:
        pen.hatch(sh_poly, 78, 1.85, W0 * 0.48, coverage=(0.65, 1.0))

    # Recessed 3/4 pointed-arch hood opening (offset slightly front-left)
    ox = cx - hw * 0.22
    oy = cy + hh * 0.05
    ow = hw * 0.50
    oh = hh * 0.64
    open_ctrl = [
        (ox - ow * 0.06, oy - oh * 0.96),       # gothic brow arch peak
        (ox + ow * 0.72, oy - oh * 0.54),
        (ox + ow * 0.90, oy + oh * 0.12),       # near (right) cheek curve
        (ox + ow * 0.52, oy + oh * 0.78),
        (ox - ow * 0.10, oy + oh * 0.90),       # chin shadow base
        (ox - ow * 0.72, oy + oh * 0.52),
        (ox - ow * 0.86, oy - oh * 0.16),       # far (left) foreshortened cheek
        (ox - ow * 0.56, oy - oh * 0.70),
    ]
    opening = chaikin(open_ctrl + [open_ctrl[0]], 2)[:-1]
    pen.blot(opening, jitter=0.08)

    # Carved stone circlet/diadem across the hood brow (occasional royal/divine variant)
    if crown:
        br = [
            (ox - ow * 0.98, oy - oh * 0.52),
            (ox - ow * 0.05, oy - oh * 1.08),
            (ox + ow * 0.98, oy - oh * 0.58),
            (ox + ow * 0.92, oy - oh * 0.34),
            (ox - ow * 0.05, oy - oh * 0.82),
            (ox - ow * 0.92, oy - oh * 0.30),
        ]
        pen.cv.erase(br)
        pen.closed(br, W0 * 0.58, wobble=0.06)

    # Thick carved cloth lip framing the near edge of the hood opening (tapers cleanly below chin)
    rim_r = chaikin([
        (ox - ow * 0.12, oy - oh * 1.14),
        (ox + ow * 0.96, oy - oh * 0.66),
        (ox + ow * 1.14, oy + oh * 0.16),
        (ox + ow * 0.68, oy + oh * 0.92),
        (ox + ow * 0.08, oy + oh * 1.12),
    ], 2)
    pen.stroke(rim_r, W0 * 0.56, taper=(0.25, 0.55), wobble=0.08)

    # Optional full sculpted stone sage beard draping cleanly OVER the chin rim
    if beard:
        b_poly = chaikin([
            (ox - ow * 0.62, oy + oh * 0.32),
            (ox + ow * 0.56, oy + oh * 0.30),
            (ox + ow * 0.52, oy + oh * 1.08),
            (ox - ow * 0.06, oy + oh * 1.62),
            (ox - ow * 0.58, oy + oh * 1.06),
        ] + [(ox - ow * 0.62, oy + oh * 0.32)], 1)[:-1]
        pen.cv.erase(b_poly)
        b_sh = clip_halfplane(b_poly, np.array([1.0, 0.0]), ox + ow * 0.05)
        if len(b_sh) >= 3:
            pen.hatch(b_sh, 84, 1.35, W0 * 0.38, coverage=(0.6, 1.0))
        for f in (-0.28, 0.0, 0.24):
            pen.stroke([(ox + ow * f, oy + oh * 0.42), (ox + ow * f * 0.6, oy + oh * (1.42 - abs(f) * 0.6))],
                       W0 * 0.34, taper=(0.15, 0.55), wobble=0.06)
        pen.closed(b_poly, W0 * 0.58, wobble=0.08)

    # Wide, gentle cowl drapery fold curving across the collarbone
    fold1 = chaikin([
        (cx - hw * 0.78, cy + hh * 0.62),
        (cowl_front[0] - hw * 0.05, cowl_front[1] - 2.2),
        (cx + hw * 0.76, cy + hh * 0.64),
    ], 2)
    pen.stroke(fold1, W0 * 0.42, taper=(0.35, 0.35), wobble=0.08)
    pen.stroke(chaikin([(peak_x + hw * 0.14, cy - peak_h * 0.74),
                        (cx + hw * 0.66, cy - hh * 0.02),
                        (cx + hw * 0.64, cy + hh * 0.56)], 1),
               W0 * 0.40, taper=(0.35, 0.45), wobble=0.08)

    pen.closed(hood_poly, W0 * 0.95, wobble=0.14)


def _draw_nimbus(pen, rng, cx, cy, r_out, r_in):
    """Carved stone halo/nimbus ring mounted behind the hood (often weathered/cracked)."""
    W0 = pen.width
    broken_gap = rng.random() < 0.45
    gap_a = rng.uniform(-2.5, -0.6)
    gap_w = rng.uniform(0.35, 0.72) if broken_gap else 0.0
    n = 36
    if broken_gap:
        ts = np.linspace(gap_a + gap_w / 2, gap_a + TAU - gap_w / 2, n)
        outer = [(cx + r_out * math.cos(t), cy + r_out * math.sin(t)) for t in ts]
        inner = [(cx + r_in * math.cos(t), cy + r_in * math.sin(t)) for t in ts[::-1]]
        ring = outer + inner
        pen.cv.erase(ring)
        sh = clip_halfplane(ring, np.array([0.9, 0.3]), cx * 0.9 + cy * 0.3 + r_in * 0.2)
        if len(sh) >= 3:
            pen.hatch(sh, 78, 1.6, W0 * 0.48, coverage=(0.65, 1.0))
        pen.closed(ring, W0 * 0.82, wobble=0.12)
    else:
        outer = ellipse_pts(cx, cy, r_out, r_out, n)
        inner = ellipse_pts(cx, cy, r_in, r_in, n)
        pen.cv.erase(outer)
        sh = clip_halfplane(outer, np.array([0.9, 0.3]), cx * 0.9 + cy * 0.3 + r_in * 0.15)
        if len(sh) >= 3:
            pen.hatch(sh, 78, 1.6, W0 * 0.48, coverage=(0.65, 1.0))
        pen.cv.clear(inner)
        pen.closed(outer, W0 * 0.82, wobble=0.12)
        pen.closed(inner, W0 * 0.72, wobble=0.1)
    for t in np.linspace(0, TAU, 12, endpoint=False):
        if broken_gap and abs((t - gap_a + math.pi) % TAU - math.pi) < gap_w * 0.6:
            continue
        p0 = (cx + (r_in + 0.6) * math.cos(t), cy + (r_in + 0.6) * math.sin(t))
        p1 = (cx + (r_out - 0.6) * math.cos(t), cy + (r_out - 0.6) * math.sin(t))
        pen.stroke([p0, p1], W0 * 0.34, taper=(0.2, 0.2), wobble=0)


def _draw_fallen_head(pen, rng, x, y, tilt):
    """Tumbled sculpted stone hooded head lying on the ground/plinth."""
    W0 = pen.width
    ca, sa = math.cos(tilt), math.sin(tilt)

    def R(px, py):
        return (x + px * ca - py * sa, y + px * sa + py * ca)

    # Distinct peaked stone cowl lying on its side with jagged broken neck edge
    ctrl = [
        R(-5.4, 5.4), R(-7.0, 0.8), R(-5.4, -5.6),
        R(0.8, -10.5), R(6.4, -4.8), R(7.2, 1.4), R(5.4, 5.4),
        R(1.6, 6.4), R(-1.8, 5.6),
    ]
    poly = chaikin(ctrl + [ctrl[0]], 2)[:-1]
    pen.cv.erase(poly)
    sh = clip_halfplane(poly, np.array([0.45, 0.89]), x * 0.45 + y * 0.89 + 1.2)
    if len(sh) >= 3:
        pen.hatch(sh, 74, 1.55, W0 * 0.48, coverage=(0.65, 1.0))
    open_pts = chaikin([
        R(-0.6, -5.8), R(3.2, -2.8), R(3.8, 1.6),
        R(1.0, 4.4), R(-2.6, 3.4), R(-3.6, -0.8), R(-2.8, -4.2),
    ] + [R(-0.6, -5.8)], 2)[:-1]
    pen.blot(open_pts, jitter=0.08)
    rim = chaikin([R(-0.6, -7.2), R(4.6, -3.2), R(4.8, 2.4), R(1.8, 5.2)], 1)
    pen.stroke(rim, W0 * 0.5, taper=(0.2, 0.35), wobble=0.08)
    pen.stroke([R(-4.4, 5.2), R(0.0, 4.5), R(4.4, 5.2)], W0 * 0.55, taper=(0.15, 0.15), wobble=0.12)
    pen.closed(poly, W0 * 0.9, wobble=0.14)


def _draw_statue_sword(pen, rng, p_tip, p_guard, p_pommel, broken=False):
    """Bold monumental two-handed greatsword held point-down in front of the statue."""
    W0 = pen.width
    p_tip = np.asarray(p_tip, float)
    p_guard = np.asarray(p_guard, float)
    p_pommel = np.asarray(p_pommel, float)
    d = p_guard - p_tip
    Lb = max(1e-6, float(np.hypot(*d)))
    u = d / Lb
    nrm = np.array([u[1], -u[0]])
    if nrm[0] < 0:
        nrm = -nrm

    # Broad tapered greatsword blade (clearly readable against the robe!)
    wb = rng.uniform(4.8, 5.6)
    if broken:
        p_brk = p_tip + u * (Lb * rng.uniform(0.32, 0.52))
        jag = [
            p_brk - nrm * (wb * 0.42),
            p_brk - nrm * (wb * 0.12) - u * 2.4,
            p_brk + nrm * (wb * 0.18) + u * 1.2,
            p_brk + nrm * (wb * 0.42) - u * 1.8,
        ]
        blade = [p_guard - nrm * (wb * 0.5), *jag, p_guard + nrm * (wb * 0.5)]
        fuller_bot = p_brk + u * 2.6
    else:
        p_taper = p_tip + u * 8.0
        blade = [
            p_guard - nrm * (wb * 0.52),
            p_taper - nrm * (wb * 0.34),
            p_tip,
            p_taper + nrm * (wb * 0.34),
            p_guard + nrm * (wb * 0.52),
        ]
        fuller_bot = p_tip + u * 7.0

    pen.cv.erase(blade)
    sh = clip_halfplane(blade, nrm, float(p_guard @ nrm) - 0.1)
    if len(sh) >= 3:
        pen.hatch(sh, 78, 1.35, W0 * 0.52, coverage=(0.7, 1.0))
    pen.stroke([fuller_bot, p_guard - u * 1.8], W0 * 0.58, taper=(0.35, 0.1), wobble=0)
    pen.closed(blade, W0 * 0.92, wobble=0.08)

    # Bold sculpted gothic crossguard
    gl = rng.uniform(16.5, 20.5)
    gt = rng.uniform(2.6, 3.2)
    droop = rng.uniform(-0.8, 2.4)
    guard = [
        p_guard - nrm * (gl * 0.5) - u * droop,
        p_guard - nrm * (gl * 0.5) + u * (gt - droop),
        p_guard + u * (gt * 1.2),
        p_guard + nrm * (gl * 0.5) + u * (gt - droop),
        p_guard + nrm * (gl * 0.5) - u * droop,
        p_guard - u * 0.9,
    ]
    pen.cv.erase(guard)
    gsh = clip_halfplane(guard, nrm, float(p_guard @ nrm))
    if len(gsh) >= 3:
        pen.hatch(gsh, 78, 1.35, W0 * 0.48, coverage=(0.7, 1.0))
    pen.closed(guard, W0 * 0.88, wobble=0.08)

    # Longer wrapped grip & faceted diamond pommel (clearly distinct from the hands!)
    g0 = p_guard + u * (gt * 0.9)
    g1 = p_pommel - u * 2.8
    grip = [g0 - nrm * 1.5, g1 - nrm * 1.3, g1 + nrm * 1.3, g0 + nrm * 1.5]
    pen.cv.erase(grip)
    pen.closed(grip, W0 * 0.78, wobble=0.06)
    pom = [
        p_pommel + u * 3.0,
        p_pommel + nrm * 2.6,
        p_pommel - u * 2.6,
        p_pommel - nrm * 2.6,
    ]
    pen.cv.erase(pom)
    psh = clip_halfplane(pom, nrm, float(p_pommel @ nrm))
    if len(psh) >= 3:
        pen.hatch(psh, 78, 1.2, W0 * 0.42)
    pen.closed(pom, W0 * 0.82, wobble=0.05)


def _draw_statue_staff(pen, rng, p_bot, p_top, side, finial="crook"):
    """Tall carved stone staff with distinct medieval/divine finials (or snapped shaft)."""
    W0 = pen.width
    p_bot = np.asarray(p_bot, float)
    p_top = np.asarray(p_top, float)
    d = p_top - p_bot
    L = max(1e-6, float(np.hypot(*d)))
    u = d / L
    nrm = np.array([u[1], -u[0]])
    if nrm[0] < 0:
        nrm = -nrm

    # subtly tapered shaft
    w_b, w_t = 1.25, 1.0
    if finial == "snapped":
        jag = [
            p_top - nrm * w_t,
            p_top - nrm * 0.25 + u * 2.4,
            p_top + nrm * 0.35 - u * 0.8,
            p_top + nrm * w_t + u * 1.2,
        ]
        shaft = [p_bot - nrm * w_b, *jag, p_bot + nrm * w_b]
    else:
        shaft = [p_bot - nrm * w_b, p_top - nrm * w_t, p_top + nrm * w_t, p_bot + nrm * w_b]
    pen.cv.erase(shaft)
    # thin shadow edge along the right side of the staff
    pen.stroke([p_bot + nrm * (w_b * 0.35), p_top + nrm * (w_t * 0.35)], W0 * 0.45, taper=(0.05, 0.05), wobble=0.1)
    pen.closed(shaft, W0 * 0.78, wobble=0.12)

    if finial == "snapped":
        return

    # decorative collar ring just below the finial
    c_p = p_top - u * 1.6
    collar = [c_p - nrm * 2.1 - u * 0.9, c_p - nrm * 2.1 + u * 0.9,
              c_p + nrm * 2.1 + u * 0.9, c_p + nrm * 2.1 - u * 0.9]
    pen.cv.erase(collar)
    pen.closed(collar, W0 * 0.68, wobble=0.05)

    if finial == "crook":
        # Pastoral crosier / shepherd's crook curving UP over the top (+u) and curling cleanly outward
        out = side
        ts = np.linspace(0, 1.42 * math.pi, 24)
        cr = 5.0
        cc = p_top + u * 0.4 + nrm * (out * cr)
        pts_o, pts_i = [], []
        for k, t in enumerate(ts):
            f = k / (len(ts) - 1)
            r_cur = cr * (1.0 - 0.35 * f)
            ang = (math.pi if out > 0 else 0.0) - out * t
            rad = np.array([math.cos(ang), math.sin(ang)])
            rvec = nrm * rad[0] + u * rad[1]
            p = cc + rvec * r_cur
            half = 1.05 * (1.0 - 0.45 * f)
            pts_o.append(p + rvec * half)
            pts_i.append(p - rvec * half)
        hook_poly = pts_o + pts_i[::-1]
        pen.cv.erase(hook_poly)
        pen.closed(hook_poly, W0 * 0.75, wobble=0.08)
    elif finial == "ring":
        # carved divine ring / sun-wheel atop the staff
        r_out, r_in = 5.6, 3.6
        rc = p_top + u * (r_out - 0.4)
        outer = ellipse_pts(rc[0], rc[1], r_out, r_out, 22)
        inner = ellipse_pts(rc[0], rc[1], r_in, r_in, 18)
        pen.cv.erase(outer)
        sh = clip_halfplane(outer, nrm, float(rc @ nrm) + 0.8)
        if len(sh) >= 3:
            pen.hatch(sh, 78, 1.3, W0 * 0.42)
        pen.cv.clear(inner)
        pen.closed(outer, W0 * 0.78, wobble=0.08)
        pen.closed(inner, W0 * 0.65, wobble=0.06)
        # tiny inner central orb or cross
        if rng.random() < 0.6:
            orb = ellipse_pts(rc[0], rc[1], 1.5, 1.5, 10)
            pen.cv.erase(orb)
            pen.closed(orb, W0 * 0.55, wobble=0)
        else:
            pen.stroke([rc - u * r_in, rc + u * r_in], W0 * 0.48, taper=(0, 0), wobble=0)
            pen.stroke([rc - nrm * r_in, rc + nrm * r_in], W0 * 0.48, taper=(0, 0), wobble=0)
    elif finial == "crescent":
        # upturned lunar crescent cradling a small orb
        rc = p_top + u * 5.2
        r_o, r_i = 5.4, 4.2
        ts_o = np.linspace(-0.22 * math.pi, 1.22 * math.pi, 20)
        ts_i = np.linspace(1.16 * math.pi, -0.16 * math.pi, 16)
        outer = [rc + nrm * (r_o * math.cos(t)) - u * (r_o * math.sin(t)) for t in ts_o]
        inner = [rc + u * 1.4 + nrm * (r_i * math.cos(t)) - u * (r_i * math.sin(t)) for t in ts_i]
        cresc = outer + inner
        pen.cv.erase(cresc)
        sh = clip_halfplane(cresc, nrm, float(rc @ nrm) + 0.4)
        if len(sh) >= 3:
            pen.hatch(sh, 78, 1.2, W0 * 0.42)
        pen.closed(cresc, W0 * 0.75, wobble=0.08)
        oc = rc + u * 1.5
        orb = ellipse_pts(oc[0], oc[1], 1.6, 1.6, 10)
        pen.cv.erase(orb)
        pen.closed(orb, W0 * 0.55, wobble=0)
    elif finial == "lantern":
        # curved hook holding a hanging gothic lantern
        out = side
        hook = chaikin([p_top, p_top + u * 5.5, p_top + u * 6.5 + nrm * (out * 4.5),
                        p_top + u * 3.5 + nrm * (out * 6.2)], 2)
        pen.stroke(hook, W0 * 1.0, taper=(0.05, 0.2), wobble=0.08)
        lp = p_top + u * 3.2 + nrm * (out * 6.2)
        pen.stroke([lp, lp - u * 3.0], W0 * 0.5, taper=(0, 0), wobble=0)
        lc = lp - u * 7.2
        # gothic lantern body: peaked cap + chamber + base finial
        cap = [lc + u * 4.2, lc + u * 1.8 + nrm * 3.4, lc + u * 1.8 - nrm * 3.4]
        body = [lc + u * 1.8 - nrm * 2.5, lc - u * 3.0 - nrm * 1.9,
                lc - u * 3.8, lc - u * 3.0 + nrm * 1.9, lc + u * 1.8 + nrm * 2.5]
        pen.cv.erase(body)
        pen.blot([lc + u * 1.2 - nrm * 1.5, lc - u * 2.2 - nrm * 1.1,
                  lc - u * 2.2 + nrm * 1.1, lc + u * 1.2 + nrm * 1.5], jitter=0.05)
        pen.closed(body, W0 * 0.68, wobble=0.05)
        pen.cv.erase(cap)
        pen.closed(cap, W0 * 0.72, wobble=0.05)
    else:  # "orb": forked stone prongs cradling an orb
        oc = p_top + u * 4.6
        for sgn in (-1, 1):
            prong = chaikin([p_top, p_top + u * 1.8 + nrm * (sgn * 3.8),
                             oc + u * 1.2 + nrm * (sgn * 4.2)], 1)
            pen.stroke(prong, W0 * 0.85, taper=(0.05, 0.35), wobble=0.06)
        orb = ellipse_pts(oc[0], oc[1], 3.4, 3.4, 16)
        pen.cv.erase(orb)
        sh = clip_halfplane(orb, np.array([0.8, 0.6]), oc[0] * 0.8 + oc[1] * 0.6 + 0.4)
        if len(sh) >= 3:
            pen.hatch(sh, 76, 1.25, W0 * 0.45)
        pen.closed(orb, W0 * 0.78, wobble=0.06)


def _draw_orb_relic(pen, cx, cy, r=4.6):
    """Carved stone sphere/orb held in the statue's hands or raised aloft."""
    W0 = pen.width
    orb = ellipse_pts(cx, cy, r, r, 20)
    pen.cv.erase(orb)
    sh = clip_halfplane(orb, np.array([0.78, 0.62]), cx * 0.78 + cy * 0.62 + r * 0.1)
    if len(sh) >= 3:
        pen.hatch(sh, 74, 1.3, W0 * 0.48, coverage=(0.65, 1.0))
    pen.closed(orb, W0 * 0.85, wobble=0.08)


def _draw_tome(pen, cx, cy, w=8.8, h=7.4):
    """Carved stone tablet or open holy tome cradled before the chest."""
    W0 = pen.width
    # angled book/tablet in 3/4 view
    spine = [(cx - w * 0.08, cy - h * 0.46), (cx + w * 0.02, cy + h * 0.50)]
    left_p = [spine[0], (cx - w * 0.64, cy - h * 0.36), (cx - w * 0.54, cy + h * 0.48), spine[1]]
    right_p = [spine[0], (cx + w * 0.56, cy - h * 0.52), (cx + w * 0.66, cy + h * 0.36), spine[1]]
    pen.cv.erase(left_p)
    pen.cv.erase(right_p)
    pen.hatch(right_p, 80, 1.5, W0 * 0.42, coverage=(0.5, 0.9))
    # subtle carved text lines on left page
    for f in (0.28, 0.50, 0.72):
        p0 = (cx - w * 0.48 + f * 0.6, cy - h * 0.28 + f * h * 0.65)
        p1 = (cx - w * 0.16 + f * 0.6, cy - h * 0.32 + f * h * 0.65)
        pen.stroke([p0, p1], W0 * 0.32, taper=(0.1, 0.1), wobble=0)
    pen.closed(left_p, W0 * 0.75, wobble=0.08)
    pen.closed(right_p, W0 * 0.75, wobble=0.08)


def _draw_flame_beacon(pen, rng, cx, cy, scale=1.0):
    """Carved stone brazier/torch flame held aloft by a warden statue."""
    W0 = pen.width
    bowl = [(cx - 3.6 * scale, cy + 1.0), (cx + 3.6 * scale, cy + 1.0),
            (cx + 2.2 * scale, cy + 4.2 * scale), (cx - 2.2 * scale, cy + 4.2 * scale)]
    pen.cv.erase(bowl)
    pen.hatch(clip_halfplane(bowl, np.array([1.0, 0.0]), cx + 0.3), 80, 1.2, W0 * 0.45)
    pen.closed(bowl, W0 * 0.75, wobble=0.08)
    flame = chaikin([
        (cx - 2.6 * scale, cy + 1.0),
        (cx - 3.4 * scale, cy - 3.2 * scale),
        (cx - 1.0 * scale, cy - 7.5 * scale),
        (cx + 0.4 * scale, cy - 11.5 * scale),
        (cx + 2.2 * scale, cy - 6.6 * scale),
        (cx + 3.4 * scale, cy - 2.8 * scale),
        (cx + 2.6 * scale, cy + 1.0),
    ], 2)
    pen.cv.erase(flame)
    sh = clip_halfplane(flame, np.array([1.0, 0.15]), cx + 0.5)
    if len(sh) >= 3:
        pen.hatch(sh, 78, 1.3, W0 * 0.42, coverage=(0.6, 0.95))
    pen.stroke([(cx - 0.3 * scale, cy + 0.2), (cx + 0.2 * scale, cy - 5.5 * scale)],
               W0 * 0.45, taper=(0.1, 0.7), wobble=0.1)
    pen.closed(flame, W0 * 0.78, wobble=0.1)


def statue_robed(pen, P, rng):
    """Monumental hooded robed statue on a plinth with varied poses, drapery, and ruin states."""
    W0 = pen.width

    # 1. Plinth & overall proportions
    plinth_kind = str(rng.choice(["square", "square", "octagonal", "round"]))
    pw = rng.uniform(19.5, 24.0)
    b0 = rng.uniform(13.0, 17.5)
    sw = rng.uniform(0.92, 1.08)          # width scale (slender saint .. broad warden)
    H_body = rng.uniform(58.0, 70.0)      # plinth top to collar/neck
    H_head = rng.uniform(15.5, 18.5)
    yaw = rng.uniform(-0.14, 0.22) if plinth_kind != "square" else rng.uniform(-0.04, 0.08)
    cy_w, sy_w = math.cos(yaw), math.sin(yaw)

    def Q(u, v, h=0.0):
        return P(u * cy_w + v * sy_w, u * sy_w - v * cy_w, b0 + h)

    # 2. Choose pose and ruin state
    ruin_roll = rng.random()
    if ruin_roll < 0.11:
        state = "broken_torso"
    elif ruin_roll < 0.36:
        state = "headless"
    else:
        state = "intact"

    pose = str(rng.choice(["sword", "staff", "relic", "raised"], p=[0.28, 0.28, 0.24, 0.20]))
    if state == "broken_torso":
        pose = "sword" if rng.random() < 0.45 else "relic"

    staff_side = 1 if rng.random() < 0.65 else -1
    raised_side = -1 if rng.random() < 0.55 else 1
    has_mantle = rng.random() < 0.68 and state != "broken_torso"
    has_cloak = rng.random() < 0.55
    has_nimbus = state == "intact" and rng.random() < 0.22
    has_vines = rng.random() < 0.32
    hood_style = str(rng.choice(["gothic", "gothic", "monk", "veiled"]))
    has_beard = state == "intact" and rng.random() < 0.18
    has_crown = state == "intact" and not has_beard and rng.random() < 0.15
    sway = rng.normal(0.0, 0.85)

    # 3. Back scatter behind the plinth
    back_items = []
    for _ in range(rng.integers(1, 4)):
        t = rng.uniform(0.15, math.pi - 0.15) + P.az
        d = rng.uniform(1.1, 1.38)
        bx, by = pw * d * math.cos(t), pw * d * math.sin(t)
        back_items.append((P.depth(bx, by), str(rng.choice(["tuft", "stone"])), (bx, by)))
    for _, k, (bx, by) in sorted(back_items, key=lambda it: -it[0]):
        if k == "tuft":
            pen.tuft(*P(bx, by), rng.uniform(3.5, 5.5))
        else:
            pen.stone(*P(bx, by), rng.uniform(1.5, 2.8))

    # 4. Draw plinth
    _statue_plinth(pen, P, rng, plinth_kind, pw, b0)

    # 5. Parametric sculpted robed body S(f, t) for f in [0, 1]
    fs_k = np.array([0.00, 0.12, 0.30, 0.55, 0.72, 0.86, 0.94, 1.00])
    rx_k = np.array([13.2, 11.4, 10.2,  8.4, 10.6, 13.0, 10.2,  5.5]) * sw
    ry_k = np.array([ 9.6,  8.5,  7.6,  6.5,  7.4,  8.2,  6.8,  5.0]) * sw
    flute_ph = rng.uniform(0, TAU, 2)
    flute_amp = rng.uniform(0.65, 1.15) * sw

    def body_radii(f):
        rx = float(np.interp(f, fs_k, rx_k))
        ry = float(np.interp(f, fs_k, ry_k))
        return rx, ry

    def body_pt(f, t):
        rx, ry = body_radii(f)
        # organic drapery flutes along lower skirt hem
        skirt_w = max(0.0, 1.0 - f / 0.56) ** 1.25
        dr = flute_amp * skirt_w * (0.65 * math.cos(5 * t + flute_ph[0]) + 0.35 * math.cos(7 * t + flute_ph[1]))
        uo = sway * math.sin(math.pi * f)
        u = uo + (rx + dr) * math.cos(t)
        v = (ry + dr * 0.75) * math.sin(t)
        return Q(u, v, f * H_body)

    daz = P.az - yaw

    def sil_angles(f):
        rx, ry = body_radii(f)
        t_r = -math.atan2(ry * math.sin(daz), rx * math.cos(daz))
        return t_r, t_r + math.pi

    # Determine effective top height if broken_torso (headless breaks at the narrow neck stub!)
    if state == "broken_torso":
        f_top = rng.uniform(0.56, 0.74)
        jag_amp = rng.uniform(4.2, 6.8)
    else:
        f_top = 1.0
        jag_amp = 0.0
    brk_ph = rng.uniform(0, TAU, 2)
    brk_tilt = rng.uniform(-0.22, 0.22) * (1.8 if state == "broken_torso" else 1.0)

    def hat_f(t):
        if jag_amp <= 0:
            return f_top
        dh = jag_amp * (0.45 + 0.32 * math.sin(2 * t + brk_ph[0]) + 0.23 * math.sin(5 * t + brk_ph[1]))
        dh += brk_tilt * 10.0 * math.cos(t)
        return max(0.32, f_top - dh / H_body)

    # 6. Far/behind layers: Stone Nimbus (behind hood) & Back Cloak (behind skirt)
    head_c = Q(sway * 0.2, 0.5, H_body + H_head * 0.42)
    if has_nimbus:
        hw_est = 8.4 * sw
        _draw_nimbus(pen, rng, head_c[0], head_c[1] - 1.0, hw_est * 1.72, hw_est * 1.24)

    if has_cloak and state != "broken_torso":
        # flowing back cape tucked smoothly behind the shoulders, flaring gracefully toward the hem
        cl_top = min(0.84, f_top - 0.05)
        cl_fs = np.linspace(0.02, cl_top, 10)
        cl_l, cl_r = [], []
        cl_w = rng.uniform(2.6, 3.6) * sw
        for f in cl_fs:
            rx, ry = body_radii(f)
            frac_down = max(0.0, (cl_top - f) / max(1e-6, cl_top))
            flare = (frac_down ** 1.15) * cl_w
            tuck = 0.84 + 0.16 * frac_down
            cl_l.append(Q(-rx * tuck - flare, -ry * 0.48, f * H_body))
            cl_r.append(Q(rx * tuck + flare * 0.65, -ry * 0.55, f * H_body))
        cloak_poly = cl_l + cl_r[::-1]
        pen.cv.erase(cloak_poly)
        pen.hatch(cloak_poly, 82, 1.6, W0 * 0.52, coverage=(0.65, 1.0))
        pen.stroke(chaikin(cl_l, 1), W0 * 0.85, taper=(0.1, 0.2), wobble=0.18)
        pen.stroke(chaikin(cl_r, 1), W0 * 0.85, taper=(0.1, 0.2), wobble=0.18)
        pen.stroke([cl_l[0], Q(0, -ry_k[0] * 0.9, 0.5), cl_r[0]], W0 * 0.8, taper=(0.1, 0.1), wobble=0.15)

    # Far-side raised arm (drawn cleanly with outward bell-sleeve!)
    def draw_raised_arm(side):
        sh_u_r = side * (10.6 * sw)
        sh = Q(sh_u_r, 0.8, H_body * 0.85)
        el = Q(sh_u_r + side * 7.5 * sw, 3.2, H_body * 0.78)
        snapped = rng.random() < 0.28
        if snapped:
            wr = Q(sh_u_r + side * 11.0 * sw, 4.4, H_body * 0.87)
            _draw_sleeve(pen, sh, el, wr, drape_drop=10.5 * sw, side=side, outward=True, w_arm=4.4 * sw, w_wr=3.6 * sw)
            # angled ragged fractured stone forearm stub
            d = np.asarray(wr) - np.asarray(el)
            u = d / max(1e-6, float(np.hypot(*d)))
            n = np.array([-u[1], u[0]])
            stub = [
                wr - n * 1.9, wr + u * 2.2 - n * 1.6, wr + u * 0.9 - n * 0.2,
                wr + u * 2.8 + n * 0.8, wr + u * 1.4 + n * 1.9, wr + n * 1.9,
            ]
            pen.cv.erase(stub)
            pen.stroke([wr, wr + u * 1.8 - n * 0.3], W0 * 0.42, taper=(0.2, 0.2))
            pen.closed(stub, W0 * 0.75, wobble=0.1)
        else:
            wr = Q(sh_u_r + side * 11.8 * sw, 5.2, H_body * 1.02)
            _draw_sleeve(pen, sh, el, wr, drape_drop=13.0 * sw, side=side, outward=True, w_arm=4.5 * sw, w_wr=3.8 * sw)
            beacon = str(rng.choice(["flame", "orb", "palm"], p=[0.42, 0.30, 0.28]))
            if beacon == "orb":
                _draw_orb_relic(pen, wr[0] + side * 1.2, wr[1] - 5.4, r=4.1)
                _draw_hand(pen, (wr[0] + side * 0.5, wr[1] - 1.2), r=2.2, ang=-side * 0.4)
            elif beacon == "flame":
                _draw_flame_beacon(pen, rng, wr[0] + side * 1.0, wr[1] - 5.6, scale=0.92)
                _draw_hand(pen, (wr[0] + side * 0.4, wr[1] - 1.0), r=2.1, ang=-side * 0.3)
            else:  # solemn raised open-palm warden gesture (with inner thumb silhouette)
                hx, hy = wr[0] + side * 1.0, wr[1] - 3.6
                inw = -side
                palm = chaikin([
                    (hx - inw * 1.8, hy + 2.4), (hx - inw * 2.1, hy - 0.6), (hx - inw * 1.4, hy - 3.4),
                    (hx + inw * 0.2, hy - 4.0), (hx + inw * 1.6, hy - 2.8), (hx + inw * 1.6, hy - 0.4),
                    (hx + inw * 3.1, hy - 0.2), (hx + inw * 2.8, hy + 1.0), (hx + inw * 1.4, hy + 2.4),
                ] + [(hx - inw * 1.8, hy + 2.4)], 1)[:-1]
                pen.cv.erase(palm)
                pen.hatch(clip_halfplane(palm, np.array([1.0, 0.0]), hx + 0.3), 80, 1.3, W0 * 0.36)
                pen.closed(palm, W0 * 0.68, wobble=0.06)
                pen.stroke([(hx - inw * 0.5, hy - 3.2), (hx - inw * 0.5, hy - 0.5)], W0 * 0.32, taper=(0.1, 0.3), wobble=0)
                pen.stroke([(hx + inw * 0.7, hy - 3.0), (hx + inw * 0.6, hy - 0.4)], W0 * 0.32, taper=(0.1, 0.3), wobble=0)

    if pose == "raised" and raised_side < 0 and state != "broken_torso":
        draw_raised_arm(-1)

    # 7. Main sculpted robed body (continuous lofted form!)
    n_z = 22
    t_r0, t_l0 = sil_angles(0.0)
    front_ts = np.linspace(t_r0, t_l0, 26)
    hem_pts = [body_pt(0.0, t) for t in front_ts]

    left_sil, right_sil = [], []
    for i in range(n_z):
        frac = i / (n_z - 1)
        f_l_max = hat_f(sil_angles(frac * f_top)[1])
        f_r_max = hat_f(sil_angles(frac * f_top)[0])
        fl = frac * f_l_max
        fr = frac * f_r_max
        t_rl, t_ll = sil_angles(fl)
        t_rr, _ = sil_angles(fr)
        left_sil.append(body_pt(fl, t_ll))
        right_sil.append(body_pt(fr, t_rr))

    if state != "broken_torso":
        t_rt, t_lt = sil_angles(1.0)
        top_arc = [body_pt(1.0, t) for t in np.linspace(t_lt, t_rt, 12)]
        body_poly = hem_pts + left_sil[1:] + top_arc[1:-1] + right_sil[:0:-1]
    else:
        brk_front = [body_pt(hat_f(t), t) for t in np.linspace(t_l0, t_r0, 22)]
        brk_ring = [body_pt(hat_f(t), t) for t in np.linspace(0, TAU, 36, endpoint=False)]
        body_poly = hem_pts + left_sil[1:] + brk_front[1:-1] + right_sil[:0:-1]

    pen.cv.erase(body_poly)
    if state == "broken_torso":
        pen.cv.erase(brk_ring)

    # Right/east sculptural shadow hatching following the curved silhouette
    sh_span = 1.02
    sh_inner, sh_outer = [], []
    for i in range(n_z):
        frac = i / (n_z - 1)
        t_rr, _ = sil_angles(frac * f_top)
        fr = frac * hat_f(t_rr)
        fi = frac * hat_f(t_rr + sh_span)
        sh_outer.append(body_pt(fr, t_rr))
        sh_inner.append(body_pt(fi, t_rr + sh_span))
    sh_poly = sh_outer + sh_inner[::-1]
    pen.hatch(sh_poly, 82, 1.55, W0 * 0.62, coverage=(0.72, 1.0))

    # Skirt drapery folds (keep center front clean when holding greatsword so blade pops!)
    n_folds = int(rng.integers(5, 8))
    fold_ts = np.linspace(t_r0 + 0.42, t_l0 - 0.24, n_folds) + rng.normal(0, 0.04, n_folds)
    f_waist = 0.55
    for idx, tf in enumerate(fold_ts):
        if f_top < 0.4:
            continue
        if pose == "sword" and abs(tf - math.pi * 0.5) < 0.28:
            continue
        f_hi = min(f_waist - rng.uniform(0.01, 0.06), hat_f(tf) - 0.04)
        f_lo = rng.uniform(0.01, 0.08) if idx % 2 == 0 else rng.uniform(0.02, 0.14)
        if f_hi <= f_lo + 0.12:
            continue
        t_mid = tf + (math.pi * 0.5 - tf) * 0.16 + rng.normal(0, 0.03)
        t_hi = tf + (math.pi * 0.5 - tf) * 0.26
        f_pts = [body_pt(f, np.interp(f, [f_lo, (f_lo + f_hi) * 0.5, f_hi], [tf, t_mid, t_hi]))
                 for f in np.linspace(f_lo, f_hi, 8)]
        major = (idx % 2 == 0)
        pen.stroke(f_pts, W0 * (0.60 if major else 0.40),
                   taper=(0.08, 0.55 if major else 0.65), smooth=1, wobble=0.1)
        if major:
            f_sh = [body_pt(f, np.interp(f, [f_lo, f_lo + 0.22], [tf - 0.09, tf - 0.05]))
                    for f in np.linspace(f_lo + 0.01, min(f_hi, f_lo + 0.22), 5)]
            pen.stroke(f_sh, W0 * 0.34, taper=(0.15, 0.7), wobble=0.05)

    # Upper chest soft gathers (above waist)
    if f_top > 0.65:
        for k in (-0.38, 0.05, 0.42):
            tc = math.pi * 0.5 + k
            fh = min(0.84, hat_f(tc) - 0.05)
            if fh > f_waist + 0.1:
                pts = [body_pt(f, tc + k * 0.12 * (f - f_waist)) for f in np.linspace(f_waist + 0.04, fh, 5)]
                pen.stroke(pts, W0 * 0.40, taper=(0.35, 0.45), smooth=1, wobble=0.08)

    # Cinched waist sash / cord belt + optional hanging knotted stole
    if f_top > f_waist + 0.04:
        t_rw, t_lw = sil_angles(f_waist)
        belt_ts = np.linspace(t_rw + 0.04, t_lw - 0.04, 18)
        b_top = [body_pt(f_waist + 0.022, t) for t in belt_ts]
        b_bot = [body_pt(f_waist - 0.022, t) for t in belt_ts]
        pen.cv.erase(b_top + b_bot[::-1])
        pen.stroke(b_top, W0 * 0.60, taper=(0.05, 0.05), wobble=0.1)
        pen.stroke(b_bot, W0 * 0.70, taper=(0.05, 0.05), wobble=0.1)
        if pose != "sword" and rng.random() < 0.55:
            st_t = math.pi * 0.5 + rng.uniform(-0.22, 0.12)
            st_l = [body_pt(f, st_t + 0.065) for f in np.linspace(f_waist - 0.01, 0.18, 8)]
            st_r = [body_pt(f, st_t - 0.065) for f in np.linspace(f_waist - 0.01, 0.20, 8)]
            stole = st_l + st_r[::-1]
            pen.cv.erase(stole)
            pen.closed(stole, W0 * 0.56, wobble=0.08)

    # Weathered stone cracks on the robe
    for _ in range(int(rng.integers(1, 3))):
        tc = rng.uniform(t_r0 + 0.45, t_l0 - 0.35)
        fc = rng.uniform(0.15, min(0.80, hat_f(tc) - 0.08))
        cx0, cy0 = body_pt(fc, tc)
        crack(pen, cx0, cy0, math.pi * 0.5 + rng.normal(0, 0.5), rng.uniform(5, 11), W0 * 0.46, branch=False)

    # Body silhouette outlines
    pen.stroke(hem_pts, W0 * 1.12, taper=(0.04, 0.04), wobble=0.14)
    pen.stroke(chaikin(left_sil, 1), W0 * 1.02, taper=(0.04, 0.04), wobble=0.16)
    pen.stroke(chaikin(right_sil, 1), W0 * 1.02, taper=(0.04, 0.04), wobble=0.16)

    if state == "broken_torso":
        pen.hatch(brk_ring, rng.uniform(12, 32), 2.4, W0 * 0.42, coverage=(0.35, 0.75))
        for _ in range(rng.integers(2, 4)):
            tb = rng.uniform(0, TAU)
            f_c = hat_f(tb)
            pen.stroke([Q(0, 0, f_c * H_body), body_pt(f_c, tb)], W0 * 0.52, taper=(0.2, 0.3), wobble=0.2)
        pen.closed(brk_ring, W0 * 0.92, wobble=0.15)

    # 8. Shoulder mantle / capelet draping smoothly over shoulders & upper chest
    if has_mantle:
        m_dip = rng.uniform(0.69, 0.76)
        t_rm, t_lm = sil_angles(0.86)
        m_ts = np.linspace(t_rm - 0.02, t_lm + 0.02, 22)

        def m_bot_f(t):
            front_w = max(0.0, math.sin(t)) ** 1.35
            return 0.85 - (0.85 - m_dip) * front_w

        def m_pt(f, t, expand=1.05):
            rx, ry = body_radii(f)
            uo = sway * math.sin(math.pi * f)
            return Q(uo + rx * expand * math.cos(t), ry * expand * math.sin(t), f * H_body)

        m_bot = [m_pt(m_bot_f(t), t, 1.06) for t in m_ts]
        m_top = [m_pt(0.98, t, 1.02) for t in m_ts[::-1]]
        mantle = m_bot + m_top
        pen.cv.erase(mantle)
        m_sh_ts = np.linspace(t_rm - 0.02, t_rm + 0.96, 10)
        m_sh = [m_pt(m_bot_f(t), t, 1.06) for t in m_sh_ts] + [m_pt(0.98, t, 1.02) for t in m_sh_ts[::-1]]
        pen.hatch(m_sh, 80, 1.65, W0 * 0.52, coverage=(0.68, 1.0))
        for tf in (t_rm + 0.55, math.pi * 0.5 - 0.15, t_lm - 0.45):
            pen.stroke([m_pt(0.95, tf, 1.03), m_pt(m_bot_f(tf) + 0.02, tf, 1.05)],
                       W0 * 0.40, taper=(0.3, 0.35), wobble=0.08)
        pen.stroke(chaikin(m_bot, 1), W0 * 0.88, taper=(0.05, 0.05), wobble=0.14)
        pen.stroke([m_bot[0], m_top[-1]], W0 * 0.92, taper=(0.05, 0.05), wobble=0.12)
        pen.stroke([m_bot[-1], m_top[0]], W0 * 0.92, taper=(0.05, 0.05), wobble=0.12)

    # 9. Head / Hood (intact) OR Jagged Broken Stone Neck Stub (headless!)
    if state == "intact":
        hw = rng.uniform(7.6, 8.7) * sw
        hh = H_head * 0.55
        bow = rng.uniform(0.4, 1.4) if pose == "relic" else rng.uniform(-0.3, 0.5)
        hc = Q(sway * 0.15, 0.4 + bow * 0.6, H_body + hh * 0.64 - bow * 0.7)
        sh_l = Q(-9.6 * sw, 0.8, H_body * 0.90)
        sh_r = Q(9.6 * sw, 0.2, H_body * 0.90)
        cowl_f = Q(0.0, 7.6 * sw, H_body * 0.84)
        _draw_hood(pen, rng, hc[0], hc[1], hw, hh, sh_l, sh_r, cowl_f,
                   style=hood_style, beard=has_beard, crown=has_crown, tilt=rng.normal(0, 0.22))
    elif state == "headless":
        # Narrow, realistic snapped stone neck stub rising cleanly out of the draped collar!
        nr = 5.4 * sw
        nh0 = b0 + H_body * 0.96
        nh1 = b0 + H_body + rng.uniform(3.2, 6.2)
        C_nk = broken_cylinder(pen, P, 0, 0, nr, nh0, nh1, jag=rng.uniform(3.0, 4.8), n=32, r_top=nr * 0.88)
        # subtle draped collar fold wrapping the base of the snapped neck
        t_rn, t_ln = sil_angles(0.98)
        collar = [body_pt(0.96, t) for t in np.linspace(t_rn + 0.15, t_ln - 0.15, 14)]
        pen.stroke(collar, W0 * 0.78, taper=(0.15, 0.15), wobble=0.12)

    # 10. Front Arms, Bell-Sleeves & Held Attributes (pose-specific, back-to-front!)
    sh_h = H_body * 0.84
    sh_u = 10.6 * sw

    if pose == "sword":
        sw_broken = rng.random() < (0.55 if state == "broken_torso" else 0.18)
        if state == "broken_torso":
            h_g = min(H_body * (f_top - 0.04), H_body * 0.56)
            p_tip = Q(0.0, 10.8 * sw, 0.8)
            p_g = Q(0.0, 8.8 * sw, h_g)
            p_pom = Q(0.0, 8.8 * sw, h_g + 9.5)
            _draw_statue_sword(pen, rng, p_tip, p_g, p_pom, broken=False)
        else:
            h_guard = H_body * rng.uniform(0.58, 0.63)
            h_pom = h_guard + rng.uniform(11.5, 13.5)
            p_tip = Q(0.0, 11.4 * sw, 0.8)
            p_guard = Q(0.0, 9.6 * sw, h_guard)
            p_pom = Q(0.0, 9.6 * sw, h_pom)
            # Draw sleeves FIRST so the broad greatsword & crossguard stand bold & unobstructed in front!
            p_grip_l = Q(-0.8, 9.8 * sw, h_guard + 6.8)
            p_grip_r = Q(0.8, 9.8 * sw, h_guard + 4.4)
            el_l = Q(-sh_u - 0.8, 4.8, H_body * 0.67)
            el_r = Q(sh_u + 0.8, 4.6, H_body * 0.66)
            _draw_sleeve(pen, Q(-sh_u, 1.2, sh_h), el_l, p_grip_l, 10.5 * sw, side=-1, outward=False)
            _draw_sleeve(pen, Q(sh_u, 0.8, sh_h), el_r, p_grip_r, 11.0 * sw, side=1, outward=False)
            _draw_statue_sword(pen, rng, p_tip, p_guard, p_pom, broken=sw_broken)
            # Clasp both sculpted hands cleanly OVER the wrapped grip between guard and pommel
            _draw_hand(pen, p_grip_l, r=2.0, ang=0.05, kind="grip")
            _draw_hand(pen, p_grip_r, r=2.0, ang=-0.05, kind="grip")

    elif pose == "staff" and state != "broken_torso":
        s = staff_side
        # Off-arm draped gracefully across the chest
        o_sh = Q(-s * sh_u, 1.2, sh_h)
        o_el = Q(-s * (sh_u + 0.5), 4.4, H_body * 0.66)
        o_wr = Q(-s * 1.4, 8.6 * sw, H_body * 0.67)
        _draw_sleeve(pen, o_sh, o_el, o_wr, 10.5 * sw, side=-s, outward=False)
        if rng.random() < 0.30:
            _draw_orb_relic(pen, o_wr[0] + s * 1.6, o_wr[1] - 1.2, r=3.0)
        _draw_hand(pen, o_wr, r=2.0, ang=s * 0.2, kind="cup")

        # Tall carved staff planted on the plinth with generous clearance from the hood
        st_u = s * (sh_u + 4.5 * sw)
        st_v = 4.2 * sw
        h_grip = H_body * rng.uniform(0.66, 0.72)
        finial = str(rng.choice(["crook", "ring", "crescent", "lantern", "orb", "snapped"],
                                p=[0.22, 0.18, 0.18, 0.18, 0.14, 0.10]))
        if finial == "snapped":
            h_st_top = h_grip + rng.uniform(13.0, 21.0)
        else:
            h_st_top = H_body + H_head * rng.uniform(0.95, 1.32)
        tilt_u = s * rng.uniform(0.6, 2.2)
        p_bot = Q(st_u - tilt_u * 0.3, st_v, 0.8)
        p_top = Q(st_u + tilt_u, st_v, h_st_top)
        t_g = (h_grip - 0.8) / max(1e-6, h_st_top - 0.8)
        p_hand = (p_bot[0] + (p_top[0] - p_bot[0]) * t_g, p_bot[1] + (p_top[1] - p_bot[1]) * t_g)

        _draw_statue_staff(pen, rng, p_bot, p_top, s, finial=finial)

        # Outward-reaching staff sleeve & gripping hand
        s_sh = Q(s * sh_u, 1.0, sh_h)
        s_el = Q(s * (sh_u + 2.2), 3.4, H_body * 0.65)
        _draw_sleeve(pen, s_sh, s_el, p_hand, 12.0 * sw, side=s, outward=True)
        _draw_hand(pen, p_hand, r=2.1, ang=-s * 0.1, kind="grip")

    elif pose == "relic" and state != "broken_torso":
        relic_kind = str(rng.choice(["orb", "tome", "prayer"], p=[0.36, 0.30, 0.34]))
        h_ha = H_body * rng.uniform(0.63, 0.70)
        el_l = Q(-sh_u - 1.0, 4.8, H_body * 0.63)
        el_r = Q(sh_u + 1.0, 4.6, H_body * 0.62)
        if relic_kind == "orb":
            wr_l = Q(-3.4, 9.0 * sw, h_ha)
            wr_r = Q(3.4, 9.2 * sw, h_ha - 0.8)
            orb_p = Q(0.0, 9.4 * sw, h_ha + 2.6)
            _draw_sleeve(pen, Q(-sh_u, 1.2, sh_h), el_l, wr_l, 12.0 * sw, side=-1, outward=False)
            _draw_hand(pen, wr_l, r=2.0, ang=0.35, kind="cup")
            _draw_orb_relic(pen, orb_p[0], orb_p[1], r=rng.uniform(4.4, 5.3))
            _draw_sleeve(pen, Q(sh_u, 0.8, sh_h), el_r, wr_r, 12.5 * sw, side=1, outward=False)
            _draw_hand(pen, wr_r, r=2.0, ang=-0.35, kind="cup")
        elif relic_kind == "tome":
            wr_l = Q(-3.6, 8.8 * sw, h_ha - 0.5)
            wr_r = Q(3.6, 9.0 * sw, h_ha - 1.0)
            tm_p = Q(0.0, 9.6 * sw, h_ha + 1.8)
            _draw_sleeve(pen, Q(-sh_u, 1.2, sh_h), el_l, wr_l, 11.5 * sw, side=-1, outward=False)
            _draw_sleeve(pen, Q(sh_u, 0.8, sh_h), el_r, wr_r, 12.0 * sw, side=1, outward=False)
            _draw_tome(pen, tm_p[0], tm_p[1], w=rng.uniform(8.8, 10.4), h=rng.uniform(7.2, 8.4))
            _draw_hand(pen, (tm_p[0] - 4.4, tm_p[1] + 1.6), r=1.9, ang=0.25, kind="cup")
            _draw_hand(pen, (tm_p[0] + 4.4, tm_p[1] + 1.2), r=1.9, ang=-0.25, kind="cup")
        else:  # solemn steepled prayer hands
            wr_l = Q(-1.8, 9.0 * sw, h_ha + 0.8)
            wr_r = Q(1.8, 9.2 * sw, h_ha + 0.4)
            _draw_sleeve(pen, Q(-sh_u, 1.2, sh_h), el_l, wr_l, 13.5 * sw, side=-1, outward=False)
            _draw_sleeve(pen, Q(sh_u, 0.8, sh_h), el_r, wr_r, 14.0 * sw, side=1, outward=False)
            clasp = ((wr_l[0] + wr_r[0]) * 0.5, (wr_l[1] + wr_r[1]) * 0.5 - 1.6)
            _draw_hand(pen, clasp, r=2.5, ang=0.0, kind="prayer")

    elif pose == "raised" and state != "broken_torso":
        s = raised_side
        o_sh = Q(-s * sh_u, 1.2, sh_h)
        o_el = Q(-s * (sh_u + 0.8), 4.4, H_body * 0.65)
        o_wr = Q(-s * 1.4, 8.6 * sw, H_body * 0.63)
        _draw_sleeve(pen, o_sh, o_el, o_wr, 11.0 * sw, side=-s, outward=False)
        _draw_hand(pen, o_wr, r=2.0, ang=s * 0.25, kind="cup")
        if s > 0:
            draw_raised_arm(1)

    # 11. Optional creeping ivy climbing the outer plinth corner & flank of the lower robe
    if has_vines:
        v_side = -1 if rng.random() < 0.65 else 1
        pts = [P(v_side * (pw * 0.82 - k * 0.9), -pw * 0.65 + k * 0.7, k * 2.6) for k in range(5)]
        t_v0 = t_l0 - 0.35 if v_side < 0 else t_r0 + 0.45
        for k in range(int(rng.integers(4, 7))):
            pts.append(body_pt(0.02 + k * 0.06, t_v0 + v_side * k * 0.05 + math.sin(k * 1.2) * 0.08))
        pen.stroke(pts, W0 * 0.52, taper=(0.05, 0.7), smooth=2, wobble=0.1)
        for j, q in enumerate(pts[1:-1]):
            sgn = 1 if j % 2 else -1
            pen.dab(q[0] + sgn * 1.8, q[1], rng.uniform(1.5, 2.1), angle=sgn * 0.45, w=W0 * 0.68)

    # 12. Fallen sculpted fragments (headless / broken_torso), cleanly south of plinth corners
    avoid_boxes = [(-pw * 1.16, -pw * 1.16, pw * 1.16, pw * 1.16)] if plinth_kind == "square" else []
    avoid_circles = [] if plinth_kind == "square" else [(0.0, 0.0, pw * 1.16)]
    if state in ("headless", "broken_torso") and rng.random() < 0.85:
        fx = rng.uniform(-pw * 1.15, pw * 1.15)
        fy = -pw * 1.15 - rng.uniform(12.0, 18.0)
        hx, hy = P(fx, fy)
        avoid_circles.append((fx, fy, 13.5))
        _draw_fallen_head(pen, rng, hx, hy - 4.0, tilt=rng.choice([-1, 1]) * rng.uniform(0.45, 1.2))
        if state == "broken_torso":
            bx = -fx * rng.uniform(0.55, 0.95) + rng.uniform(-6, 6)
            by = -pw * 1.15 - rng.uniform(14.0, 20.0)
            tr = rng.uniform(5.0, 6.6)
            lying_cylinder(pen, P, bx - 7.5, bx + 7.5, by, tr)
            avoid_boxes.append((bx - 9.5, by - tr - 2.0, bx + 9.5, by + tr + 2.0))

    # 13. Front ground scatter strictly outside plinth footprint & fallen fragments
    _scatter(pen, P, rng, 0, 0, pw * 1.42, pw * 1.42, n=(4, 9),
             avoid_boxes=avoid_boxes, avoid_circles=avoid_circles)


def gods_sword(pen, P, rng, in_stone=None):
    """Monumental divine greatsword driven into a faceted boulder or shattered earth crater."""
    W0, cv = pen.width, pen.cv
    in_stone = rng.random() < 0.45 if in_stone is None else in_stone
    gx, gy = P.ox, P.oy
    if in_stone:  # Craggy faceted stone boulder split by the blade!
        bw = rng.uniform(58, 76)
        bh = rng.uniform(32, 44)
        b_pts = hill_pts(rng, gx, gy + 4, bw, bh, m=7)
        rock(pen, b_pts, base_sag=rng.uniform(4, 7))
        gy = gy - bh * 0.65
        for _ in range(rng.integers(3, 6)):
            t = rng.uniform(0.15, math.pi - 0.15)
            crack(pen, gx + math.cos(t) * 3, gy + 2, t, rng.uniform(10, 20), W0 * 0.65, branch=False)

    a = rng.normal(0, 0.15)
    d = np.array([math.sin(a), -math.cos(a)])
    nrm = np.array([-d[1], d[0]])
    g = np.array([gx, gy])
    Lb, wb = rng.uniform(66, 96) * (0.82 if in_stone else 1.0), rng.uniform(9.5, 13.5)
    if not in_stone:  # Shattered impact crater & radiating ground fissures
        for _ in range(rng.integers(6, 11)):
            t = rng.uniform(0, TAU)
            crack(pen, gx + math.cos(t) * 9, gy + math.sin(t) * 3.5, t, rng.uniform(18, 42), W0 * 0.78)
        pen.stroke(ellipse_pts(gx, gy, rng.uniform(17, 23), rng.uniform(6, 9.5), 24, math.pi, TAU),
                   W0 * 0.9, taper=(0.3, 0.3))

    gc = g + d * Lb
    blade = [g + nrm * wb * 0.40, gc + nrm * wb * 0.52, gc - nrm * wb * 0.52, g - nrm * wb * 0.40]
    cv.erase(blade)
    pen.hatch(clip_halfplane(blade, -nrm, float((g - nrm * 0.15) @ -nrm)), 80, 1.5, W0 * 0.58, coverage=(0.72, 1.0))
    pen.stroke([g + d * 4, gc - d * 3], W0 * 0.72, taper=(0.4, 0.1), wobble=0)  # fuller
    pen.closed(blade, W0 * 1.1, wobble=0.14)
    for _ in range(rng.integers(0, 4)):  # battle nicks along the edge
        f = rng.uniform(0.15, 0.85)
        side = rng.choice([-1, 1])
        p = g + (gc - g) * f + nrm * side * (wb * (0.40 + 0.12 * f))
        cv.erase([p + d * 1.2, p - d * 1.2, p - nrm * side * rng.uniform(1.2, 2.4)])
    if rng.random() < 0.65:  # ancient runes engraved down the fuller
        for f in np.linspace(0.28, 0.84, int(rng.integers(3, 6))):
            p = g + (gc - g) * f + nrm * 2.4
            _glyph(pen, p[0], p[1], 2.7, str(rng.choice(GLYPHS)))
    if rng.random() < 0.32:  # weathered moss/ivy tendril winding up the base of the ancient blade
        vpts = [g + d * (k * 4.5) + nrm * (math.sin(k * 1.1) * wb * 0.45) for k in range(int(rng.integers(4, 8)))]
        pen.stroke(vpts, W0 * 0.52, taper=(0.05, 0.7), smooth=1, wobble=0.1)
        for q in vpts[1::2]:
            pen.dab(q[0] + rng.choice([-1.5, 1.5]), q[1], 1.4, w=W0 * 0.65)

    # Varied sculpted crossguard (classic straight, winged/flared gothic, or arched)
    gl, gt = rng.uniform(36, 48), rng.uniform(4.2, 6.2)
    gstyle = str(rng.choice(["classic", "winged", "arched"]))
    if gstyle == "winged":
        guard = [
            gc + nrm * gl * 0.52 - d * 4.5, gc + nrm * gl * 0.44 + d * (gt + 1.5),
            gc + nrm * 6.0 + d * (gt * 1.1), gc + d * (gt * 1.45),
            gc - nrm * 6.0 + d * (gt * 1.1), gc - nrm * gl * 0.44 + d * (gt + 1.5),
            gc - nrm * gl * 0.52 - d * 4.5, gc - d * 2.2,
        ]
    else:
        curve = rng.uniform(2.0, 5.0) if gstyle == "arched" else rng.uniform(-2.0, 2.0)
        guard = [gc + nrm * gl / 2 - d * curve, gc + nrm * gl / 2 + d * gt - d * curve, gc + d * gt * 1.25,
                 gc - nrm * gl / 2 + d * gt - d * curve, gc - nrm * gl / 2 - d * curve, gc - d * 1.8]
    cv.erase(guard)
    pen.hatch(clip_halfplane(guard, -nrm, float(gc @ -nrm)), 80, 1.45, W0 * 0.58)
    pen.closed(guard, W0 * 1.0)
    # Central crossguard jewel / medallion boss
    boss = ellipse_pts(gc[0] + d[0] * gt * 0.45, gc[1] + d[1] * gt * 0.45, 2.8, 2.8, 12)
    cv.erase(boss)
    pen.closed(boss, W0 * 0.68, wobble=0)

    # Cord-wrapped grip & varied pommel (round, diamond, or ringed)
    gL = rng.uniform(16, 22)
    g0, g1 = gc + d * gt * 1.2, gc + d * (gt * 1.2 + gL)
    grip = [g0 + nrm * 2.7, g1 + nrm * 2.2, g1 - nrm * 2.2, g0 - nrm * 2.7]
    cv.erase(grip)
    for f in np.linspace(0.1, 0.9, 6):
        p = g0 + (g1 - g0) * f
        pen.stroke([p + nrm * 2.5, p - nrm * 2.5 + d * 1.5], W0 * 0.55, taper=(0, 0), wobble=0)
    pen.closed(grip, W0 * 0.9)

    pc = g1 + d * 4.2
    pstyle = str(rng.choice(["round", "diamond", "ring"]))
    if pstyle == "diamond":
        pom = [pc + d * 5.8, pc + nrm * 5.0, pc - d * 4.5, pc - nrm * 5.0]
        cv.erase(pom)
        pen.hatch(clip_halfplane(pom, -nrm, float(pc @ -nrm)), 80, 1.3, W0 * 0.48)
        pen.closed(pom, W0 * 0.98, wobble=0.05)
    elif pstyle == "ring":
        pom = ellipse_pts(pc[0], pc[1], 5.6, 5.6, 18)
        cv.erase(pom)
        cv.clear(ellipse_pts(pc[0], pc[1], 2.5, 2.5, 14))
        pen.closed(pom, W0 * 0.95, wobble=0)
        pen.closed(ellipse_pts(pc[0], pc[1], 2.5, 2.5, 14), W0 * 0.72, wobble=0)
    else:
        pom = ellipse_pts(pc[0], pc[1], 5.2, 5.2, 16)
        cv.erase(pom)
        pen.closed(pom, W0 * 1.0, wobble=0)
        pen.blot(ellipse_pts(pc[0], pc[1], 1.7, 1.7, 10), jitter=0.05)

    if rng.random() < 0.5:  # divine radiance rays
        for t in np.linspace(0, TAU, 16, endpoint=False) + rng.uniform(0, 0.4):
            r0, r1 = rng.uniform(12, 16), rng.uniform(20, 30)
            pen.stroke([pc + np.array([math.cos(t), math.sin(t)]) * r0,
                        pc + np.array([math.cos(t), math.sin(t)]) * r1],
                       W0 * 0.44, taper=(0.5, 0.5), wobble=0)
    if in_stone:
        pen.blot([g + nrm * wb * 0.65, g - nrm * wb * 0.65,
                  g - nrm * wb * 0.45 - d * 2.0, g + nrm * wb * 0.45 - d * 2.0], jitter=0.2)
        for _ in range(rng.integers(2, 5)):
            pen.tuft(gx + rng.uniform(-42, 42), P.oy + rng.uniform(1, 9), rng.uniform(3.5, 5.5))
        return
    for _ in range(rng.integers(5, 10)):
        t = rng.uniform(0.08, math.pi - 0.08)
        pen.stone(gx + math.cos(t) * rng.uniform(9, 24), gy + math.sin(t) * rng.uniform(3, 10), rng.uniform(1.6, 3.8))
    for _ in range(rng.integers(2, 5)):
        pen.tuft(gx + rng.uniform(-42, 42), gy + rng.uniform(4, 14), rng.uniform(3.5, 5.5))


GLYPHS = ["circle", "eye", "bar", "tri", "wave", "cross"]


def _glyph(pen, x, y, s, kind):
    W0 = pen.width * 0.45
    if kind == "circle":
        pen.closed(ellipse_pts(x, y, s * 0.4, s * 0.4, 10), W0, wobble=0)
    elif kind == "eye":
        pen.closed(ellipse_pts(x, y, s * 0.5, s * 0.25, 12), W0, wobble=0)
        pen.cv.disc(x, y, s * 0.12)
    elif kind == "bar":
        pen.stroke([(x - s * 0.4, y), (x + s * 0.4, y)], W0, taper=(0, 0), wobble=0)
    elif kind == "tri":
        pen.closed([(x, y - s * 0.4), (x + s * 0.4, y + s * 0.35), (x - s * 0.4, y + s * 0.35)], W0, wobble=0)
    elif kind == "wave":
        pen.stroke([(x - s * 0.5, y), (x - s * 0.25, y - s * 0.2), (x, y), (x + s * 0.25, y + s * 0.2), (x + s * 0.5, y)],
                   W0, taper=(0, 0), smooth=1, wobble=0)
    else:
        pen.stroke([(x, y - s * 0.4), (x, y + s * 0.4)], W0, taper=(0, 0), wobble=0)
        pen.stroke([(x - s * 0.3, y - s * 0.1), (x + s * 0.3, y - s * 0.1)], W0, taper=(0, 0), wobble=0)


def _fallen_obelisk_tip(pen, P, rng, x0, y0, L, w, dir_x=1):
    """Tapered broken obelisk upper shaft with its pointed pyramidion tip lying on the ground!"""
    W0 = pen.width
    # Lying along x axis, tapering from broken base at x0 to pyramidion point at x0 + dir_x * (L + tip_L)
    wt = w * 0.68
    x_e = x0 + dir_x * L
    x_tip = x_e + dir_x * (wt * 1.45)
    # Main lying shaft faces (front south face + top face + broken base end)
    xa, xb = (x0, x_e) if dir_x > 0 else (x_e, x0)
    wa, wb = (w, wt) if dir_x > 0 else (wt, w)
    south = [P(xa, y0, 0), P(xb, y0, 0), P(xb, y0, wb), P(xa, y0, wa)]
    topf = [P(xa, y0, wa), P(xb, y0, wb), P(xb, y0 + wb, wb), P(xa, y0 + wa, wa)]
    # Pointed pyramidion faces meeting at x_tip
    apex = P(x_tip, y0 + wt * 0.5, wt * 0.5)
    p_front = [P(x_e, y0, 0), P(x_e, y0, wt), apex]
    p_top = [P(x_e, y0, wt), P(x_e, y0 + wt, wt), apex]
    for f in (topf, south, p_top, p_front):
        pen.cv.erase(f)
    pen.hatch(south, 14, 1.6, W0 * 0.52, coverage=(0.6, 0.95))
    pen.hatch(p_front, 20, 1.5, W0 * 0.55, coverage=(0.65, 1.0))
    for f in (topf, south, p_top, p_front):
        pen.closed(f, W0 * 0.82, wobble=0.14)
    # Carved glyph on the fallen upper shaft
    _glyph(pen, *P((x0 + x_e) * 0.5, y0 + (w + wt) * 0.25, (w + wt) * 0.5), 3.4, str(rng.choice(GLYPHS)))


def obelisk(pen, P, rng):
    """Ancient inscribed monolith obelisk: stepped plinth, framed glyph cartouche, pyramidion or fractured tip."""
    W0 = pen.width
    s = rng.uniform(7.5, 10.5)
    H = rng.uniform(82, 118)
    box(pen, P, -s * 2.25, -s * 2.25, s * 2.25, s * 2.25, 0, 4.5, inset=0.4)
    box(pen, P, -s * 1.65, -s * 1.65, s * 1.65, s * 1.65, 4.5, 10.0, inset=0.5)
    broken = rng.random() < 0.42
    top = H * rng.uniform(0.46, 0.74) if broken else H
    ins = s * 0.35 * (top - 10) / (H - 10)

    if broken:
        # Jagged slanted stone break on top (NO flat solid black tabletop!)
        tilt = rng.uniform(-3.8, 3.8)
        t_sw = top + tilt * 0.6 + rng.uniform(-1.5, 1.5)
        t_se = top - tilt * 0.5 + rng.uniform(-1.5, 1.5)
        t_ne = top - tilt * 0.8 + rng.uniform(-1.5, 1.5)
        t_nw = top + tilt * 0.4 + rng.uniform(-1.5, 1.5)
        si = s - ins
        south = [P(-s, -s, 10), P(s, -s, 10), P(si, -si, t_se),
                 P(0, -si, (t_sw + t_se) * 0.5 + rng.uniform(-2.0, 2.2)), P(-si, -si, t_sw)]
        east = [P(s, -s, 10), P(s, s, 10), P(si, si, t_ne),
                P(si, 0, (t_se + t_ne) * 0.5 + rng.uniform(-1.8, 1.8)), P(si, -si, t_se)]
        topf = [P(-si, -si, t_sw), south[3], P(si, -si, t_se), east[3], P(si, si, t_ne), P(-si, si, t_nw)]
        for f in (south, east, topf):
            pen.cv.erase(f)
        pen.hatch(east, 82, 1.55, W0 * 0.7, coverage=(0.75, 1.0))
        pen.hatch(topf, rng.uniform(14, 34), 2.3, W0 * 0.42, coverage=(0.35, 0.75))
        pen.stroke([P(-si * 0.6, -si * 0.4, (t_sw + t_nw) * 0.5), P(si * 0.5, si * 0.4, (t_se + t_ne) * 0.5)],
                   W0 * 0.48, taper=(0.2, 0.2), wobble=0.2)
        for f in (south, east, topf):
            pen.closed(f, W0 * 0.86, overshoot=0.3, wobble=0.18)
    else:
        box(pen, P, -s, -s, s, s, 10, top, inset=ins)
        si = s - ins
        pyramid(pen, P, -si, -si, si, si, top, top + si * 2.0)

    # Subtle carved vertical cartouche border framing the glyph column on the south face
    if top > 28:
        h_lo, h_hi = 15.0, top - 6.0
        for sgn in (-1, 1):
            f0, f1 = (h_lo - 10) / (H - 10), (h_hi - 10) / (H - 10)
            w0, w1 = s * (1 - 0.35 * f0) * 0.62, s * (1 - 0.35 * f1) * 0.62
            pen.stroke([P(sgn * w0, -s * (1 - 0.35 * f0), h_lo), P(sgn * w1, -s * (1 - 0.35 * f1), h_hi)],
                       W0 * 0.36, taper=(0.1, 0.1), wobble=0.1)

    for _ in range(rng.integers(1, 3)):
        x, y = P(rng.uniform(-s * 0.5, s * 0.5), -s + 0.5, rng.uniform(16, top - 5))
        crack(pen, x, y, math.pi / 2 + rng.normal(0, 0.45), rng.uniform(8, 16), W0 * 0.5)

    if rng.random() < 0.45:  # vines creeping up from the plinth
        for side in (-1, 1)[: int(rng.integers(1, 3))]:
            x0 = side * s * rng.uniform(0.35, 0.85)
            pts = [P(x0 + math.sin(k * 0.9) * 1.4, -s - 0.3, 10 + k * 3.2) for k in range(int(rng.integers(4, 10)))]
            pen.stroke(pts, W0 * 0.56, taper=(0.05, 0.65), smooth=1)
            for q in pts[1::2]:
                pen.dab(q[0] + rng.choice([-1.5, 1.5]), q[1], 1.4, w=W0 * 0.68)

    kinds = rng.choice(GLYPHS, size=max(1, int((top - 22) / 7)))
    for i, k in enumerate(kinds):
        h = 17 + i * 7
        frac = (h - 10) / (H - 10)
        x, y = P(0, -s + s * 0.35 * frac, h)
        _glyph(pen, x, y, 4.4 * (1 - 0.28 * frac), k)

    avoid_boxes = [(-s * 2.38, -s * 2.38, s * 2.38, s * 2.38)]
    if broken:
        # Fallen pointed pyramidion tip lying cleanly in front of the plinth (never overlapping steps!)
        L = max(14.0, (H - top) * 0.72)
        w_tip = (s - ins) * 1.45
        dir_x = 1 if rng.random() < 0.5 else -1
        fx = dir_x * rng.uniform(s * 1.2, s * 2.2) - (L * 0.35 if dir_x > 0 else -L * 0.35)
        fy = -rng.uniform(s * 4.5, s * 5.6)
        _fallen_obelisk_tip(pen, P, rng, fx, fy, L, w_tip, dir_x=dir_x)
        x_tip = fx + dir_x * (L + w_tip * 1.1)
        avoid_boxes.append((min(fx, x_tip) - 2, fy - 2, max(fx, x_tip) + 2, fy + w_tip + 3))

    _scatter(pen, P, rng, 0, 0, s * 3.5, s * 3.5, n=(3, 7), avoid_boxes=avoid_boxes)
