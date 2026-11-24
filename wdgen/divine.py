"""'Godly' ruins: temples, stone circles, colossi, a god's sword, obelisks."""
import math

import numpy as np

from .ink import TAU, clip_halfplane, ellipse_pts
from .lairs import crack
from .solids import box, broken_cylinder, cylinder, lying_cylinder, pyramid


def _scatter(pen, P, rng, cx, cy, rx, ry, n=(3, 8), kinds=("tuft", "tuft", "rubble", "stone")):
    items = []
    for _ in range(rng.integers(*n)):
        t = rng.uniform(math.pi * 0.9, math.pi * 2.1) + P.az  # mostly in front
        d = rng.uniform(1.0, 1.35)
        x, y = cx + rx * d * math.cos(t), cy + ry * d * math.sin(t)
        items.append((P.depth(x, y), str(rng.choice(kinds)), (x, y)))
    for _, kind, (x, y) in sorted(items, key=lambda it: -it[0]):
        sx, sy = P(x, y)
        if kind == "tuft":
            pen.tuft(sx, sy, rng.uniform(3.5, 6))
        elif kind == "rubble":
            pen.rubble(sx, sy, rng.uniform(5, 10), rng.uniform(2.5, 4))
        else:
            pen.stone(sx, sy, rng.uniform(1.5, 3.2))


# --------------------------------------------------------------------------- temple

def temple(pen, P, rng):
    W0 = pen.width
    W, D = rng.uniform(100, 140), rng.uniform(55, 75)
    x0, y0 = -W / 2, -D / 2
    steps, sh = int(rng.integers(2, 4)), rng.uniform(2.5, 3.5)
    for i in range(steps):
        ins = i * 3
        box(pen, P, x0 + ins, y0 + ins, x0 + W - ins, y0 + D - ins, i * sh, (i + 1) * sh)
    top = steps * sh
    m = steps * 3 + 5
    cr, Hc = rng.uniform(3.2, 4.2), rng.uniform(34, 48)
    sev = rng.uniform(0.25, 0.85)
    nx = max(3, int((W - 2 * m) / rng.uniform(11, 14)))
    ny = max(2, int((D - 2 * m) / rng.uniform(12, 15)))
    gx = np.linspace(x0 + m, x0 + W - m, nx + 1)
    gy = np.linspace(y0 + m, y0 + D - m, ny + 1)
    cols = {}
    for i, x in enumerate(gx):
        for j, y in enumerate(gy):
            if 0 < i < nx and 0 < j < ny:
                continue  # perimeter colonnade only
            r = rng.random()
            state = "intact" if r > sev else ("broken" if r > sev * 0.45 else "stump")
            cols[(i, j)] = (x, y, state, Hc * rng.uniform(0.25, 0.8) if state == "broken" else Hc if state == "intact" else rng.uniform(2, 6))
    items = []
    for (i, j), (x, y, state, h) in cols.items():
        items.append((P.depth(x, y), "col", (x, y, state, h)))
    for row in (0, ny):  # architraves along front and back rows
        i = 0
        while i < nx:
            a, b = cols.get((i, row)), cols.get((i + 1, row))
            if a and b and a[2] == "intact" and b[2] == "intact" and rng.random() < 0.75:
                j = i + 1
                while j < nx and cols[(j + 1, row)][2] == "intact" and rng.random() < 0.7:
                    j += 1
                xa, xb = gx[i], gx[j]
                items.append((min(P.depth(xa, gy[row]), P.depth(xb, gy[row])) - 0.01, "beam", (xa, xb, gy[row])))
                i = j
            i += 1
    for _ in range(rng.integers(1, 4)):  # fallen drums on the platform / ground
        x = rng.uniform(x0 + 8, x0 + W - 20)
        y = rng.uniform(y0 - 14, y0 + D - m) if rng.random() < 0.5 else y0 - rng.uniform(8, 16)
        on = y > y0 + 3
        items.append((P.depth(x, y), "drum", (x, y, top if on else 0)))

    for _, kind, args in sorted(items, key=lambda it: -it[0]):
        if kind == "col":
            x, y, state, h = args
            cylinder(pen, P, x, y, cr * 1.35, top, top + 1.8)  # base
            if state == "intact":
                C = cylinder(pen, P, x, y, cr, top + 1.8, top + h, top=False)
                for k in (-0.5, 0.0, 0.5):
                    t = P.az + 1.5 * math.pi + k
                    pen.stroke([C(t, top + 3), C(t, top + h - 1)], W0 * 0.35, taper=(0.1, 0.1), wobble=0)
                box(pen, P, x - cr * 1.4, y - cr * 1.4, x + cr * 1.4, y + cr * 1.4, top + h, top + h + 2.6)
            else:
                broken_cylinder(pen, P, x, y, cr, top + 1.8, top + h, jag=min(5, h * 0.4))
        elif kind == "beam":
            xa, xb, y = args
            hb = top + Hc + 2.6
            if rng.random() < 0.3:  # snapped beam
                xb = xa + (xb - xa) * rng.uniform(0.5, 0.85)
            box(pen, P, xa - cr * 1.4, y - cr * 1.3, xb + cr * 1.4, y + cr * 1.3, hb, hb + 5)
        else:
            x, y, h0 = args
            lying_cylinder(pen, P, x, x + rng.uniform(8, 14), y, cr, h0)
    _scatter(pen, P, rng, 0, 0, W / 2, D / 2)


# --------------------------------------------------------------------------- monuments

def stone_circle(pen, P, rng):
    Rc = rng.uniform(55, 75)
    n = int(rng.integers(8, 12))
    items = []
    ts = np.linspace(0, TAU, n, endpoint=False) + rng.uniform(0, TAU)
    for t in ts:
        x, y = Rc * math.cos(t) + rng.normal(0, 2), Rc * 0.9 * math.sin(t) + rng.normal(0, 2)
        r = rng.random()
        if r < 0.12:
            continue
        items.append((P.depth(x, y), "fallen" if r < 0.3 else "stone", (x, y)))
    if rng.random() < 0.65:  # a trilithon gate at the back
        t = P.az + math.pi / 2 + rng.uniform(-0.6, 0.6)
        x, y = Rc * 0.55 * math.cos(t), Rc * 0.5 * math.sin(t)
        items.append((P.depth(x, y), "trilithon", (x, y)))
    items.append((P.depth(0, 0), "altar", (0, 0)))
    for _, kind, (x, y) in sorted(items, key=lambda it: -it[0]):
        if kind == "stone":
            w, d, h = rng.uniform(8, 12), rng.uniform(4.5, 6.5), rng.uniform(24, 38)
            box(pen, P, x - w / 2, y - d / 2, x + w / 2, y + d / 2, 0, h, inset=rng.uniform(0.6, 2.0),
                lean=(rng.normal(0, 2.5), rng.normal(0, 1.5)))
            if rng.random() < 0.5:
                pen.tuft(*P(x - w / 2, y - d / 2 - 1), rng.uniform(3.5, 5))
        elif kind == "fallen":
            L, w = rng.uniform(24, 34), rng.uniform(8, 11)
            box(pen, P, x - L / 2, y - w / 2, x + L / 2, y + w / 2, 0, rng.uniform(5, 7), inset=1.0)
        elif kind == "trilithon":
            h = rng.uniform(38, 48)
            for dx in (-10, 10):
                box(pen, P, x + dx - 4, y - 3, x + dx + 4, y + 3, 0, h, inset=0.6, lean=(rng.normal(0, 0.8), 0))
            if rng.random() < 0.8:
                box(pen, P, x - 16, y - 3.5, x + 16, y + 3.5, h, h + 6)
        else:
            box(pen, P, x - 13, y - 7, x + 13, y + 7, 0, 6)
            if rng.random() < 0.5:  # old stains on the altar
                cx, cy = P(x + rng.uniform(-6, 6), y, 6)
                pen.blot(ellipse_pts(cx, cy, rng.uniform(2, 4), rng.uniform(1, 2), 10), jitter=0.4)
    _scatter(pen, P, rng, 0, 0, Rc * 1.1, Rc, n=(3, 7), kinds=("tuft", "tuft", "stone"))


def _cracks_on(pen, P, C, rng, h_lo, h_hi, n):
    if h_hi <= h_lo:
        return
    for _ in range(n):
        t = P.az + 1.5 * math.pi + rng.uniform(-0.6, 0.6)
        x, y = C(t, rng.uniform(h_lo, h_hi))
        crack(pen, x, y, math.pi / 2 + rng.normal(0, 0.5), rng.uniform(6, 14), pen.width * 0.55, branch=False)



def _leg(pen, P, rng, lx, ly, base, h):
    """Giant stone leg: sandalled foot, ankle, calf swelling to a broken knee or thigh."""
    W0 = pen.width
    # foot: heel block plus a rounded front with toes
    box(pen, P, lx - 5, ly - 4, lx + 5, ly + 5, base, base + 5)
    cylinder(pen, P, lx, ly - 6, 5.2, base, base + 4.2)
    for k in range(4):
        tx = lx - 3.6 + k * 2.4
        a, b = P(tx, ly - 10.6, base + 2.6), P(tx, ly - 9.2, base + 4.0)
        pen.stroke([a, b], W0 * 0.45, taper=(0, 0.2), wobble=0)
    pen.beam(P(lx - 5.5, ly - 2, base + 4.5), P(lx + 5.5, ly - 2, base + 4.5), 1.0)  # sandal strap
    knee = base + 5 + min(h, rng.uniform(28, 36))
    C = broken_cylinder(pen, P, lx, ly, 4.2, base + 4.5, knee if h > knee - base else base + 5 + h,
                        jag=0.6 if h > knee - base else 5, r_top=6.6)
    _cracks_on(pen, P, C, rng, base + 10, base + h * 0.8, int(rng.integers(0, 3)))
    if h > knee - base:  # thigh above the knee, snapped
        C = broken_cylinder(pen, P, lx, ly, 6.6, knee, base + h, jag=7, r_top=8.2)
        _cracks_on(pen, P, C, rng, knee + 3, base + h - 4, int(rng.integers(0, 2)))


def statue_legs(pen, P, rng):
    """The colossus: only the legs still stand on the plinth."""
    box(pen, P, -32, -20, 32, 20, 0, 6)
    box(pen, P, -27, -16, 27, 16, 6, 14)
    spread = rng.uniform(8, 12)
    hs = sorted([rng.uniform(26, 75), rng.uniform(20, 60)], reverse=rng.random() < 0.5)
    _leg(pen, P, rng, -spread, 3, 14, hs[0])
    _leg(pen, P, rng, spread, -2, 14, hs[1])
    for _ in range(rng.integers(1, 3)):  # fallen chunks of the body
        x = rng.uniform(-70, 25)
        lying_cylinder(pen, P, x, x + rng.uniform(18, 30), rng.uniform(-48, -30), rng.uniform(6, 9))
    _scatter(pen, P, rng, 0, -10, 42, 30, n=(4, 9))


def statue_robed(pen, P, rng):
    """Hooded robed figure on a plinth, perhaps headless, holding a staff or sword."""
    W0 = pen.width
    front = P.az + 1.5 * math.pi
    box(pen, P, -20, -20, 20, 20, 0, 5)
    box(pen, P, -16, -16, 16, 16, 5, 14)
    b0 = 14
    C = broken_cylinder(pen, P, 0, 0, 10.5, b0, b0 + 30, jag=0.5, r_top=7.2)  # skirt of the robe
    for k in (-0.55, -0.2, 0.2, 0.55):  # folds
        t = front + k
        pen.stroke([C(t, b0 + 1), C(t + rng.normal(0, 0.05), b0 + 16), C(t * 1.0 + k * 0.2, b0 + 28)],
                   W0 * 0.5, taper=(0.05, 0.6), smooth=1, wobble=0)
    headless = rng.random() < 0.35
    top = b0 + 30 + 16
    C2 = broken_cylinder(pen, P, 0, 0, 7.2, b0 + 30, top, jag=5 if headless else 0.4, r_top=8.6)
    _cracks_on(pen, P, C2, rng, b0 + 32, top - 3, int(rng.integers(0, 3)))
    hand = C2(front + 0.9, b0 + 38)
    if not headless:
        hx, hy = P(0, 0, top + 6.5)
        hood = [(hx + 6.2 * math.cos(t), hy + 7.5 * math.sin(t) - 0.5) for t in np.linspace(0, TAU, 22, endpoint=False)]
        pen.cv.erase(hood)
        pen.closed(hood, W0 * 0.9)
        face = ellipse_pts(hx + 0.6, hy + 1.2, 3.6, 4.6, 16)
        pen.blot(face, jitter=0.1)  # deep hood shadow
        pen.stroke([(hx - 5.5, hy + 6), (hx - 7, hy + 9)], W0 * 0.6, taper=(0, 0.5), wobble=0)
    # the staff / sword held in front
    if rng.random() < 0.5:
        g, t = (hand[0], hand[1] + 26), (hand[0] + rng.normal(0, 1), hand[1] - 30)
        if rng.random() < 0.3:  # snapped staff
            t = (hand[0], hand[1] - 6)
        pen.beam(g, t, 2.0)
    else:
        pen.beam((hand[0], hand[1] + 2), (hand[0], hand[1] + 30), 3.4)  # sword point-down
        pen.beam((hand[0] - 5, hand[1] + 1), (hand[0] + 5, hand[1] + 1), 1.6)
    hand_blob = ellipse_pts(hand[0], hand[1], 2.6, 2.2, 12)
    pen.cv.erase(hand_blob)
    pen.closed(hand_blob, W0 * 0.7, wobble=0)
    if headless and rng.random() < 0.7:  # the head lies at its feet
        x, y = P(rng.uniform(-30, 30), -rng.uniform(26, 34))
        head = ellipse_pts(x, y - 6, 7, 6, 18)
        pen.cv.erase(head)
        pen.closed(head, W0 * 0.9)
        pen.blot(ellipse_pts(x + 1, y - 5, 4, 3.5, 12))
    _scatter(pen, P, rng, 0, 0, 30, 28, n=(3, 8))


def gods_sword(pen, P, rng, in_stone=None):
    W0, cv = pen.width, pen.cv
    in_stone = rng.random() < 0.4 if in_stone is None else in_stone
    gx, gy = P.ox, P.oy
    if in_stone:  # a great boulder holds the blade
        rr = rng.uniform(26, 34)
        rock = [(gx + rr * math.cos(t) * (1 + rng.normal(0, 0.06)), gy - rr * 0.35 + rr * 0.62 * math.sin(t) * (1 + rng.normal(0, 0.06)))
                for t in np.linspace(0, TAU, 14, endpoint=False)]
        rock = [(x, min(y, gy + 3)) for x, y in rock]
        cv.erase(rock)
        pen.hatch(clip_halfplane(rock, (1, 0), gx + rr * 0.2), 60, 2.0, W0 * 0.6)
        pen.closed(rock, W0 * 1.2)
        for _ in range(rng.integers(2, 5)):
            t = rng.uniform(0, TAU)
            crack(pen, gx + math.cos(t) * 3, gy - rr * 0.6, t, rng.uniform(8, 16), W0 * 0.7, branch=False)
        gy = gy - rr * 0.6
    a = rng.normal(0, 0.15)
    d = np.array([math.sin(a), -math.cos(a)])
    nrm = np.array([-d[1], d[0]])
    g = np.array([gx, gy])
    Lb, wb = rng.uniform(65, 95) * (0.8 if in_stone else 1), rng.uniform(9, 13)
    if not in_stone:  # crater and cracks
        for _ in range(rng.integers(5, 10)):
            t = rng.uniform(0, TAU)
            crack(pen, gx + math.cos(t) * 10, gy + math.sin(t) * 4, t, rng.uniform(18, 40), W0 * 0.8)
        pen.stroke(ellipse_pts(gx, gy, rng.uniform(16, 22), rng.uniform(6, 9), 24, math.pi, TAU), W0 * 0.9, taper=(0.3, 0.3))
    gc = g + d * Lb
    blade = [g + nrm * wb * 0.42, gc + nrm * wb / 2, gc - nrm * wb / 2, g - nrm * wb * 0.42]
    cv.erase(blade)
    pen.hatch(clip_halfplane(blade, nrm * -1, float((g - nrm * 0.2) @ -nrm)), 80, 1.5, W0 * 0.6, coverage=(0.7, 1.0))
    pen.stroke([g + d * 4, gc - d * 4], W0 * 0.7, taper=(0.4, 0.1), wobble=0)  # fuller
    pen.closed(blade, W0 * 1.1, wobble=0.15)
    for _ in range(rng.integers(0, 4)):  # nicks in the edge
        f = rng.uniform(0.15, 0.85)
        side = rng.choice([-1, 1])
        p = g + (gc - g) * f + nrm * side * (wb * (0.42 + 0.08 * f))
        cv.erase([p + d * 1.2, p - d * 1.2, p - nrm * side * rng.uniform(1.2, 2.2)])
    if rng.random() < 0.6:  # runes down the fuller
        for k, f in enumerate(np.linspace(0.3, 0.85, int(rng.integers(3, 6)))):
            p = g + (gc - g) * f + nrm * 2.4
            _glyph(pen, p[0], p[1], 2.6, str(rng.choice(GLYPHS)))
    gl, gt = rng.uniform(34, 46), rng.uniform(4, 6)
    curve = rng.uniform(-3, 3)
    guard = [gc + nrm * gl / 2 - d * curve, gc + nrm * gl / 2 + d * gt - d * curve, gc + d * gt * 1.2,
             gc - nrm * gl / 2 + d * gt - d * curve, gc - nrm * gl / 2 - d * curve, gc - d * 1.5]
    cv.erase(guard)
    pen.hatch(clip_halfplane(guard, -nrm, float(gc @ -nrm)), 80, 1.4, W0 * 0.6)
    pen.closed(guard, W0 * 1.0)
    gL = rng.uniform(16, 22)
    g0, g1 = gc + d * gt * 1.2, gc + d * (gt * 1.2 + gL)
    grip = [g0 + nrm * 2.6, g1 + nrm * 2.2, g1 - nrm * 2.2, g0 - nrm * 2.6]
    cv.erase(grip)
    for f in np.linspace(0.1, 0.9, 6):
        p = g0 + (g1 - g0) * f
        pen.stroke([p + nrm * 2.4, p - nrm * 2.4 + d * 1.5], W0 * 0.55, taper=(0, 0), wobble=0)
    pen.closed(grip, W0 * 0.9)
    pc = g1 + d * 4
    pom = ellipse_pts(pc[0], pc[1], 5, 5, 16)
    cv.erase(pom)
    pen.closed(pom, W0 * 1.0, wobble=0)
    pen.blot(ellipse_pts(pc[0], pc[1], 1.6, 1.6, 10), jitter=0.05)
    if rng.random() < 0.5:  # radiance
        for t in np.linspace(0, TAU, 16, endpoint=False) + rng.uniform(0, 0.4):
            r0, r1 = rng.uniform(12, 16), rng.uniform(20, 30)
            pen.stroke([pc + np.array([math.cos(t), math.sin(t)]) * r0, pc + np.array([math.cos(t), math.sin(t)]) * r1],
                       W0 * 0.45, taper=(0.5, 0.5), wobble=0)
    if in_stone:
        pen.blot([g + nrm * wb * 0.6, g - nrm * wb * 0.6, g - nrm * wb * 0.4 + d * -1.5, g + nrm * wb * 0.4 + d * -1.5], jitter=0.2)
        for _ in range(rng.integers(2, 5)):
            pen.tuft(gx + rng.uniform(-40, 40), P.oy + rng.uniform(0, 8), rng.uniform(3.5, 5.5))
        return
    for _ in range(rng.integers(4, 9)):
        t = rng.uniform(0.1, math.pi - 0.1)
        pen.stone(gx + math.cos(t) * rng.uniform(8, 22), gy + math.sin(t) * rng.uniform(3, 9), rng.uniform(1.5, 3.5))
    for _ in range(rng.integers(2, 5)):
        pen.tuft(gx + rng.uniform(-40, 40), gy + rng.uniform(4, 14), rng.uniform(3.5, 5.5))


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


def obelisk(pen, P, rng):
    s = rng.uniform(7, 10)
    H = rng.uniform(80, 115)
    box(pen, P, -s * 2.2, -s * 2.2, s * 2.2, s * 2.2, 0, 4)
    box(pen, P, -s * 1.6, -s * 1.6, s * 1.6, s * 1.6, 4, 9)
    broken = rng.random() < 0.4
    top = H * rng.uniform(0.45, 0.75) if broken else H
    ins = s * 0.35 * (top - 9) / (H - 9)
    box(pen, P, -s, -s, s, s, 9, top, inset=ins)
    for _ in range(rng.integers(0, 3)):
        x, y = P(rng.uniform(-s * 0.5, s * 0.5), -s + 0.5, rng.uniform(15, top - 5))
        crack(pen, x, y, math.pi / 2 + rng.normal(0, 0.5), rng.uniform(8, 18), pen.width * 0.55)
    if rng.random() < 0.4:  # vines creeping up from the base
        for side in (-1, 1)[: int(rng.integers(1, 3))]:
            x0 = side * s * rng.uniform(0.3, 0.8)
            pts = [P(x0 + math.sin(k * 0.9) * 1.5, -s - 0.3, 9 + k * 3) for k in range(int(rng.integers(4, 10)))]
            pen.stroke(pts, pen.width * 0.6, taper=(0, 0.6), smooth=1)
            for q in pts[1::2]:
                pen.dab(q[0] + rng.choice([-1.5, 1.5]), q[1], 1.4, w=pen.width * 0.7)
    kinds = rng.choice(GLYPHS, size=int((top - 20) / 7))
    for i, k in enumerate(kinds):
        h = 16 + i * 7
        frac = (h - 9) / (H - 9)
        x, y = P(0, -s + s * 0.35 * frac, h)
        _glyph(pen, x, y, 4.5 * (1 - 0.3 * frac), k)
    if broken:
        pen.blot([P(-s + ins + 0.8, -s + ins + 0.8, top), P(s - ins - 0.8, -s + ins + 0.8, top),
                  P(s - ins - 0.8, s - ins - 0.8, top), P(-s + ins + 0.8, s - ins - 0.8, top)], jitter=0.2)
        L = H - top
        x0 = rng.choice([-1, 1]) * rng.uniform(s * 3, s * 4) - (L if rng.random() < 0.5 else 0)
        y0 = rng.uniform(-s * 4, -s * 2.5)
        box(pen, P, x0, y0, x0 + L, y0 + s * 1.4, 0, s * 1.4)
    else:
        si = s - s * 0.35
        pyramid(pen, P, -si, -si, si, si, H, H + si * 1.8)
    _scatter(pen, P, rng, 0, 0, s * 3, s * 3, n=(3, 7))
