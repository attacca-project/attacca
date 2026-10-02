"""Tiled /fill helper for command-built scenes."""
from __future__ import annotations


def _fill(world, x0, y0, z0, x1, y1, z1, block, MAX=28000):
    x0, x1 = sorted((int(x0), int(x1))); y0, y1 = sorted((int(y0), int(y1))); z0, z1 = sorted((int(z0), int(z1)))
    dx, dz = x1 - x0 + 1, z1 - z0 + 1
    area = dx * dz
    ystep = max(1, MAX // max(1, area))
    y = y0
    while y <= y1:
        ye = min(y1, y + ystep - 1)
        if area > MAX:
            xstep = max(1, MAX // max(1, dz))
            x = x0
            while x <= x1:
                xe = min(x1, x + xstep - 1)
                world.cmd(f"/fill {x} {y} {z0} {xe} {ye} {z1} minecraft:{block}")
                x = xe + 1
        else:
            world.cmd(f"/fill {x0} {y} {z0} {x1} {ye} {z1} minecraft:{block}")
        y = ye + 1
