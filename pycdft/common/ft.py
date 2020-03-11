"""Following codes define 3D forward/backward FFT on a given grid.

    Forward/backward FT are defined with following conventions:
        f(G) = 1/omega * int{ f(r) exp(-iGr) dr }
        f(r) = sigma{ f(G) exp(iGr) }
"""

import numpy as np

try:
    from pyfftw.interfaces.numpy_fft import fftn, ifftn, rfftn, irfftn
except ImportError:
    from numpy.fft import fftn, ifftn, rfftn, irfftn
from numpy.fft import fftshift, ifftshift


class FFTGrid:
    def __init__(self, n1, n2, n3):
        """Grid for FFT.

        Args:
            n1, n2, n3 (int): FFT grid size (same for R and G space)
        """
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.N = n1 * n2 * n3

        # following quantities are used to deal with gamma-trick cases
        self.n1h = self.n1 // 2 + 1
        self.n2h = self.n2 // 2 + 1
        self.n3h = self.n3 // 2 + 1
        full_grid = np.zeros((self.n1, self.n2, self.n3))
        full_grid[self.n1h:, :, :] = 1
        full_grid[0, self.n2h:, :] = 1
        full_grid[0, 0, self.n3h:] = 1
        self.yzlowerplane = zip(*np.nonzero(full_grid[0, ...]))
        self.xyzlowerspace = zip(*np.nonzero(full_grid))


def ftgg(fg, source, dest, real=False):
    """Crop or pad G space function fg defined on source grid to match dest grid.

    Args:
        fg (np.ndarray): G space function. shape == (source.n1, source.n2, source.n3)
        source (FFTGrid): FFT grid on which fg is defined.
        dest (FFTGrid): FFT grid on which output is defined.
        real (bool): if True, fg is only defind on iG1 >= 0 and thus has the shape of
             (source.n1 // 2 + 1, source.n2, source.n3), and the returned array will be
             of shape (dest.n1 // 2 + 1, source.n2, source.n3)

    Returns:
        Cropped or padded G space function defined on dest grid.
    """
    if real:
        assert fg.shape == (source.n1h, source.n2, source.n3)
    else:
        assert fg.shape == (source.n1, source.n2, source.n3)

    m = np.array([source.n1, source.n2, source.n3], dtype=int)
    n = np.array([dest.n1, dest.n2, dest.n3], dtype=int)
    d = m - n

    if all(d == 0):
        return

    elif all(d > 0):
        # crop fg
        idxs = np.zeros(3, dtype=int)
        for i in range(3):
            if m[i] % 2 == 0:
                # needed ?
                idxs[i] = (d[i] - 1) // 2 + 1
            else:
                idxs[i] = d[i] // 2
        if real:
            fgnew = ifftshift(
                fftshift(fg, axes=(1, 2))[
                    0:n[0] // 2 + 1,
                    idxs[1]:idxs[1] + n[1],
                    idxs[2]:idxs[2] + n[2],
                ], axes=(1, 2)
            )
        else:
            fgnew = ifftshift(
                fftshift(fg)[
                    idxs[0]:idxs[0] + n[0],
                    idxs[1]:idxs[1] + n[1],
                    idxs[2]:idxs[2] + n[2],
                ]
            )

    elif all(d < 0):
        # pad fg
        nleft = np.zeros(3, dtype=int)
        for i in range(3):
            if m[i] % 2 == 0:
                nleft[i] = -d[i] // 2
            else:
                nleft[i] = (-d[i] - 1) // 2 + 1
        if real:
            fgnew = ifftshift(
                np.pad(
                    fftshift(fg, axes=(1, 2)),
                    (
                        (0, m[0] // 2 - n[0] // 2),
                        (nleft[1], -d[1] - nleft[1]),
                        (nleft[2], -d[2] - nleft[2])
                    ),
                    mode="constant"
                ), axes=(1, 2)
            )
        else:
            fgnew = ifftshift(
                np.pad(
                    fftshift(fg),
                    (
                        (nleft[0], -d[0] - nleft[0]),
                        (nleft[1], -d[1] - nleft[1]),
                        (nleft[2], -d[2] - nleft[2])
                    ),
                    mode="constant"
                )
            )
    else:
        raise ValueError("Unable to connect grid ({}, {}, {}) with grid ({}, {}, {})".format(
            m[0], m[1], m[2], n[0], n[1], n[2]
        ))

    if real:
        assert fgnew.shape == (dest.n1h, dest.n2, dest.n3)
    else:
        assert fgnew.shape == (dest.n1, dest.n2, dest.n3)
    return fgnew


def ftrg(fr, grid):
    """Fourier transform function fr from R space to G space.

    Args:
        fr (np.ndarray): R space function. shape == (grid.n1, grid.n2, grid.n3).
        grid (FFTGrid): FFT grid on which fr is defined.

    Returns:
        G space function.
    """
    assert fr.shape == (grid.n1, grid.n2, grid.n3)
    return (1. / grid.N) * fftn(fr)


def ftgr(fg, grid, real=False):
    """Fourier transform function fg from G space to R space.

    Args:
        fg (np.ndarray): G space function. shape == (grid.n1, grid.n2, grid.n3).
        grid (FFTGrid): FFT grid on which fg is defined.
        real (bool): if True, fg contains only iG1 >= 0 and thus has the shape
            of (grid.n1 // 2 + 1, grid.n2, grid.n3), after FT R space function is real.

    Returns:
        R space function.
    """
    if real:
        assert fg.shape == (grid.n1h, grid.n2, grid.n3)
        fgzyx = fg.T
        return grid.N * irfftn(fgzyx, s=(grid.n1, grid.n2, grid.n3)).T
    else:
        assert fg.shape == (grid.n1, grid.n2, grid.n3)
        return grid.N * ifftn(fg)


def ftrr(fr, source, dest):
    """Fourier interpolate fr from source grid to dest grid.

    Args:
        fr (np.ndarray): R space function. shape == (source.n1, source.n2, source.n3)
        source (FFTGrid): FFT grid on which fr is defined.
        dest (FFTGrid): FFT grid on which output is defined.

    Returns:
        Interpolated R space function defined on dest grid.
    """
    assert fr.shape == (source.n1, source.n2, source.n3)
    fg = ftrg(fr, grid=source)
    fgnew = ftgg(fg, source, dest)
    return ftgr(fgnew, dest)


def embedd_g(fg_arr, gvecs, grid, fill=None):
    """Embedd f(G) defined on a list of G vectors to a G space grid.

    Args:
        fg_arr (np.ndarray): f(G) defined on G vector list. shape == (ngvecs,)
        gvecs (np.ndarray): G vectors. shape == (ngvecs, 3)
        grid (FFTGrid): G space grid.
        fill: control flag for gamma-trick case. if not None, gvecs only contains half
        of G vectors and exclude iG1 < 0 or (iG1 == 0 and iG2 < 0) or (iG1 == iG2 == 0 and iG3 < 0) are not included
            fill == None: non Gamma-trick case, the returned f(G) will have
                shape of (grid.n1, grid.n2, grid.n3)
            fill == "yz": the lower yz plane is filled by symmetry, the returned f(G) will have
                shape of (grid.n1 // 2 + 1, grid.n2, grid.n3)
            fill == "xyz": the whole grid is filled by symmetry, the returned f(G) will have
                shape of (grid.n1, grid.n2, grid.n3)
    Returns:
        f(G) defined on grid.
    """
    if fill is None:
        fg = np.zeros((grid.n1, grid.n2, grid.n3), dtype=np.complex_)
        fg[gvecs[:, 0], gvecs[:, 1], gvecs[:, 2]] = fg_arr

    elif fill == "yz":
        fg = np.zeros((grid.n1h, grid.n2, grid.n3), dtype=np.complex_)
        fg[gvecs[:, 0], gvecs[:, 1], gvecs[:, 2]] = fg_arr
        for ig2, ig3 in grid.yzlowerplane:
            fg[0, ig2, ig3] = fg[0, -ig2, -ig3].conjugate()

    elif fill == "xyz":
        fg = np.zeros((grid.n1, grid.n2, grid.n3), dtype=np.complex_)
        fg[gvecs[:, 0], gvecs[:, 1], gvecs[:, 2]] = fg_arr
        for ig1, ig2, ig3 in grid.xyzlowerspace:
            fg[ig1, ig2, ig3] = fg[-ig1, -ig2, -ig3].conjugate()

    else:
        raise ValueError("fill = {}".format(fill))

    return fg
