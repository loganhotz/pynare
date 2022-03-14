"""
testing first-order solutions of Models
"""
from __future__ import annotations

import unittest
import numpy as np

from pynare import Model

from rich import print



basic_nk_gy = np.asarray([
     0.076389474708802,
     0.083192982352646,
     0.116706141916225,
    -0.168056844359365,
     0.119694069273901,
     0.900000000000000,
     0.076389474708802,
     0.076389474708802,
     0.040151195613381
], dtype=float)


basic_nk_gu = np.asarray([
     0.084877194120891,
     0.092436647058495,
     0.129673491018028,
    -0.186729827065960,
     0.132993410304334,
     1.000000000000000,
     0.084877194120891,
     0.084877194120891,
     0.044612439570423
], dtype=float)


basic_nk_two_shocks_gu = np.asarray([
    [ 0.084877194120891, -0.619354838709677],
    [ 0.092436647058495,  0.625610948191593],
    [ 0.129673491018028, -0.946236559139785],
    [-0.186729827065960,  1.362580645161290],
    [ 0.132993410304334,  0.625610948191593],
    [ 1.000000000000000,                  0],
    [ 0.084877194120891, -0.619354838709677],
    [ 0.084877194120891, -0.619354838709677],
    [ 0.044612439570423, -0.035483870967742]
], dtype=float)


infl_track_gy = np.asarray([
    [0.949000000000000,  0.102552607292985],
    [0.949000000000000,  0.102552607292985],
    [0.949000000000000,  0.195529826524922],
    [0.949000000000000, -0.000000000000000],
    [                0,  0.780942261985476],
    [                0, -0.892594930894236],
    [                0, -0.119057738014524]
], dtype=float)


infl_track_gu = np.asarray([
     0.002370000000000,
     0.002370000000000,
     0.002370000000000,
     0.002370000000000,
     0.000000000000000,
    -0.000000000000000,
     0.000000000000000
], dtype=float)


four_eq_gy = np.asarray([
    [-1.564212172145183,  0.333070993389498,  0.065991018915460,  0.153979044136074],
    [-0.000000000000000, -0.238805970149254, -0.000000000000000, -0.000000000000000],
    [ 0.568593546709014, -0.069076553221190,  0.002739448655071,  0.006392046861832],
    [                 0,  0.800000000000000,                  0,                  0],
    [ 0.000000000000000,  0.000000000000000,  0.800000000000000, -0.000000000000000],
    [ 0.000000000000000, -0.000000000000000, -0.000000000000000,  0.800000000000000],
    [-0.771354844303286, -0.230255177403966,  0.009131495516902,  0.021306822872772],
    [-1.564212172145183, -0.466929006610502,  0.065991018915460,  0.153979044136074]
], dtype=float)


four_eq_gu = np.asarray([
    [ 0.004163387417369, 0.001924738051701, 0.000824887736443, -0.019552652151815],
    [-0.002985074626866, 0.000000000000000, 0.000000000000000,  0.000000000000000],
    [-0.000863456915265, 0.000079900585773, 0.000034243108188,  0.007107419333863],
    [ 0.010000000000000,                 0,                 0,                  0],
    [                 0,                 0, 0.010000000000000,                  0],
    [                 0, 0.010000000000000,                 0,                  0],
    [-0.002878189717550, 0.000266335285910, 0.000114143693961, -0.009641935553791],
    [-0.005836612582631, 0.001924738051701, 0.000824887736443, -0.019552652151815]
], dtype=float)


herbst_schorfheide_gy = np.asarray([
    [ 0.442033956116078, 0.000000000000001, 0.615207111052711],
    [ 0.000000000000000, 0.982010000000000, 0.000000000000000],
    [                 0,                 0, 0.924140000000000],
    [-0.684833732144766, 0.982010000000001, 0.915755243735346],
    [-0.806689695617668, 0.000000000000004, 1.388525220983930]
], dtype=float)


herbst_schorfheide_gu = np.asarray([
    [ 0.102726924498275, 0.000000000000001, 0.332853848471396],
    [-0.000000000000000, 0.650000000000000, 0.000000000000000],
    [-0.000000000000000, 0.000000000000000, 0.500000000000000],
    [-0.159152621925346, 0.650000000000001, 0.495463481580359],
    [-0.187471460752421, 0.000000000000002, 0.751252635414509]
], dtype=float)



class TestBasicNK(unittest.TestCase):
    """
    one state variable & one stoch
    """

    sol = Model.from_path('tests/models/basic_nk.txt').solve(order=1)

    def test_gy(self):
        self.assertTrue(np.allclose(self.sol.arrays.gy, basic_nk_gy))

    def test_gu(self):
        self.assertTrue(np.allclose(self.sol.arrays.gu, basic_nk_gu))



class TestBasicNKTwoShocks(unittest.TestCase):
    """
    one state variable & many stochs
    """

    sol = Model.from_path('tests/models/basic_nk_two_shocks.txt').solve(order=1)

    def test_gy(self):
        self.assertTrue(np.allclose(self.sol.arrays.gy, basic_nk_gy))

    def test_gu(self):
        self.assertTrue(np.allclose(self.sol.arrays.gu, basic_nk_two_shocks_gu))



class TestInflTrackingPerfect(unittest.TestCase):
    """
    many state variables and one stoch
    """

    sol = Model.from_path('tests/models/infl_tracking_nk_perfect_policy.txt')\
        .solve(order=1)

    def test_gy(self):
        self.assertTrue(np.allclose(self.sol.arrays.gy, infl_track_gy))

    def test_gu(self):
        self.assertTrue(np.allclose(self.sol.arrays.gu, infl_track_gu))



class TestFourEquation(unittest.TestCase):
    """
    many state variables and many stochs
    """

    sol = Model.from_path('tests/models/four_eq_nk.txt').solve(order=1)

    def test_gy(self):
        # print('')
        # print(self.sol.arrays.gy)
        # print(four_eq_gy)
        # print('')
        self.assertTrue(np.allclose(self.sol.arrays.gy, four_eq_gy))

    def test_gu(self):
        self.assertTrue(np.allclose(self.sol.arrays.gu, four_eq_gu))



class TestHerbstSchorfheide(unittest.TestCase):
    """
    no static variables
    """

    sol = Model.from_path('tests/models/herbst_schorfheide.txt').solve(order=1)

    def test_gy(self):
        self.assertTrue(np.allclose(self.sol.arrays.gy, herbst_schorfheide_gy))

    def test_gu(self):
        self.assertTrue(np.allclose(self.sol.arrays.gu, herbst_schorfheide_gu))





if __name__ == '__main__':
    # python3 -m unittest tests/test_first_order_solution.py
    unitest.main()

    print(basic_nk_gy)
    print(basic_nk_gu)
    print(basic_nk_gu_two_shocks)
