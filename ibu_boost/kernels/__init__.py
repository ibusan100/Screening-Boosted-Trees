"""Triton-accelerated kernels for Screening Boosted Trees.

These modules import `triton` at module load time, so they should only be
imported when the optional `triton` (Linux/macOS) or `triton-windows` (Windows)
dependency has been installed. Top-level `sbt` uses lazy `__getattr__` loading
to keep the NumPy reference path import-clean on machines without Triton.
"""
