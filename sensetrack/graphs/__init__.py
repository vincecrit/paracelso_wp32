"""
SenseTrack Graphs Package

This package contains XML graph configurations for various satellite data processing workflows.
The graphs are used to define processing chains for different satellite data types:

- Sentinel-1 (S1) graphs for:
    * GRD (Ground Range Detected) data
    * SLC (Single Look Complex) data with different configurations
- Sentinel-2 (S2) graphs for:
    * Level-2A product processing
- COSMO-SkyMed graphs for:
    * SCS-B product processing

These XML files serve as processing templates that can be used with SNAP Graph Processing Tool (GPT)
for automated satellite data processing.

The package is part of the SenseTrack toolkit, which provides tools and utilities
for satellite data processing and analysis.
"""