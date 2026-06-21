# Θ-Control: Memory as a Control Resource

## Overview

This repository contains the complete implementation of **Θ-Control**, a new control framework that transforms memory from problem to resource.

## Key Results

| Metric | PID | Θ-Control | Improvement |
|--------|-----|-----------|-------------|
| ISE | 1.19 | 0.35 | **70%** |
| Rise Time | 3.52s | 1.52s | **2.3x faster** |
| Overshoot | 14.8% | 2.8% | **81% reduction** |

**Plus**: Θ-Control stabilizes the chaotic Duffing oscillator with a single parameter!

## Quick Start

```bash
git clone https://github.com/marconi-unicamp/theta-control
cd theta-control
pip install -r requirements.txt
python codes/reproduce_figures.py