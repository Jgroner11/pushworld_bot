#!/usr/bin/env bash

sbatch -J test --partition gpu run.sh main.py
