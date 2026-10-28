#!/bin/bash
# JOB 297 = job 296 (RANDQ128 / RANDQ512 random-row control) with P=0.10 (same number of banded rows per chunk as GAINQ at p=0.10: 102 of 1024); output e57_randq_p0.10/. NOT APPROVED yet.
export P=0.10
exec bash "$(dirname "$0")/296_e57_news_randq_band_write_arms.sh" "$@"
