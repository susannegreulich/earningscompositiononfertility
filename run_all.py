"""Run all stages: python run_all.py   (or a subset: python run_all.py 2 3)

Stage 0 downloads the Statistics Denmark figures. It is skipped when 0dst_data/
already holds them; `python run_all.py 0` downloads them again.
"""
import sys
import time

import stage0_download_dst_data
import stage1_simulate_registers
import stage2_process_data
import stage3_analyse
from common import DST_DATA_FOLDER

STAGES = {"0": ("Statistics Denmark figures", stage0_download_dst_data.download_all),
          "1": ("raw registers", stage1_simulate_registers.simulate),
          "2": ("couple-years and event counts", stage2_process_data.process),
          "3": ("model", stage3_analyse.analyse)}

if __name__ == "__main__":
    figures_already_downloaded = any(DST_DATA_FOLDER.glob("*.csv"))
    all_stages = [number for number in STAGES if number != "0" or not figures_already_downloaded]
    # The table of the paper's Danish odds ratios needs no download, so it is
    # always rewritten; it then matches the numbers in stage 0 after any edit.
    stage0_download_dst_data.save_danish_table_a2()
    for stage_number in sys.argv[1:] or all_stages:
        stage_name, run_stage = STAGES[stage_number]
        start_time = time.time()
        print(f"Stage {stage_number}: {stage_name}")
        run_stage()
        print(f"  ({time.time() - start_time:.0f}s)\n")
