"""Run all stages: python run_all.py   (or a subset: python run_all.py 2 3 4)"""
import sys
import time

import stage1_simulate_registers
import stage2_build_couple_years
import stage3_aggregate
import stage4_analyse

STAGES = {"1": ("raw registers", stage1_simulate_registers.simulate),
          "2": ("couple-years", stage2_build_couple_years.build),
          "3": ("aggregate table", stage3_aggregate.aggregate),
          "4": ("models", stage4_analyse.analyse)}

if __name__ == "__main__":
    for stage_number in sys.argv[1:] or list(STAGES):
        stage_name, run_stage = STAGES[stage_number]
        start_time = time.time()
        print(f"Stage {stage_number}: {stage_name}")
        run_stage()
        print(f"  ({time.time() - start_time:.0f}s)\n")
