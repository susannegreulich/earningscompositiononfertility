"""Run stages 1-3: python run_all.py   (or a subset: python run_all.py 2 3)

Stage 0 downloads the Statistics Denmark figures again. They are kept in 0data/,
so it only runs when asked for: python run_all.py 0
"""
import importlib
import sys
import time

# Each stage: its name, and the module and function that run it. A module is
# imported only when its stage runs, because stages 1-3 read the files in 0data/
# as soon as they are imported.
STAGES = {"0": ("Statistics Denmark figures", "stage0_data", "download_all"),
          "1": ("raw registers", "stage1_simulate_registers", "simulate"),
          "2": ("couple-years and event counts", "stage2_process_data", "process"),
          "3": ("model", "stage3_analyse", "analyse")}

if __name__ == "__main__":
    for stage_number in sys.argv[1:] or ["1", "2", "3"]:
        stage_name, module_name, function_name = STAGES[stage_number]
        start_time = time.time()
        print(f"Stage {stage_number}: {stage_name}")
        getattr(importlib.import_module(module_name), function_name)()
        print(f"  ({time.time() - start_time:.0f}s)\n")
