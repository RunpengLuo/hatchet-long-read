import pandas as pd
import numpy as np

from hatchet.utils.additional_features import (
    sort_chroms,
    load_seg_file
)

"""
compute intersect segments
assume all segments with in A/B are non-overlapping
"""
def intersect_segments(segments_A: pd.DataFrame, segments_B: pd.DataFrame, raise_err=False):
    strict_lt_A = segments_A.START < segments_A.END
    if not strict_lt_A.all():
        if raise_err:
            raise ValueError(f"segments_A has invalid interval")
        print(f"Warning, segments_A has invalid interval, slient")
        segments_A = segments_A[strict_lt_A]

    strict_lt_B = segments_B.START < segments_B.END
    if not strict_lt_B.all():
        if raise_err:
            raise ValueError(f"segments_B has invalid interval")
        print(f"Warning, segments_B has invalid interval, slient")
        segments_B = segments_B[strict_lt_B]

    groups_a = segments_A.groupby("CHR")
    groups_b = segments_B.groupby("CHR")

    segments_C = {"CHR": [], "START": [], "END": []}

    common_chroms = list(set(groups_a.indices.keys()).intersection(set(groups_b.indices.keys())))
    for ch in sort_chroms(common_chroms):
        group_a = groups_a.get_group(ch).sort_values(by="START", ignore_index=True)
        group_b = groups_b.get_group(ch).sort_values(by="START", ignore_index=True)
        la, lb = len(group_a), len(group_b)
        if la == 0 or lb == 0:
            continue
        
        starts = []
        ends = []
        i, j = 0, 0
        while i < la:
            row_a = group_a.iloc[i]
            s1, t1 = row_a.START, row_a.END # [s1, t1)
            while j < lb:
                row_b = group_b.iloc[j]
                s2, t2 = row_b.START, row_b.END # [s2, t2)
                if t2 < s1: # [s2, t2)-[s1, t1)
                    j += 1
                    continue
                if s2 >= t1: # [s1, t1)-[s2, t2)
                    break
                # found overlap
                starts.append(max(s1, s2))
                ends.append(min(t1,t2))
                if t2 > t1:
                    break
                j += 1
            i += 1
        num_ovlp_segments = len(starts)
        segments_C["CHR"].extend([ch]*num_ovlp_segments)
        segments_C["START"].extend(starts)
        segments_C["END"].extend(ends)
    
    return pd.DataFrame(segments_C)