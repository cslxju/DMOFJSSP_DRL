import numpy as np

def EndTime_a(row, col, mch_a, dur_a, mch_T, mch_time, job_time, temp1):
    StartTime_a=max(job_time[row],mch_time[mch_a])
    job_time[row] = StartTime_a + dur_a
    mch_time[mch_a] = StartTime_a + dur_a
    mch_T[mch_a] = mch_T[mch_a] + dur_a
    temp1[row, col] = StartTime_a + dur_a

