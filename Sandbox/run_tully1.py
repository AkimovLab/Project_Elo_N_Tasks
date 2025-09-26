import os, sys

#================= NAMD ========================

# Tully 1

cases = [ (-6.0, 20.0, 0),
          (-6.0, 20.0, 1),
          (-10.0, 20.0, 0),
          (-10.0, 20.0, 1),
          (-6.0, 10.0, 0),
          (-6.0, 10.0, 1)
        ]

for q0, p0, istate in cases:
    os.system(F"python tully1_fssh_g_plus.py --q0={q0} --p0={p0} --istate={istate}")
    os.system(F"python tully1_fssh_g_jt.py --q0={q0} --p0={p0} --istate={istate}")
    os.system(F"python tully1_fssh_g_minus.py --q0={q0} --p0={p0} --istate={istate}")
    os.system(F"python tully1_ida_fssh_g_minus.py --q0={q0} --p0={p0} --istate={istate}")
    os.system(F"python tully1_fssh_h_minus.py --q0={q0} --p0={p0} --istate={istate}")
    os.system(F"python tully1_fssh_v_minus.py --q0={q0} --p0={p0} --istate={istate}")
    os.system(F"python tully1_exact.py --q0={q0} --p0={p0} --istate={istate}")

