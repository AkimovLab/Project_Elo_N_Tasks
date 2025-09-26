**ALL scripts must be in the same folder**

We essentially build a database of model problems and initial conditions 

## ========== Running calculations =========================
### Tully 1 & Tully2 ###

`run_tully1.py` - master script to run any of the following with the desired initial 
                conditions 

    * `tully1_exact.py`  -  exact
    * `tully1_fssh_g_jt.py` - FSSH with JasperTruhlar criterion
    * `tully1_fssh_g_plus.py` - FSSH with no reversal on frustrated
    * `tully1_fssh_g_minus.py` - FSSH with reversal on frustrated
    * `tully1_fssh_h_minus.py` - velocity rescaling along NACV, reversal on frustrated
    * `tully1_fssh_v_minus.py` - velocity rescaling along momenta, reversal on frustrated
    * `tully1_ida_fssh_g_minus.py` - also add IDA decoherence corrections

same with

`run_tully2.py`

and other scripts

    * `tully2_exact.py`  -  exact
    * `tully2_fssh_g_jt.py` - FSSH with JasperTruhlar criterion
    * `tully2_fssh_g_plus.py` - FSSH with no reversal on frustrated
    * `tully2_fssh_g_minus.py` - FSSH with reversal on frustrated
    * `tully2_fssh_h_minus.py` - velocity rescaling along NACV, reversal on frustrated
    * `tully2_fssh_v_minus.py` - velocity rescaling along momenta, reversal on frustrated
    * `tully2_ida_fssh_g_minus.py` - also add IDA decoherence corrections


## ============ Analysis of methods complexity ==================


commons.py - common utility functions that may be used by several other 
            scripts: plotting results of NAMD and exact calculations, 
            computing different kinds of information

main-Metrics.ipynb - compute the complexity metrics of different test cases from
            the data from fully quantum calculations

main-Error-Measures.ipynb - modeling error measures of differently-distributed
            random deviations of the pretended approximate method from
            pretended reference results. This leads to different regimes of
            the accumulated error evolution.
            This notebook also shows how the Entropies can be computed (Shannon, KL)
             and how they depend on data and metaparameters.

main-ELO.ipynb - scripts for modeling ELOs based on the known win matrices
               also includes an example of a multidimensional ELO
            

## =============== Some metrics produced by the analysis scripts ===============

`metrics_Tully_1_2_exact.csv`  - complexity metrics from exact simulations
`metrics_Tully_1_2_inf1.csv`   - type I of data to compute information/entropies 
`metrics_Tully_1_2_inf2.csv`   - type II of data to compute information/entropies
`metrics_Tully_1_2_inf3.csv`   - type III of data to compute information/entropies



## ============== Some raw scripts for testing some ides ===================
These are not user-friendly files, but may still be useful. 
It is okay if one doesn't check them

test-exact.ipynb - running exact Tully 1 simulation and plotting the corresponding results


test-integration.ipynb - testing different numerical methods of computing information
              or entropy - based on histogram, on KDE (kernel density estimator),
              and Kozachenko-Leonenko - THIS CAN BE ABANDONED NOW

test-ELO.ipynb - testing ELO simulation



