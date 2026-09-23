import numpy as np
from src.utils.statistics import grouped_bootstrap_ci, paired_grouped_bootstrap_difference
from src.utils.reproducibility import set_global_seed

def test_grouped_bootstrap_is_reproducible():
    x=np.arange(12,dtype=float); g=np.repeat(['a','b','c','d'],3)
    a=grouped_bootstrap_ci(x,g,np.mean,n_resamples=100,seed=7)
    b=grouped_bootstrap_ci(x,g,np.mean,n_resamples=100,seed=7)
    assert a==b and a.lower<=a.estimate<=a.upper

def test_paired_difference():
    a=np.array([2,4,6,8.]); b=np.array([1,3,5,7.]); g=np.array([0,0,1,1])
    ci=paired_grouped_bootstrap_difference(a,b,g,np.mean,n_resamples=50)
    assert ci.estimate==1 and ci.lower==1 and ci.upper==1

def test_seed_controls_numpy():
    set_global_seed(42); a=np.random.rand(3)
    set_global_seed(42); b=np.random.rand(3)
    assert np.array_equal(a,b)
