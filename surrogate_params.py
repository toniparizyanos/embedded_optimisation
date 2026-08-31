"""
surrogate_params.py
===================
Manufacturing economics derived from the SuperPro flowsheet surrogates.
Imported by model_CART_iSHIPMENT.py.

    from surrogate_params import *

Everything here is a plain Python value, not a Pyomo component: these
numbers are needed while Pyomo BUILDS the model (to initialise CIM/CVM
and, later, to set the OMLT block input bounds), which happens before
any Param has a value.
"""
import numpy as np
import onnx
from pathlib import Path
import tempfile
from omlt import OmltBlock
from omlt.io import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
from omlt.neuralnet import ReluBigMFormulation
from pyomo.environ import Var, Constraint, Block, value
import onnxruntime as _ort

# ---------------------------------------------------------------------
# 1. Patient biology (stochastic input)
# ---------------------------------------------------------------------
SEED     = 42          # fixed for reproducibility - record in the thesis
THETA_LO = 0.25        # lower bound of the surrogate training domain
THETA_HI = 0.60        # upper bound of the surrogate training domain


def sample_theta(patients, seed=SEED):
    """Draw one enriched T-cell fraction per patient, U(THETA_LO, THETA_HI).

    `patients` is the list of patient labels from the .dat file, so the
    sample size follows the data file automatically.
    """
    rng = np.random.default_rng(seed)
    return {p: float(v)
            for p, v in zip(patients, rng.uniform(THETA_LO, THETA_HI, len(patients)))}


# Design basis for CAPEX: the facility is sized for the highest-throughput
# batch in the operating envelope, not for the average patient.
THETA_CAPEX = THETA_HI


# ---------------------------------------------------------------------
# 1b. A&T duration (stochastic input, second biological quantity)
# ---------------------------------------------------------------------
# Sampled per patient exactly like theta, but from an INDEPENDENT RNG
# stream (TAT_SEED != SEED) so that adding this sampler does not perturb
# the theta realisation - every earlier theta-only run stays reproducible.
#
# Distribution: Triangular(1, 2, 3) days.
#   * bounded to [T_AT_LO, T_AT_HI] = the surrogate's training box, so no
#     patient ever lands outside the domain the OPEX network was trained on
#   * peaks at the DoE nominal (2.0 days), mean = 2.0
#   * same distribution family the sister thesis uses for its process-time
#     variable -> methodologically consistent on the same flowsheet
TAT_SEED = 2024


def sample_t_at(patients, seed=TAT_SEED):
    """Draw one A&T duration per patient, Triangular(1, 2, 3) days.

    Same signature and return shape as sample_theta: a dict keyed by
    patient label, so instance.t_at[p] = tat[p] works everywhere.
    """
    rng = np.random.default_rng(seed)
    vals = rng.triangular(T_AT_LO, 2.0, T_AT_HI, len(patients))
    vals = np.clip(vals, T_AT_LO, T_AT_HI)          # guard the domain edges
    return {p: float(v) for p, v in zip(patients, vals)}

# ---------------------------------------------------------------------
# 2. Viral vector supply contract (fixed price)
# ---------------------------------------------------------------------
# Nominal = dataset mean = midpoint of the sampling domain
# (1,207,423.493 - 2,463,606.386).
VV_COST = 1835514.939

# ---------------------------------------------------------------------
# 3. Cost split
# ---------------------------------------------------------------------
# PHI = share of operating cost that scales with BATCHES rather than with
# the facility.  From the SuperPro Economic Evaluation Report, annual
# operating cost $15,984,000:
#     batch-driven : utilities 13,044,000 + raw materials 642,000
#                    + lab/QC/QA 319,000 + consumables 280,000 = 14,285,000
#     facility     : facility-dependent 1,119,000 + labour 580,000
#     PHI = 14,285,000 / 15,984,000 = 0.894
PHI = 0.894

# ---------------------------------------------------------------------
# 4. Surrogate reference outputs
# ---------------------------------------------------------------------
OPEX_ref = 163406.92      # $/batch at nominal inputs

_cd    = np.load('capex_scaling.npz', allow_pickle=True)
_csess = _ort.InferenceSession('capex_surrogate.onnx')
_cxs   = np.array([[(THETA_CAPEX - _cd['x_offset'][0]) / _cd['x_factor'][0],
                    (VV_COST     - _cd['x_offset'][1]) / _cd['x_factor'][1]]])
TCI_LINE = (float(_csess.run(None, {'input': _cxs})[0].ravel()[0])
            * float(_cd['y_factor'][0]) + float(_cd['y_offset'][0]))
print(f'[surrogate_params] TCI_LINE = ${TCI_LINE:,.0f}')



WC_LINE    = 573000.0   # $ working capital per ADDITIONAL line
AMORT_DAYS = 15 * 365   # 15-year amortisation, as in the SuperPro basis
BATCH_YR   = 46         # batches per line per year at nominal batch time

# ---------------------------------------------------------------------
# 5. Surrogate 2: batch time from A&T duration (exact, R^2 = 1.0)
# ---------------------------------------------------------------------
BT_SLOPE     = 2.0      # days of batch time per day of A&T
BT_INTERCEPT = 7.4755   # days (645,887.4 s) of fixed stages
T_AT_LO      = 1.0      # days - training domain lower bound
T_AT_HI      = 3.0      # days - training domain upper bound


# ---------------------------------------------------------------------
# 6. Pyomo Param initialisers
# ---------------------------------------------------------------------
def cim_init(model, m):
    """CAPEX per facility per day. FCAP[m] is the number of lines."""
    from pyomo.environ import value
    n = value(model.FCAP[m])
    return (TCI_LINE + (n - 1) * WC_LINE) / AMORT_DAYS


def cvm_init(model, m):
    """Facility-dependent operating cost per day (the (1-PHI) share)."""
    from pyomo.environ import value
    return (1 - PHI) * OPEX_ref * BATCH_YR / 365 * value(model.FCAP[m])


# ---------------------------------------------------------------------
# 7. OPEX surrogate - OMLT block construction  (step 6b)
# ---------------------------------------------------------------------
OPEX_ONNX_PATH    = 'opex_surrogate.onnx'
OPEX_SCALING_PATH = 'opex_scaling.npz'

_od      = np.load(OPEX_SCALING_PATH, allow_pickle=True)
X_OFFSET = _od['x_offset']          # min-max: Xmin
X_FACTOR = _od['x_factor']          # min-max: Xmax - Xmin
Y_OFFSET = float(_od['y_offset'][0])   # StandardScaler: mean
Y_FACTOR = float(_od['y_factor'][0])   # StandardScaler: scale

I_THETA, I_VV, I_TAT = 0, 1, 2      # column order in input_names


def _scale_in(x, i):
    """Physical value -> the [0,1] scale the network was trained on."""
    return (x - X_OFFSET[i]) / X_FACTOR[i]


def _net_def(onnx_model, bounds, path):
    """Load a NetworkDefinition with per-patient input bounds.

    OMLT quirk: write_onnx_model_with_bounds stores bounds under integer
    keys 0,1,2 but a (1,3)-shaped ONNX graph makes the formulation look
    them up under (0,0),(0,1),(0,2).  Remap, or build_formulation raises
    KeyError.  The attribute is name-mangled, hence the long name.
    """
    write_onnx_model_with_bounds(str(path), onnx_model, bounds)
    nd = load_onnx_neural_network_with_bounds(str(path))
    keys = list(nd.layers)[0].input_indexes
    if keys:
        sib = nd.scaled_input_bounds
        remapped = {}
        for k in keys:
            scalar = k[-1] if isinstance(k, tuple) else k
            # source bounds may be keyed by the tuple k or the scalar
            src = sib[k] if k in sib else sib[scalar]
            remapped[scalar] = src
        nd._NetworkDefinition__scaled_input_bounds = remapped
    return nd, keys


def attach_opex_surrogate(instance, theta, eps=1e-6):
    """Replace the constant OPEX rule with one OMLT block per patient.

    Call AFTER create_instance().  For each patient the two fixed inputs
    (theta_p, VV_COST) are stamped as degenerate bounds, so the network's
    input box collapses to a line segment in t_at - this keeps the big-M
    values tight, which matters once there are many blocks.
    """
    instance.OPEX_def.deactivate()          # drop the constant OPEX == PHI*OPEX_REF

    onnx_model = onnx.load(OPEX_ONNX_PATH)
    tmpdir     = Path(tempfile.mkdtemp())

    instance.nn = Block(instance.p)
    idx = None
    for k, p in enumerate(instance.p):
        s_th = _scale_in(theta[p], I_THETA)
        s_vv = _scale_in(VV_COST,  I_VV)
        bounds = {I_THETA: (s_th - eps, s_th + eps),
                  I_VV:    (s_vv - eps, s_vv + eps),
                  I_TAT:   (0.0, 1.0)}
        nd, idx = _net_def(onnx_model, bounds, tmpdir / f'opex_{k}.onnx')
        instance.nn[p].net = OmltBlock()
        instance.nn[p].net.build_formulation(ReluBigMFormulation(nd))

    i_th, i_vv, i_tat = idx[I_THETA], idx[I_VV], idx[I_TAT]
    i_out = list(instance.nn[list(instance.p)[0]].net.outputs.keys())[0]

    def _lk_theta(mdl, p):
        return mdl.nn[p].net.inputs[i_th] == _scale_in(theta[p], I_THETA)
    instance.NNLINK_THETA = Constraint(instance.p, rule=_lk_theta)

    def _lk_vv(mdl, p):
        return mdl.nn[p].net.inputs[i_vv] == _scale_in(VV_COST, I_VV)
    instance.NNLINK_VV = Constraint(instance.p, rule=_lk_vv)

    def _lk_tat(mdl, p):
        # t_at is in DAYS; the network was trained on SECONDS
        return mdl.nn[p].net.inputs[i_tat] == _scale_in(86400.0*mdl.t_at[p], I_TAT)
    instance.NNLINK_TAT = Constraint(instance.p, rule=_lk_tat)

    def _lk_out(mdl, p):
        # standardised output -> dollars, then take the batch-driven share
        return mdl.OPEX[p] == PHI * (mdl.nn[p].net.outputs[i_out]*Y_FACTOR + Y_OFFSET)
    instance.NNLINK_OPEX = Constraint(instance.p, rule=_lk_out)

    return instance