
from pynare.parsing.base import Token

# the different variable types that can be declared in dynare - Section 4.2 in doc.
VAR             = 'var'
VAREXO             = 'varexo'
VAREXO_DET         = 'varexo_det'
PARAMETERS         = 'parameters'
PREDETERMINED     = 'predetermined_variables'
TREND_VAR         = 'trend_var'
LOG_TREND_VAR     = 'log_trend_var'
CHANGE_TYPE     = 'change_type'
END                = 'end'


# model block beginning - Section 4.5 of doc.
MODEL             = 'model'

# simulation commands - Section 4.7 of doc.
INITVAL     = 'initval'
ENDVAL         = 'endval'
HISTVAL     = 'histval'
RESID         = 'resid'
INITVALFILE = 'initval_file'
HISTVALFILE = 'histval_file'

# shock block commands - Section 4.8 of doc.
SHOCKS     = 'shocks'
PERIODS = 'periods'
VALUES     = 'values'
CORR     = 'corr'
STDERR     = 'stderr'
# MSHOCKS = 'mshocks'

# other general declarations - Section 4.9 of doc.
DSAMPLE = 'dsample'


# different steady state commands - Section 4.10 of doc.
STEADY                 = 'steady'
HOMOTOPY_SETUP         = 'homotopy_setup'
STEADY_STATE_MODEL     = 'steady_state_model'

# getting informationg about the model - Section 4.11 of doc.
CHECK                             = 'check'
MODEL_DIAGNOSTICS                 = 'model_diagnostics'
MODEL_INFO                         = 'model_info'
PRINT_BYTECODE_DYNAMIC_MODEL     = 'print_bytecode_dynamic_model'
PRINT_BYTECODE_STATIC_MODEL     = 'print_bytecode_static_model'

# deterministic simulation - Section 4.12 of doc.
PERFECT_FORESIGHT_SETUP         = 'perfect_foresight_setup'
PERFECT_FORESIGHT_SOLVER         = 'perfect_foresight_solver'
SIMUL                             = 'simul'

# stochastic solution and simulation - Section 4.13 of doc.
STOCH_SIMUL = 'stoch_simul'
EXTENDED_PATH = 'extended_path'

# estimation - Section 4.14 of doc.
VAROBS                     = 'varobs'
OBSERVATION_TRENDS         = 'observation_trends'
ESTIMATED_PARAMS          = 'estimated_params'
ESTIMATED_PARAMS_INIT     = 'estimated_params_init'
ESTIMATED_PARAMS_BOUNDS = 'estimated_params_bounds'
ESTIMATION                 = 'estimation'
UNIT_ROOT_VARS             = 'unit_root_vars'
BVAR_DENSITY             = 'bvar_density'

# model comparison - Section 4.15 of doc.
MODEL_COMPARISON         = 'model_comparison'

# shock decomposition - Section 4.16 of doc.
SHOCK_DECOMPOSITION             = 'shock_decomposition'
SHOCK_GROUPS                     = 'shock_groups'
REALTIME_SHOCK_DECOMPOSITION     = 'realtime_shock_decomposition'
PLOT_SHOCK_DECOMPOSITION         = 'plot_shock_decomposition'

# calibrated smoother - Section 4.17 of doc.
CALIB_SMOOTHER             = 'calib_smoother'

# forecasting - Section 4.18 of doc.
FORECAST                     = 'forecast'
CONDITIONAL_FORECAST         = 'conditional_forecast'
CONDITIONAL_FORECAST_PATHS     = 'conditional_forecast_paths'
PLOT_CONDITIONAL_FORECAST     = 'plot_conditional_forecast'
BVAR_FORECAST                 = 'bvar_forecast'
INIT_PLAN                     = 'init_plan'
BASIC_PLAN                     = 'basic_plan'
FLIP_PLAN                     = 'flip_plan'
DET_COND_FORECAST             = 'det_cond_forecast'
SMOOTHER2HISTVAL             = 'smoother2histval'

# optimal policy - Section 4.19 of doc.
OSR                     = 'osr'
OSR_PARAMS                 = 'osr_params'
OPTIM_WEIGHTS             = 'optim_weights'
OSR_PARAMS_BOUNDS         = 'osr_params_bounds'
RAMSEY_MODEL             = 'ramsey_model'
RAMSEY_CONSTRAINTS         = 'ramsey_constraints'
RAMSEY_POLICY             = 'ramsey_policy'
DISCRETIONARY_POLICY     = 'discretionary_policy'
PLANNER_OBJECTIVE         = 'planner_objective'

# sensitivity and identification analysis - Section 4.20 of doc.
DYNARE_SENSITIVITY     = 'dynare_sensitivity'
IRF_CALIBRATION     = 'irf_calibration'
MOMENT_CALIBRATION     = 'momen_calibration'
IDENTIFICATION         = 'identification'

# Markov-switching SBVAR
MARKOV_SWITCHING             = 'markov_switching'
SVAR                         = 'svar'
SBVAR                         = 'sbvar'
SVAR_IDENTIFICATION         = 'svar_identification'
MS_ESTIMATION                 = 'ms_estimation'
MS_SIMULATION                 = 'ms_simulation'
MS_COMPUTE_MDD                 = 'ms_compute_mdd'
MS_COMPUTE_PROBABILITIES     = 'ms_compute_probabilities'
MS_IRF                         = 'ms_irf'
MS_FORECAST                    = 'ms_forecast'
MS_VARIANCE_DECOMPOSITION     = 'ms_variance_decomposition'

#


dynare_reserved_kw = {
    # the different variable types that can be declared in dynare - Section 4.2 in doc.
    VAR:             Token(VAR, VAR),
    VAREXO:         Token(VAREXO, VAREXO),
    VAREXO_DET:     Token(VAREXO_DET, VAREXO_DET),
    PARAMETERS:     Token(PARAMETERS, PARAMETERS),
    PREDETERMINED:     Token(PREDETERMINED, PREDETERMINED),
    TREND_VAR:         Token(TREND_VAR, TREND_VAR),
    LOG_TREND_VAR:     Token(LOG_TREND_VAR, LOG_TREND_VAR),
    CHANGE_TYPE:     Token(CHANGE_TYPE, CHANGE_TYPE),
    END:             Token(END, END),

    # model block beginning - Section 4.5 of doc.
    MODEL:             Token(MODEL, MODEL),

    # simulation commands - Section 4.7 of doc.
    INITVAL:         Token(INITVAL, INITVAL),
    ENDVAL:         Token(ENDVAL, ENDVAL),
    HISTVAL:        Token(HISTVAL, HISTVAL),
    RESID:             Token(RESID, RESID),
    INITVALFILE:     Token(INITVALFILE, INITVALFILE),
    HISTVALFILE:     Token(HISTVALFILE, HISTVALFILE),

    # shock block commands - Section 4.8 of doc.
    SHOCKS:         Token(SHOCKS, SHOCKS),
    PERIODS:        Token(PERIODS, PERIODS),
    VALUES:            Token(VALUES, VALUES),
    CORR:            Token(CORR, CORR),
    STDERR:            Token(STDERR, STDERR),

    # other general declarations - Section 4.9 of doc.
    DSAMPLE:         Token(DSAMPLE, DSAMPLE),

    # different steady state commands - Section 4.10 of doc.
    STEADY:             Token(STEADY, STEADY),
    HOMOTOPY_SETUP:     Token(HOMOTOPY_SETUP, HOMOTOPY_SETUP),
    STEADY_STATE_MODEL: Token(STEADY_STATE_MODEL, STEADY_STATE_MODEL),

    # getting information about the model - Section 4.11 of doc.
    CHECK:                             Token(CHECK, CHECK),
    MODEL_DIAGNOSTICS:                 Token(MODEL_DIAGNOSTICS, MODEL_DIAGNOSTICS),
    MODEL_INFO:                     Token(MODEL_INFO, MODEL_INFO),
    PRINT_BYTECODE_DYNAMIC_MODEL:     Token(
        PRINT_BYTECODE_DYNAMIC_MODEL, PRINT_BYTECODE_DYNAMIC_MODEL
    ), 
    PRINT_BYTECODE_STATIC_MODEL:     Token(
        PRINT_BYTECODE_STATIC_MODEL, PRINT_BYTECODE_STATIC_MODEL
    ),

    # deterministic simulation - Section 4.12 of doc.
    PERFECT_FORESIGHT_SETUP:     Token(PERFECT_FORESIGHT_SETUP, PERFECT_FORESIGHT_SETUP),
    PERFECT_FORESIGHT_SOLVER:     Token(PERFECT_FORESIGHT_SOLVER, PERFECT_FORESIGHT_SOLVER),
    SIMUL:                         Token(SIMUL, SIMUL),

    # stochastic solution and simulation - Section 4.13 of doc.
    STOCH_SIMUL:     Token(STOCH_SIMUL, STOCH_SIMUL),
    EXTENDED_PATH:     Token(EXTENDED_PATH, EXTENDED_PATH),

    # estimation - Section 4.14 of doc
    VAROBS:                     Token(VAROBS, VAROBS),
    ESTIMATED_PARAMS:             Token(ESTIMATED_PARAMS, ESTIMATED_PARAMS),
    ESTIMATED_PARAMS_INIT:         Token(ESTIMATED_PARAMS_INIT, ESTIMATED_PARAMS_INIT),
    ESTIMATED_PARAMS_BOUNDS:     Token(ESTIMATED_PARAMS_BOUNDS, ESTIMATED_PARAMS_BOUNDS),
    ESTIMATION:                 Token(ESTIMATION, ESTIMATION),
    UNIT_ROOT_VARS:             Token(UNIT_ROOT_VARS, UNIT_ROOT_VARS),
    BVAR_DENSITY:                 Token(BVAR_DENSITY, BVAR_DENSITY),

    # model comparison - Section 4.15 of doc.
    MODEL_COMPARISON: Token(MODEL_COMPARISON, MODEL_COMPARISON),

    # shock decomposition - Section 4.16 of doc.
    SHOCK_DECOMPOSITION:             Token(SHOCK_DECOMPOSITION, SHOCK_DECOMPOSITION),
    SHOCK_GROUPS:                     Token(SHOCK_GROUPS, SHOCK_GROUPS),
    REALTIME_SHOCK_DECOMPOSITION:     Token(
        REALTIME_SHOCK_DECOMPOSITION, REALTIME_SHOCK_DECOMPOSITION
    ),
    PLOT_SHOCK_DECOMPOSITION:         Token(
        PLOT_SHOCK_DECOMPOSITION, PLOT_SHOCK_DECOMPOSITION
    ),

    # calibrated smoothing - Section 4.17 of doc.
    CALIB_SMOOTHER: Token(CALIB_SMOOTHER, CALIB_SMOOTHER),

    # forecasting - Section 4.18 of doc.
    FORECAST:                     Token(FORECAST, FORECAST),
    CONDITIONAL_FORECAST:         Token(CONDITIONAL_FORECAST, CONDITIONAL_FORECAST),
    CONDITIONAL_FORECAST_PATHS: Token(
        CONDITIONAL_FORECAST_PATHS, CONDITIONAL_FORECAST_PATHS
    ),
    PLOT_CONDITIONAL_FORECAST:     Token(
        PLOT_CONDITIONAL_FORECAST, PLOT_CONDITIONAL_FORECAST
    ),
    BVAR_FORECAST:                 Token(BVAR_FORECAST, BVAR_FORECAST),
    INIT_PLAN:                     Token(INIT_PLAN, INIT_PLAN),
    BASIC_PLAN:                 Token(BASIC_PLAN, BASIC_PLAN),
    FLIP_PLAN:                     Token(FLIP_PLAN, FLIP_PLAN),
    DET_COND_FORECAST:             Token(DET_COND_FORECAST, DET_COND_FORECAST),
    SMOOTHER2HISTVAL:             Token(SMOOTHER2HISTVAL, SMOOTHER2HISTVAL),

    # optimal policy - Section 4.19 of doc.
    OSR:                     Token(OSR, OSR),
    OSR_PARAMS:             Token(OSR_PARAMS, OSR_PARAMS),
    OSR_PARAMS_BOUNDS:         Token(OSR_PARAMS_BOUNDS, OSR_PARAMS_BOUNDS),
    RAMSEY_MODEL:             Token(RAMSEY_MODEL, RAMSEY_MODEL),
    RAMSEY_CONSTRAINTS:     Token(RAMSEY_CONSTRAINTS, RAMSEY_CONSTRAINTS),
    RAMSEY_POLICY:             Token(RAMSEY_POLICY, RAMSEY_POLICY),
    DISCRETIONARY_POLICY:     Token(DISCRETIONARY_POLICY, DISCRETIONARY_POLICY),
    PLANNER_OBJECTIVE:         Token(PLANNER_OBJECTIVE, PLANNER_OBJECTIVE),

    # sensitivity and identification analysis - Section 4.20 of doc.
    DYNARE_SENSITIVITY: Token(DYNARE_SENSITIVITY, DYNARE_SENSITIVITY),
    IRF_CALIBRATION:     Token(IRF_CALIBRATION, IRF_CALIBRATION),
    MOMENT_CALIBRATION: Token(MOMENT_CALIBRATION, MOMENT_CALIBRATION),
    IDENTIFICATION:     Token(IDENTIFICATION, IDENTIFICATION),

    # markov-switching SBVAR = Section 4.21 of doc.
    MARKOV_SWITCHING:             Token(MARKOV_SWITCHING, MARKOV_SWITCHING),
    SVAR:                         Token(SVAR, SVAR),
    SBVAR:                         Token(SBVAR, SBVAR),
    SVAR_IDENTIFICATION:         Token(SVAR_IDENTIFICATION, SVAR_IDENTIFICATION),
    MS_ESTIMATION:                 Token(MS_ESTIMATION, MS_ESTIMATION),
    MS_SIMULATION:                 Token(MS_SIMULATION, MS_SIMULATION),
    MS_COMPUTE_MDD:             Token(MS_COMPUTE_MDD, MS_COMPUTE_MDD),
    MS_COMPUTE_PROBABILITIES:     Token(
        MS_COMPUTE_PROBABILITIES, MS_COMPUTE_PROBABILITIES
    ),
    MS_IRF:                     Token(MS_IRF, MS_IRF),
    MS_FORECAST:                 Token(MS_FORECAST, MS_FORECAST),
    MS_VARIANCE_DECOMPOSITION:     Token(
        MS_VARIANCE_DECOMPOSITION, MS_VARIANCE_DECOMPOSITION
    )
}
