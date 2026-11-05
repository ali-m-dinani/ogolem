# Legacy ReaxFF inputs to OGolem

`reaxff_bridge.py` is a standalone Python 3.10+ converter with no third-party
runtime dependencies. This Windows checkout has a dedicated `.venv` for it.
It uses OGolem's `externalcaller` / `ExternalLocOpt` protocol. Optional adapters
support AMS standalone ReaxFF and JAX-MD through the installed JAX-ReaxFF Python
API. The converter itself does not import those optional libraries or ReaxKit.
Three missing adaptive configuration keywords are now exposed in Java:
`CrossProbability`, `MutationRatio` and `ParentsChoice`. Rebuild this checkout
before using the S4-inspired preset; an unmodified upstream jar rejects them.

## Supported inputs

| Input | Treatment |
| --- | --- |
| `ffield` | Index the classic seven sections. Resolve selected values and preserve unselected text. |
| `params` | Convert one-based section/entry/parameter pointers into an ordered vector and bounds. Preserve intervals and comments as metadata. |
| `ffield_bool`, `ffield_min`, `ffield_max` | Alternative when `params` is absent, as in the Shchygol SI. Validate matching topology and binary selection flags; generate ordered params/bounds. |
| `geo` | Validate BIOGRF/XTLGRF structures and DESCRP names; inventory atoms and CRYSTX cells. Preserve original bytes, including restraints and directives. |
| `trainset.in` | Parse CHARGE, HEATFO, GEOMETRY, CELL PARAMETERS, and ENERGY blocks, including signed/divided energy combinations. |
| `control`, `iopt` | Preserve original source copies. Copy control into evaluations; the engine adapter must choose evaluation-only settings instead of inheriting iopt. |

The classic layout has four lines of eight values per atom and two lines of
eight values per bond. Extended layouts and unsupported training sections are
rejected. Missing structures, invalid pointers, duplicate pointers, absent
bounds, equal bounds and non-finite data are rejected. Inputs use UTF-8/ASCII.
Atom-type references are validated against the ffield atom count, including
zero wildcards at torsion endpoints. Out-of-range references are reported by
`inspect` and rejected by conversion, `check` and evaluation, including for
previously generated bundles. No disabled-term convention is inferred.

No physical training geometry is reduced to an isolated-molecule XYZ. The
generated `dummy.xyz` only initializes OGolem's external interface; the evaluator
receives the original geometry file, including periodicity.

## Inspect and convert

From `C:\Users\alimo\PycharmProjects\ogolem`:

```powershell
.\reaxff-convert.cmd inspect C:\Users\alimo\PycharmProjects\sample_ff_opt
.\reaxff-convert.cmd convert C:\path\to\complete_inputs converted\my_fit
```

The destination must not exist. Source files are never changed. Optional flags:

- `--normalize-bounds`: explicitly sort reversed lower/upper endpoints and report changes.
- `--iterations 100 --population 10`: configure OGolem's global search.
- `--profile s4-inspired`: use the compatible S4 operator mixture; defaults
  to population 500 and 110000 iterations unless overridden. See limitations below.
- `--backend ams` or `--backend jax-md`: configure a supplied evaluator instead
  of the default `custom` adapter contract.
- `--engine-python PATH`: interpreter for the engine adapter. Defaults to the
  bridge interpreter; for JAX use the JAX-ReaxFF environment, not the empty bridge venv.
- `--ams-job-file PATH`: explicitly copy a trusted AMS candidate-run shell script.
- `--seed`: include initial parameters in population initialization; rejects
  out-of-bounds initial values. Without this flag, initialize randomly within
  bounds and retain the original values separately in `initial.params`.

The `params` search interval is metadata, not an OGolem mutation step. The
bridge performs global genetic search only: its external `locopt` callback
evaluates candidates without changing their parameter vectors. It does not
implement local parameter minimization or reproduce the legacy sequential
parameter-search algorithm.

Generated files include `fit.ogo`, `fit-borders.aux`, `initial.params`,
`manifest.json`, unchanged `inputs/`, `evaluator.json`, and callback launchers.
The manifest contains parameter mappings, parsed targets and geometries,
warnings and SHA256 hashes. Bounds retain ReaxFF's original parameter units.

## Connect the evaluator

The converter does not install a ReaxFF engine or bundle proprietary software.
The built-in backends are described below. With `--backend custom`, configure
`evaluator.json` with a trusted **evaluation-only adapter**, for example:

```json
{
  "command": ["C:\\path\\to\\python.exe", "C:\\path\\to\\evaluate_reaxff.py"],
  "timeout_seconds": 3600
}
```

The command runs in a unique candidate directory containing `ffield`, `params`,
`geo`, `trainset.in`, `control` when supplied, and `request.json`. The request
points to the manifest. The adapter must evaluate all targets at the supplied
force field, using appropriate geometry relaxation, charge and cell settings.
Choose `iopt` or equivalent settings for the actual engine/version. The sample's
`iopt=4` is not defined by the attached 2017 manual and is not assumed safe.
The sample's scheduler script is not copied unless explicitly selected with
`--ams-job-file`. Conversion never executes or submits any script.

On success, write one of these response forms to `fitness.json`:

```json
{"fitness": 123.45}
```

```json
{"predictions": [1.2, -3.4]}
```

For predictions, provide one scalar per manifest target, in the same order
and units. An ENERGY prediction is the full signed/divided energy combination,
not individual structure energies. The bridge computes
`sum(((prediction - target) / weight)**2)` (ReaxFF manual Eq. 3.1).
If the engine already reports total training error, return `fitness` directly.
It is not squared again. Exactly one response form is allowed. Do not mix
Hartree and kcal/mol. The placeholder ENERGY reference in `fit.ogo` is not a
training target and is never evaluated by a chemistry backend.

The adapter must exit zero only on success and leave `ffield` unchanged.
The bridge rejects incomplete/non-finite responses, negative fitness,
out-of-bounds candidates, changed source hashes and stale result files.
Candidate values are rounded to the original ffield decimal precision;
values too large for a field are rejected. Each successful evaluation's force
field, response, request and logs are archived in `evaluations/` before OGolem
cleans up its working directory. Failed working directories remain for diagnosis.

## Live checkpoints and interruption recovery

The bridge saves results after EACH successful candidate evaluation, before
returning its fitness to OGolem; it does not wait for the optimization to end.
At the run-directory root it now maintains:

- `best.ffield` and `best.params`: lowest-error evaluated candidate so far.
- `latest.ffield` and `latest.params`: most recently checkpointed candidate.
- `progress.json`: completed evaluation count, best/latest fitness and paths
  to their authoritative evaluation archives, with completion timestamps.
- `evaluations/candidate_*/`: all completed candidates, including full force
  fields, full-precision `parameters.params`, fitness, request, and engine logs.

The best includes initialization, preflight evaluations, and candidates rejected
by GA population/niching rules; it means best evaluated, not necessarily a
current population member. Always use a NEW bundle when changing datasets,
engine, or geometry-relaxation settings; losses from different objectives
must not be mixed in the same archive/checkpoint history.

Candidates are first archived in hidden `.pending_*` directories and only
published under `candidate_*` once their core files are complete. Best/latest
files and `progress.json` use flushed/fsynced temporary files and atomic
replacement. Writers use OS file locking (released on process exit) to avoid
parallel-worker races. The shared filesystem must support these locks and
atomic same-directory renames. This protects against process interruption;
it is not a guarantee against filesystem failures or every power-loss scenario.

`progress.json` is committed last and points to immutable, matching files in
the archives. The four convenience files are individually atomic, not one
multi-file transaction: if interrupted during their update, use the archive
paths in `progress.json` rather than assuming convenience files are in sync.
The current unfinished energy calculation may be lost; earlier completed
evaluations remain. This is NOT exact restart of the GA population/RNG state.
OGolem also enables its own pool serialization on new bests in the generated
S4-inspired preset; that is separate from these portable force-field outputs.

To backfill old archives or recover a candidate archived just before an
interruption in checkpoint publication, stop the run and execute from the
OGolem project root:

```powershell
.\reaxff-convert.cmd refresh-checkpoints --bundle converted\cobalt_demo
```

Recovery validates each candidate against its archived vector, bounds and
fitness, ignores hidden incomplete archives, and rebuilds best/latest without
running an engine. Run it only when the bundle is idle. Corrupt public archives
cause an error instead of being silently skipped. To monitor a running job:

```powershell
Get-Content C:\Users\alimo\PycharmProjects\ogolem\converted\cobalt_demo\progress.json
```

## S4-compatible algorithm settings

The Shchygol et al. 2019 supplementary information, S4 (S9-S13), describes a
private development interface (`reaxffbench`) and private nicher (`WhichNicher=2;75`).
Neither is present in this public checkout. `s4-inspired` is deliberately NOT
an exact reproduction of that branch or its published results.

The preset uses S4's 20% arctic / 80% one-cut portugal crossover mixture and
20% germany / 40% narrow-arctic / 40% broad-arctic mutation mixture, with
cross probability 0.8, mutation ratio 0.2 and rank-selection width 0.3.
`germany:mode=all` maps to random multi-parameter mutation in this checkout,
consistent with the caveat in S4. These are OGolem operator probabilities,
not a promise of mutually exclusive crossover and mutation; public OGolem
always mutates when crossover is skipped and may also mutate after crossover.

It substitutes public static-grid niching (`WhichNicher=0`, 30 bins per
dimension, two individuals per niche), disables extra diversity thresholds,
and preserves new-best serialization and per-evaluation archives. It does
not copy obsolete weight-conversion/fitness settings: the adapter returns
the training error directly, without another square or unit conversion.
The paper's 110000-iteration setting was for cobalt; choose your own budget.
One global iteration can evaluate two children, in addition to initialization.

The SI ZIP supplies mask/min/max files rather than `params`. Copy a chosen
`ffield_init_<dataset>_<guess>` as `ffield`, and its matching
`ts_<dataset>_{ffield_bool,ffield_min,ffield_max,geo,trainset.in}` under the
corresponding unprefixed names before conversion. Do not mix datasets or
atom ordering. Mask/min/max layouts must match each other. If the initial
ffield has extra interaction rows (as in the supplied silica literature file),
selected entries are mapped by their exact ordered atom-index tuples; missing
or ambiguous matches fail rather than using shifted row numbers. Its README explicitly mentions missing geometry
references; this converter remains strict and does not silently drop them.

## AMS standalone on Roar Collab

```powershell
.\reaxff-convert.cmd convert ..\sample_ff_opt converted\sample_ams --backend ams --profile s4-inspired --iterations 100 --population 10 --seed
```

This creates `submit.slurm` with the supplied two-node/16-rank allocation,
64G memory per node, one-hour walltime, `open` account, `basic` partition,
`genoa` constraint and the three requested AMS/Intel/MPI modules. These are
user-provided settings, not a claim that they are available on every cluster.
Edit the resource budget before a large run. No scheduler job is submitted here.

Copy the entire bundle to shared storage visible to every allocated node.
Install Python 3.10+ and the matching Java runtime there. This checkout's jar
requires Java 25 with preview and incubator-vector flags; do not copy a Windows
JDK to Linux. Use the jar rebuilt from this checkout and set:

```bash
cd /shared/path/sample_ams
export OGOLEM_JAR=/shared/path/ogolem-snapshot.jar
export JAVA_BIN=/path/to/linux-jdk-25/bin/java
export REAXFF_BRIDGE_PYTHON=/path/to/python3
sbatch submit.slurm
```

The job runs ONE OGolem worker. Each candidate's `$AMSBIN/reaxff` uses the
entire existing MPI allocation through `NSCM=SLURM_NTASKS`; no nested `sbatch`
or multiple competing MPI launches are used. The candidate directories and
bundle must be on shared storage. `submit.slurm` starts from `SLURM_SUBMIT_DIR`,
so submit from the bundle directory. The launcher checks configuration, not
whether a license or a module is actually available.

The AMS adapter explicitly sets `iopt=6`, `ffoact=3` and writes pointer-only
`params` (no real-valued columns). According to SCM's FFOptimizer documentation,
this evaluates the current ffield once rather than fitting it. It resets
`igeopt=0` so each candidate starts with the supplied geometries, and preserves
other control settings, including geometry/cell relaxation. Original inputs
remain under `inputs/`. It parses the cumulative error from `fort.99`, validates
target count, target/weight coverage and cumulative sums, and rejects truncated,
nonfinite or unrecognized reports. Precision is limited by AMS's printed report.

For a custom candidate script, use `--ams-job-file ..\sample_ff_opt\job.sh`.
It is run with `bash` in each candidate directory INSIDE the outer allocation;
its `#SBATCH` lines are comments, not a second resource request. Keep its
environment setup and final `"$AMSBIN/reaxff"`, but do not let it overwrite
ffield, iopt, params, control or change directory. Do not supply an optimization
script or one that submits asynchronous jobs. The generated default outer
`submit.slurm` still controls resources and can be edited independently.

Before a long fit, use `sbatch submit-evaluate.slurm` for one initial-vector
evaluation (no parameter search), and inspect `fort.99` in the evaluation
archive. Do not submit it concurrently with `submit.slurm` in the same bundle.
The AMS adapter has automated
mock-engine tests, not a licensed AMS/HPC execution test on this computer.

## JAX-MD backend

```powershell
.\reaxff-convert.cmd convert ..\sample_ff_opt converted\sample_jax --backend jax-md --engine-python C:\Users\alimo\PycharmProjects\JAX-ReaxFF\.venv\Scripts\python.exe --profile s4-inspired --iterations 100 --population 10 --seed
```

This backend imports JAX-MD through JAX-ReaxFF's evaluation API and calls the
driver with `--num_trials 0 --num_steps 0`. It does NOT optimize force-field
parameters. It captures full-precision per-target errors before the rounded
text report is written, checks target coverage, and archives the report.
It requires the installed JAX-ReaxFF-compatible `jax_md.reaxff` API; newer
JAX-MD releases have reorganized modules and are not drop-in replacements.
On Linux, change `evaluator.json` to point at the Linux JAX environment.

`engine.json` controls geometry minimization steps, learning rate, RMSG target,
cutoffs, clusters and interaction-counting processes. Unspecified steps/RMSG/
angle-torsion cutoff are derived from `imetho`/`imaxit`, `endmm`, and `cutof2`.
The default process count is one; this does not limit all XLA CPU threads.
JAX-ReaxFF honors its supported geo periodicity, total-charge and restraint
directives, but does not read the complete legacy control file. Its minimizer
and convergence behavior differ from AMS. Do not treat backend switching as
numerically identical physics or assume every legacy directive is implemented.

Only ENERGY, CHARGE and explicit distance/angle/torsion targets are accepted.
HEATFO, cell targets, cell optimization, unsupported charge modes, RMSG targets
and some unsupported geometry directives are rejected. Validate every chosen
ffield/geometry against your reference engine before a production fit.
Each candidate starts a new JAX process and recompiles: correct isolation but
potentially expensive; this is not yet a persistent high-throughput JAX server.

## Run and reconstruct

After configuring and independently verifying the real evaluator:

```powershell
cd C:\Users\alimo\PycharmProjects\ogolem\converted\my_fit
& ..\..\.venv\Scripts\python.exe bridge.py check --bundle .
$env:REAXFF_BRIDGE_PYTHON = (Resolve-Path ..\..\.venv\Scripts\python.exe).Path
$env:OGO_EXTCALLCMD = (Resolve-Path .\ogo_evaluator.cmd).Path
& ..\..\ogolem.cmd --adaptive fit.ogo 1
```

The last argument is concurrent OGolem workers. Account for engine threads
and license limits when increasing it. `check` validates configuration; it
does not test the engine. On Linux, point `OGO_EXTCALLCMD` at executable
`ogo_evaluator.sh`, set `REAXFF_BRIDGE_PYTHON`, and use the Java invocation in
the repository README.

Reconstruct a full force field from a saved OGolem vector:

```powershell
.\reaxff-convert.cmd render --bundle converted\my_fit --parameters path\to\saved.params --output recovered.ffield
```

Run that command from the repository root. `render` allows original values
outside bounds for source round trips; evaluation enforces bounds. Existing
output files are not overwritten.

## Supplied sample

The sample contains 22 selected parameters, 149 geometries and 263 ENERGY targets.
The audit found 117 missing structures in the `m_c11`, `m_c12`, `m_c13`, `m_c22`,
`m_c23`, `m_c33`, `m_c44`, `m_c55` and `m_c66` groups. It also found reversed bounds
on params line 5 (`-0.03, -0.20`) and initial parameter `(3, 52, 9) = -0.1033`
outside its requested `[0, 25]` interval.

At the user's request, the 117 affected targets have now been removed, leaving
146 ENERGY targets and all 149 existing geometries. Original params/trainset
are preserved in `sample_ff_opt/before_ogolem_cleanup/`. The reversed interval
is sorted, and `(3,52,9)` now has bounds `[-0.1033,25]`, minimally expanded to
include the original value without modifying ffield. Review this bound for
physical appropriateness. These changes alter the objective: the omitted
elastic-deformation targets no longer constrain the fit. The original report
`converted/sample_ff_opt_validation.json` records the pre-cleanup problems;
The subsequent full-sample JAX preflight exposed another input problem:
ffield lines 445 and 465 contain torsion tuples `0 3 53 0` and `17 3 3 57`,
although only 19 atom types are defined. These also occur in the original
ReaxKit source dataset, not just the converted copy. Their intended meanings
must be confirmed before correcting or omitting them. The sample bundles are
therefore NOT currently ready to run. The earlier clean report did not check
atom-type references; current `inspect` reports both problematic entries.
An OGolem callback failure can print `Fitness of this parameter set is
1000000.0`; after a nonzero callback exit this is a failure sentinel, not a
calculated training loss. Inspect the retained `externalcaller-*` directory's
`evaluator.stderr.log` for the engine error.

## Tests and references

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools/tests -v
```

The integration smoke test invokes the actual built Windows OGolem JAR with
a synthetic evaluator. It verifies the protocol, not ReaxFF physics.
The engine tests additionally run a tiny S4-inspired GA with a synthetic
objective. For the optional real two-cobalt-atom JAX single-point smoke test:

```powershell
$env:JAX_REAXFF_TEST_PYTHON = 'C:\Users\alimo\PycharmProjects\JAX-ReaxFF\.venv\Scripts\python.exe'
& $env:JAX_REAXFF_TEST_PYTHON -m unittest discover -s tools/tests -v
```

The real JAX smoke test passed locally. It is not a cross-engine accuracy
benchmark or a validation of the Zn/Pt/O training set's physical model.
The supplied SI mask files were also checked: cobalt selects 12 parameters,
disulfide 87 and silica 67, including topology remapping for silica's extra rows.

Sources consulted: the supplied ReaxFF June 2017 manual, training files
pp. 37–40 and error definition pp. 47–48; ReaxKit's ffield/params/training
readers; OGolem's manual and `examples/external_adaptive`,
`AdaptiveParameters.java`, `AdaptiveExternalCaller.java`, `ExternalLocOpt.java`.
No ReaxKit source is vendored.

Additional sources: Shchygol et al., *ReaxFF Parameter Optimization with Monte
Carlo and Evolutionary Algorithms: Guidelines and Insights*, 2019, SI S4 and
the `ct9b00769_si_001/README.txt` supplied locally; SCM's
[FFOptimizer inputs](https://www.scm.com/doc/OldReaxFF/ffopt_Input_files.html),
[evaluation task](https://www.scm.com/doc/OldReaxFF/ffopt_Control_parameters.html),
and [standalone manual, section 5](https://www.scm.com/doc/OldReaxFF/_downloads/d7a52e6497a56f2d45060be77b3ff6ff/OldStandaloneReaxFF.pdf).
