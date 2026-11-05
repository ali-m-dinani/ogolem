# OGOLEM

The Java source of the OGOLEM framework for global optimization.

## To build OGOLEM
Prerequisites:
* Java25+ JDK
* gradle 9 or later

Invoke `gradle build` and a shadowjar will be created in `build/libs/ogolem-snapshot.jar`.

## To run OGOLEM
Prerequisites:
* Java25+ JRE
* the OGOLEM shadowjar - either from building or from downloading an artifact

Please see the manual for input options.

Currently, one MUST explicitly enable incubator modules used by OGOLEM:
```
java --add-modules=jdk.incubator.vector --enable-preview -jar ogolem-snapshot.jar ....
```

## Local Windows setup

This checkout has a project-local Temurin JDK 25.0.5+8 and Gradle 9.2.1
under `.tools/`. OGolem is a Java application and does not need a Python venv.
The launchers select these tools without changing system-wide Java settings.
Gradle dependencies and caches are also stored under `.tools/`.

From PowerShell:

```powershell
cd C:\Users\alimo\PycharmProjects\ogolem
.\ogolem.cmd --help
.\ogolem.cmd --version
```

To rebuild and run the project's checks and unit tests:

```powershell
.\build.cmd
```

The runnable artifact is `build/libs/ogolem-snapshot.jar`.
The launcher enables the required preview features and vector incubator module;
the corresponding incubator warning is expected.
The `--adaptive` mode is the entry point for parameter optimization.
External computational engines and the optional native MPI helper are not
installed by this setup. The local tools and generated build files are ignored
by Git; the launchers require the `.tools/` installations to be present.

## ReaxFF input conversion

The local Python converter and external evaluation bridge are documented in
[tools/REAXFF_BRIDGE.md](tools/REAXFF_BRIDGE.md). Run `reaxff-convert.cmd --help`
from PowerShell for inspection, conversion, and force-field reconstruction.
It supports an S4-inspired genetic-search preset, SI mask/min/max parameter
files, AMS standalone evaluation with Roar Collab Slurm templates, and JAX-MD
evaluation through JAX-ReaxFF. Read the backend limitations before fitting.
