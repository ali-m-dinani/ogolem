@echo off
setlocal
call "%~dp0environment.cmd"
if errorlevel 1 exit /b 1
if not exist "%~dp0build\libs\ogolem-snapshot.jar" (
  echo Build OGolem first with build.cmd.
  exit /b 1
)
"%JAVA_HOME%\bin\java.exe" --add-modules=jdk.incubator.vector --enable-preview -jar "%~dp0build\libs\ogolem-snapshot.jar" %*
exit /b %ERRORLEVEL%
