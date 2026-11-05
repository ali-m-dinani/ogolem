@echo off
setlocal
call "%~dp0environment.cmd"
if errorlevel 1 exit /b 1
pushd "%~dp0"
call gradle --no-daemon build %*
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %BUILD_EXIT_CODE%
