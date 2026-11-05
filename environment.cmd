@echo off
for /d %%D in ("%~dp0.tools\jdk-25*") do set "JAVA_HOME=%%~fD"
if not exist "%JAVA_HOME%\bin\javac.exe" exit /b 1
set "GRADLE_USER_HOME=%~dp0.tools\gradle-cache"
set "PATH=%JAVA_HOME%\bin;%~dp0.tools\gradle-9.2.1\bin;%PATH%"
