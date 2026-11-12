@echo off
rem Build the GPU programs on Windows with the CUDA toolkit + Visual Studio C++ tools.
rem usage: build.bat            (builds verify_bench.exe, rocket.exe and visualizer.exe)
rem        build.bat rocket     (builds one)
rem Set ARCH for other GPUs (default sm_86 = RTX 30 series), e.g. set ARCH=sm_89

setlocal
cd /d "%~dp0"
if "%ARCH%"=="" set ARCH=sm_86

where cl >nul 2>nul
if errorlevel 1 (
  for %%V in ("%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools" "%ProgramFiles%\Microsoft Visual Studio\2022\Community" "%ProgramFiles%\Microsoft Visual Studio\18\Community") do (
    if exist "%%~V\VC\Auxiliary\Build\vcvars64.bat" if not defined VCDONE (
      call "%%~V\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
      set VCDONE=1
    )
  )
)

if not exist outputs mkdir outputs
set FLAGS=-O3 -std=c++17 -arch=%ARCH% -Xcompiler "/EHsc /wd4244 /wd4267 /wd4305" -D_USE_MATH_DEFINES

if "%1"=="" (
  call :build verify_bench || exit /b 1
  call :build rocket || exit /b 1
  call :build visualizer || exit /b 1
) else (
  call :build %1 || exit /b 1
)
exit /b 0

:build
echo building %1.exe
set LIBS=
if "%1"=="visualizer" set LIBS=-lopengl32 -lglu32 -luser32 -lgdi32
nvcc %FLAGS% %1.cu -o %1.exe %LIBS%
exit /b %errorlevel%
