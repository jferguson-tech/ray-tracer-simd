@echo off
rem Build pathtracer.exe on Windows with Visual Studio (MSVC, x64, AVX2).
rem
rem   build_windows.bat          build
rem   build_windows.bat run      build, then start the interactive viewer
rem
rem Needs Visual Studio 2019 or 2022 with "Desktop development with C++".
rem SDL2 is downloaded once into third_party\ (official release, checksum verified).

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "SDL_VER=2.32.10"
set "SDL_SHA256=af347939395a58b365846aaea27391e69f9ec9d4dd650d6ac40802159b418a6e"
set "SDL_URL=https://github.com/libsdl-org/SDL/releases/download/release-%SDL_VER%/SDL2-devel-%SDL_VER%-VC.zip"
set "SDL_DIR=third_party\SDL2-%SDL_VER%"
set "SDL_ZIP=third_party\SDL2-devel-%SDL_VER%-VC.zip"

rem ---- compiler: use the current prompt if cl is already set up, else find Visual Studio
rem (no parenthesized blocks here: the "(x86)" in the path would close them early)
where cl >nul 2>nul
if not errorlevel 1 goto have_compiler
set "VSINSTALLER=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
if not exist "%VSINSTALLER%\vswhere.exe" goto no_visual_studio
set "PATH=%PATH%;%VSINSTALLER%"
set "VSDIR="
for /f "usebackq delims=" %%i in (`vswhere.exe -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR goto no_cpp_tools
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
where cl >nul 2>nul
if errorlevel 1 goto no_cpp_tools
goto have_compiler

:no_visual_studio
echo ERROR: Visual Studio was not found. Install Visual Studio 2022 with "Desktop development with C++".
exit /b 1

:no_cpp_tools
echo ERROR: Visual Studio's C++ tools were not found. Add "Desktop development with C++" in the Visual Studio Installer.
exit /b 1

:have_compiler

rem ---- SDL2: download and unpack once
if not exist "%SDL_DIR%\lib\x64\SDL2.lib" (
    if not exist third_party mkdir third_party
    echo Downloading SDL2 %SDL_VER% ...
    curl.exe -L --fail --silent --show-error -o "%SDL_ZIP%" "%SDL_URL%"
    if errorlevel 1 (
        echo ERROR: could not download %SDL_URL%
        exit /b 1
    )
    set "GOT="
    for /f "skip=1 tokens=* delims=" %%h in ('certutil -hashfile "%SDL_ZIP%" SHA256') do if not defined GOT set "GOT=%%h"
    set "GOT=!GOT: =!"
    if /i not "!GOT!"=="%SDL_SHA256%" (
        echo ERROR: the SDL2 download does not match its expected checksum; deleting it.
        echo   expected %SDL_SHA256%
        echo   got      !GOT!
        del "%SDL_ZIP%"
        exit /b 1
    )
    tar.exe -xf "%SDL_ZIP%" -C third_party
    if errorlevel 1 (
        echo ERROR: could not unpack %SDL_ZIP%
        exit /b 1
    )
    del "%SDL_ZIP%"
)
rem trace.cpp includes <SDL2/SDL.h>; the Visual C++ package keeps its headers flat.
if not exist "%SDL_DIR%\include\SDL2\SDL.h" (
    mkdir "%SDL_DIR%\include\SDL2" 2>nul
    copy /y "%SDL_DIR%\include\*.h" "%SDL_DIR%\include\SDL2\" >nul
)

rem ---- compile
echo Compiling trace.cpp ...
if not exist build mkdir build
cl /nologo /std:c++17 /O2 /arch:AVX2 /EHsc /MD /I"%SDL_DIR%\include" trace.cpp ^
   /Fo"build\\" /Fe"pathtracer.exe" ^
   /link /SUBSYSTEM:CONSOLE /LIBPATH:"%SDL_DIR%\lib\x64" SDL2main.lib SDL2.lib shell32.lib
if errorlevel 1 (
    echo.
    echo BUILD FAILED
    exit /b 1
)
copy /y "%SDL_DIR%\lib\x64\SDL2.dll" . >nul

echo.
echo Built pathtracer.exe
echo   pathtracer.exe                       interactive viewer
echo   pathtracer.exe --play                play demo.json in the window
echo   pathtracer.exe --offline --samples 64 --resolution 3    render demo.json to output\
echo   pathtracer.exe --help                all options

if /i "%~1"=="run" pathtracer.exe
endlocal
