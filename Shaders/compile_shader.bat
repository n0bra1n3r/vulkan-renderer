@echo off
setlocal enabledelayedexpansion

set "FILE=%~1"
set "DIR=%~dp1"
set "NAME=%~n1"

cd /d "%~dp0"

set "DXC=%VULKAN_SDK%\Bin\dxc.exe"

rem Split base.stage.hlsl correctly
for /f "tokens=1-3 delims=." %%a in ("%~nx1") do (
    set "base=%%a"
    set "stage=%%b"
    set "ext2=%%c"
)

set "compile=0"

if /i "!ext2!"=="hlsl" (
    if /i "!stage!"=="frag" (
        set "profile=ps_6_6"
        set "compile=1"
    ) else if /i "!stage!"=="vert" (
        set "profile=vs_6_6"
        set "compile=1"
    )
)

if "!compile!"=="1" (
    echo Compiling %FILE% with !profile!...
    "%DXC%" -spirv -T !profile! -E main -Fo "%DIR%!NAME!.spv" "%FILE%"
) else (
    echo Skipping %FILE%.
)

endlocal
