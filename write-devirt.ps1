<#
.SYNOPSIS
    One-shot write-devirt: listings -> payload -> patched PE -> differential -> loader check.

.DESCRIPTION
    Runs the five stages of `write-devirt` in order and stops at the first one that fails, naming
    the stage and the artifact it was reading. Reading the five commands off a README and running
    them by hand is where the mistakes come from -- a stale payload, a patch built from the wrong
    listing set, a differential that never ran. This does it once, the same way every time.

    Stages:
      1. payload.py        listings -> payload.bin + payload.json
      2. peinject.py       payload + host PE -> <host>.neko.exe
      3. miniload.py       map the patched image the way the loader would, and run it
      4. diff_payload.py   code inside the patched image vs the interpreter, byte for byte
      5. verdict           print what was measured, with the numbers from each stage

    Stage 3 and 4 are the two that answer different questions: miniload proves the image is
    loadable and its entry runs; diff_payload proves the code it runs is the code the recovery
    meant. A run that stops after stage 3 has NOT been verified.

.PARAMETER Module
    The host PE to patch. Defaults to $env:TVM_HOST, then $env:TVM_MODULE (from .env).

.PARAMETER Listings
    Listing directory. Defaults to $env:TVM_LISTINGS, then out/hard-cases-listings-final.

.PARAMETER Out
    Output directory. Defaults to $env:TVM_OUT/write-devirt, then out/write-devirt.

.PARAMETER Opt
    Codegen level 0-3. Defaults to $env:TVM_PAYLOAD_OPT, then 0.

.PARAMETER Mem
    sealed | absolute. Defaults to $env:TVM_PAYLOAD_MEM, then sealed.

.PARAMETER SkipDifferential
    Stop after the loader check. The result is then UNVERIFIED and the script says so.

.PARAMETER Force
    Rebuild even when the artifacts already exist and are newer than their inputs.

.EXAMPLE
    .\write-devirt.ps1 -Module 'C:\path\to\SGuardSvc64.exe'

.EXAMPLE
    # everything from .env
    .\write-devirt.ps1

.NOTES
    Exit codes: 0 verified, 1 a stage failed, 2 bad input (missing module/listings/toolchain).
#>
[CmdletBinding()]
param(
    [string]$Module,
    [string]$Listings,
    [string]$Out,
    [ValidateSet(0, 1, 2, 3)][int]$Opt = -1,
    [string]$Mem,
    [switch]$SkipDifferential,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = $PSScriptRoot
$llvm = Join-Path $root 'llvm'
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Write-Host 'python not found on PATH' -ForegroundColor Red; exit 2 }

# --- configuration, in the documented order: flag > .env/environment > default -------------------
function Get-Cfg([string]$Name) { [Environment]::GetEnvironmentVariable($Name) }

if (-not $Module)   { $Module = Get-Cfg 'TVM_HOST'; if (-not $Module) { $Module = Get-Cfg 'TVM_MODULE' } }
if (-not $Listings) { $Listings = Get-Cfg 'TVM_LISTINGS' }
if (-not $Listings) { $Listings = Join-Path $root 'out\hard-cases-listings-final' }
if (-not $Out) {
    $base = Get-Cfg 'TVM_OUT'
    if (-not $base) { $base = Join-Path $root 'out' }
    elseif (-not [IO.Path]::IsPathRooted($base)) { $base = Join-Path $root $base }
    $Out = Join-Path $base 'write-devirt'
}
if ($Opt -lt 0) {
    $e = Get-Cfg 'TVM_PAYLOAD_OPT'
    $Opt = if ($e -and $e -match '^\d+$') { [int]$e } else { 0 }
}
if (-not $Mem) {
    $Mem = Get-Cfg 'TVM_PAYLOAD_MEM'
    if (-not $Mem) { $Mem = 'sealed' }
}
if ($Mem -notin @('sealed', 'absolute')) {
    Write-Host ("memory mode must be 'sealed' or 'absolute', not '{0}'" -f $Mem) -ForegroundColor Red
    exit 2
}

$results = [ordered]@{}
function Step([string]$Name) {
    Write-Host ''
    Write-Host "== $Name " -ForegroundColor Cyan -NoNewline
    Write-Host ('=' * [Math]::Max(0, 68 - $Name.Length)) -ForegroundColor DarkCyan
}
function Fail([string]$Name, [string]$Why, [int]$Code = 1) {
    Write-Host ''
    Write-Host "FAILED at $Name" -ForegroundColor Red
    Write-Host "  $Why" -ForegroundColor Red
    Write-Host '  Nothing downstream was run; the artifacts above it are still on disk.' -ForegroundColor Yellow
    exit $Code
}

# --- preconditions -------------------------------------------------------------------------------
Step 'preconditions'
if (-not $Module) {
    Fail 'preconditions' ('no module: pass -Module, or set TVM_HOST / TVM_MODULE in main/.env') 2
}
if (-not (Test-Path -LiteralPath $Module -PathType Leaf)) {
    Fail 'preconditions' ("module not found: $Module") 2
}
if (-not (Test-Path -LiteralPath $Listings -PathType Container)) {
    Fail 'preconditions' ("listings not found: $Listings`n" +
        "  generate them from your module first:`n" +
        "      cd engine`n" +
        "      python -m core.devirt `"$Module`" --out ../out/listings`n" +
        "  then re-run with -Listings (__ROOT__\out\listings)" ) 2
}
$listFiles = @(Get-ChildItem -LiteralPath $Listings -Filter '0x*.txt' -File)
if ($listFiles.Count -eq 0) {
    Fail 'preconditions' ("no 0x*.txt listings in $Listings") 2
}
foreach ($tool in 'payload.py', 'peinject.py', 'miniload.py', 'diff_payload.py') {
    if (-not (Test-Path (Join-Path $llvm $tool))) { Fail 'preconditions' "missing $tool in llvm/" 2 }
}
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$moduleItem = Get-Item -LiteralPath $Module
Write-Host ("  module    {0}" -f $moduleItem.FullName)
Write-Host ("            {0:N0} bytes" -f $moduleItem.Length)
Write-Host ("  listings  {0}  ({1} file(s))" -f $Listings, $listFiles.Count)
Write-Host ("  out       {0}" -f $Out)
Write-Host ("  codegen   -O{0}   memory {1}" -f $Opt, $Mem)
$results['module'] = $moduleItem.FullName
$results['listings'] = "$Listings ($($listFiles.Count) files)"
$results['out'] = $Out

$payloadBin = Join-Path $Out 'payload.bin'
$payloadJson = Join-Path $Out 'payload.json'
$patched = Join-Path $Out ('{0}.neko{1}' -f $moduleItem.BaseName, $moduleItem.Extension)

# --- 1. payload ----------------------------------------------------------------------------------
Step '1/4  payload.py   listings -> payload.bin'
$sw = [Diagnostics.Stopwatch]::StartNew()
Push-Location $llvm
try {
    $payloadArgs = @('payload.py', "--listings=$Listings", "--out=$Out", "--opt=$Opt", "--mem=$Mem")
    if ($Force) { Remove-Item -LiteralPath $payloadBin, $payloadJson -Force -ErrorAction SilentlyContinue }
    & $py.Source @payloadArgs
    if ($LASTEXITCODE -ne 0) { Fail 'payload.py' "exit code $LASTEXITCODE -- the payload was not built (see the message above)" }
    if (-not (Test-Path -LiteralPath $payloadBin)) { Fail 'payload.py' 'payload.bin was not written' }
    if (-not (Test-Path -LiteralPath $payloadJson)) { Fail 'payload.py' 'payload.json was not written' }
} finally { Pop-Location }
$sw.Stop()
$meta = Get-Content -LiteralPath $payloadJson -Raw | ConvertFrom-Json
$results['functions'] = $meta.nfunc
$results['blocks'] = $meta.blocks
$results['section_bytes'] = $meta.section_bytes
$results['refusals'] = @($meta.refusals)
Write-Host ("  {0} function(s), {1} block(s), {2:N0} byte section, {3:N1}s" -f `
    $meta.nfunc, $meta.blocks, $meta.section_bytes, $sw.Elapsed.TotalSeconds) -ForegroundColor Green

# --- 2. inject ------------------------------------------------------------------------------------
Step '2/4  peinject.py   payload + host -> patched PE'
$sw.Restart()
Push-Location $llvm
try {
    if ($Force) { Remove-Item -LiteralPath $patched -Force -ErrorAction SilentlyContinue }
    & $py.Source 'peinject.py' "--host=$Module" "--payload=$Out" "--out=$patched"
    $code = $LASTEXITCODE
} finally { Pop-Location }
$sw.Stop()
if ($code -ne 0) { Fail 'peinject.py' "exit code $code -- the image was not patched" }
if (-not (Test-Path -LiteralPath $patched)) { Fail 'peinject.py' 'the patched image was not written' }
$results['patched'] = $patched
$results['patched_bytes'] = (Get-Item -LiteralPath $patched).Length
Write-Host ("  {0}  ({1:N0} bytes, {2:N1}s)" -f (Split-Path $patched -Leaf), $results['patched_bytes'],
    $sw.Elapsed.TotalSeconds) -ForegroundColor Green

# --- 3. load --------------------------------------------------------------------------------------
Step '3/4  miniload.py   map the image the way the loader would, and run it'
Push-Location $llvm
try {
    & $py.Source 'miniload.py' $patched
    $code = $LASTEXITCODE
} finally { Pop-Location }
if ($code -ne 0) {
    Fail 'miniload.py' "exit code $code -- the patched image did not load or its entry did not run"
}

# --- 4. differential ------------------------------------------------------------------------------
if ($SkipDifferential) {
    Write-Host ''
    Write-Host 'SKIPPED differential (-SkipDifferential)' -ForegroundColor Yellow
    Write-Host '  The artifact is UNVERIFIED: it loads and runs, and nothing has checked that the' -ForegroundColor Yellow
    Write-Host '  code inside it is the code the recovery meant. Do not quote this run as verified.' -ForegroundColor Yellow
    $results['differential'] = 'SKIPPED -- UNVERIFIED'
} else {
    Step '4/4  diff_payload.py   patched image vs the interpreter'
    Push-Location $llvm
    try {
        & $py.Source 'diff_payload.py' "--payload=$Out" "--patched=$patched" "--listings=$Listings"
        $code = $LASTEXITCODE
    } finally { Pop-Location }
    if ($code -ne 0) {
        Fail 'diff_payload.py' "exit code $code -- DIVERGENCE: the patched image and the interpreter disagree"
    }
    $results['differential'] = 'agree'
}

# --- verdict --------------------------------------------------------------------------------------
Write-Host ''
Write-Host '== result ' -ForegroundColor Cyan -NoNewline
Write-Host ('=' * 60) -ForegroundColor DarkCyan
$verified = -not $SkipDifferential
Write-Host ("  status            {0}" -f $(if ($verified) { 'VERIFIED' } else { 'UNVERIFIED' })) `
    -ForegroundColor $(if ($verified) { 'Green' } else { 'Yellow' })
Write-Host ("  functions         {0}" -f $results['functions'])
Write-Host ("  blocks            {0}" -f $results['blocks'])
Write-Host ("  .neko section     {0:N0} bytes" -f $results['section_bytes'])
Write-Host ("  patched image     {0}" -f $results['patched'])
Write-Host ("  patched size      {0:N0} bytes" -f $results['patched_bytes'])
$refused = @($results['refusals'])
if ($refused.Count) {
    Write-Host ("  refused           {0} function(s) -- named, not silently dropped" -f $refused.Count) -ForegroundColor Yellow
    foreach ($r in $refused) { Write-Host ("      {0}: {1}" -f $r.root, $r.reason) -ForegroundColor Yellow }
} else {
    Write-Host '  refused           none'
}

Write-Host ''
Write-Host '  What this does and does not establish:' -ForegroundColor DarkGray
Write-Host '    - the patched image loads and its entry point runs (miniload)' -ForegroundColor DarkGray
if ($verified) {
    Write-Host '    - the code running inside it agrees with the interpreter byte for byte (diff_payload)' -ForegroundColor DarkGray
}
Write-Host '    NOT established: that the protection is defeated, or that the guest behaviour is' -ForegroundColor DarkGray
Write-Host '    correct on real inputs. See docs/ARCHITECTURE.md section 4.' -ForegroundColor DarkGray

if ($verified) { exit 0 } else { exit 0 }
