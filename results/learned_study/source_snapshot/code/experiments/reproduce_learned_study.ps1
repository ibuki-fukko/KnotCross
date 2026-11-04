param(
    [string]$PythonPath = 'E:\study\python\python.exe',
    [string]$OutputDirectory = 'results/learned_study_reproduction',
    [switch]$FinalEvaluation
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $projectRoot
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NURBS_STUDY_DIR = Join-Path $projectRoot $OutputDirectory
New-Item -ItemType Directory -Path $env:NURBS_STUDY_DIR -Force | Out-Null

& $PythonPath code/experiments/check_learned_correction.py
if ($LASTEXITCODE -ne 0) { throw 'Numerical checks failed' }
$stages = @('audit', 'depth', 'steps', 'scale', 'positive', 'geometry', 'duration',
            'regression', 'reference', 'depth_revisit', 'dynamic', 'dynamic_depth',
            'dynamic_steps', 'refresh_duration', 'refresh_depth', 'refresh_steps',
            'refresh_scale', 'refresh_rate', 'refresh_regression', 'seeds')
foreach ($stage in $stages) {
    & $PythonPath -u code/experiments/controlled_study.py --stage $stage 2>&1 |
        Tee-Object -FilePath (Join-Path $env:NURBS_STUDY_DIR "$stage.log")
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $stage" }
}
foreach ($action in @('isolation', 'diagnose', 'reproduce')) {
    & $PythonPath -u code/experiments/study_audits.py --action $action
    if ($LASTEXITCODE -ne 0) { throw "Audit failed: $action" }
}
if ($FinalEvaluation) {
    & $PythonPath -u code/experiments/controlled_study.py --stage final
    if ($LASTEXITCODE -ne 0) { throw 'Final evaluation failed' }
    & $PythonPath -u code/experiments/study_audits.py --action analytic
    if ($LASTEXITCODE -ne 0) { throw 'Analytic evaluation failed' }
}
Remove-Item Env:NURBS_STUDY_DIR
