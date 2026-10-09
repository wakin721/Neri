$ErrorActionPreference = 'Stop'
$pythonExe = 'C:\Users\wakin\AppData\Local\Programs\Python\Python313\python.exe'
$sourceRoot = 'F:\files\python\Neri_plus'
$runRoot = 'E:\Files\Neri\runs\seq_memory_reviewed_20261009'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'

# Outputs are immutable. Change runRoot and the preparation script OUT for a new run.
Push-Location -LiteralPath $sourceRoot
try {
    & $pythonExe -X utf8 -u -m training.extract_incremental_memory_features `
        --data "$sourceRoot\data\by_species\00已校验" `
        --manifest "$runRoot\classification_bank.jsonl" `
        --previous "$sourceRoot\runs\memory_head_verified_20260927\features" `
        --encoder "$sourceRoot\models\dinov2-base" `
        --out "$runRoot\features" --batch-size 16
    if ($LASTEXITCODE -ne 0) { throw 'Feature extraction failed' }
    & $pythonExe -X utf8 -u -m training.export_verified_standard_memory `
        --features "$runRoot\features" `
        --config-head "$sourceRoot\runs\training_free_head_20260921\deployment\memory_head.npz" `
        --out "$runRoot\deployment" --target-frr 0.04 --seed 20260923
    if ($LASTEXITCODE -ne 0) { throw 'Memory head export failed' }
} finally {
    Pop-Location
}
