# Releasing both model artifacts

Prerequisites: install GitHub CLI, run `gh auth login`, and execute these commands from the Koda repository root.

```bash
set -euo pipefail

sha256sum -c <<'CHECKSUMS'
ccc3af877fe51a27c729aa3837604bbf21a4119241c2e1fd8e5d914ab835d95d  models/koda-3.pkl
9852780c484b358c3a297aee1aa0ad825b015fb2d4199bf8b792f3939ac8a642  models/koda-waf.pkl
CHECKSUMS

git push origin main
git tag -a koda-models-v2026.09.04 -m "Koda-3 and Koda-WAF models"
git push origin koda-models-v2026.09.04

gh release create koda-models-v2026.09.04 \
  'models/koda-3.pkl#Koda-3 model' \
  'models/koda-waf.pkl#Koda-WAF model' \
  --verify-tag \
  --title 'Koda-3 and Koda-WAF models' \
  --notes-file RELEASE_NOTES.md
```

The `.pkl` files are ignored by Git and are uploaded only as release assets.

