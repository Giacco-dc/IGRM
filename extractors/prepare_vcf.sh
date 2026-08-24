#!/usr/bin/env bash
set -euo pipefail
input=${1:?input VCF required}
output=${2:?output VCF required}
add_chr=${3:-true}
grep '^#' "$input" > "$output"
if [[ "$add_chr" == true ]]; then
    grep -v '^#' "$input" | awk 'BEGIN {FS=OFS="\t"} {if ($1 !~ /^chr/) $1="chr"$1; print}' | sort -k1,1V -k2,2n >> "$output"
else
    grep -v '^#' "$input" | sort -k1,1V -k2,2n >> "$output"
fi
bgzip -f -c "$output" > "${output}.gz"
tabix -f -p vcf "${output}.gz"
ls -lh "${output}.gz" "${output}.gz.tbi"
