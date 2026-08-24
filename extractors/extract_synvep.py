#!/usr/bin/env python3
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_MAP = {'observed': 1.0, 'singleton': 0.5, 'not_seen': 0.0, 'notseen': 0.0}


def read_vcf(path):
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            fields = line.rstrip().split('\t')
            if len(fields) < 5:
                continue
            chrom = fields[0].removeprefix('chr').removeprefix('Chr')
            for alt in fields[4].split(','):
                rows.append((chrom, int(fields[1]), fields[3].upper(), alt.upper()))
    return rows


def encode_score(value):
    if value is None or value == '':
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return SCORE_MAP.get(str(value).strip().lower(), np.nan)


def extract(vcf, database):
    query = 'SELECT class, HGNC_gene_symbol, dbSNP_ID FROM VARIANT_SCORE WHERE chr=? AND pos=? AND ref=? AND alt=? LIMIT 1'
    rows = []
    with sqlite3.connect(database) as connection:
        cursor = connection.cursor()
        for chrom, pos, ref, alt in read_vcf(vcf):
            cursor.execute(query, (chrom, pos, ref, alt))
            record = cursor.fetchone()
            rows.append({
                'chr': chrom,
                'pos': pos,
                'ref': ref,
                'alt': alt,
                'synvep_score': encode_score(record[0]) if record else np.nan,
                'has_gene': int(bool(record and record[1])),
                'has_dbsnp': int(bool(record and record[2])),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--database', default='/home/cc/synvep_local/synvep_database_v1.1.db')
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = extract(args.input, args.database)
    result.to_csv(output, index=False)
    print(f'{output}: {len(result)} variants')


if __name__ == '__main__':
    main()
