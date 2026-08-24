#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

import pandas as pd


FEATURE_MAP = {
    'GERP++_RS': 'GerpRS',
    'phyloP100way_vertebrate': 'verPhyloP',
    'phyloP30way_mammalian': 'mamPhyloP',
    'phastCons100way_vertebrate': 'verPhCons',
    'phastCons30way_mammalian': 'mamPhCons',
    'SiPhy_29way_logOdds': 'siPhy_rankscore',
    'GenoCanyon_score': 'GenoCanyon_Score',
    'ReMM_score': 'ReMM_Score',
    'integrated_fitCons_score': 'integrated_fitCons_score',
    'integrated_confidence_value': 'integrated_confidence_value',
    'integrated_fitCons_score_rankscore': 'integrated_fitCons_score_rankscore',
    'GM12878_fitCons_score': 'GM12878_fitCons_score',
    'GM12878_confidence_value': 'GM12878_confidence_value',
    'GM12878_fitCons_score_rankscore': 'GM12878_fitCons_score_rankscore',
    'H1-hESC_fitCons_score': 'H1-hESC_fitCons_score',
    'H1-hESC_confidence_value': 'H1-hESC_confidence_value',
    'H1-hESC_fitCons_score_rankscore': 'H1-hESC_fitCons_score_rankscore',
    'HUVEC_fitCons_score': 'HUVEC_fitCons_score',
    'HUVEC_confidence_value': 'HUVEC_confidence_value',
    'HUVEC_fitCons_score_rankscore': 'HUVEC_fitCons_score_rankscore',
    'SpliceAI_pred_DS_AG': 'SpliceAI-acc-gain',
    'SpliceAI_pred_DS_AL': 'SpliceAI-acc-loss',
    'SpliceAI_pred_DS_DG': 'SpliceAI-don-gain',
    'SpliceAI_pred_DS_DL': 'SpliceAI-don-loss',
    'bStatistic': 'bStatistic',
}


def numeric(series):
    return pd.to_numeric(series.astype(str).str.split(';').str[0], errors='coerce')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--annovar', default='/home/cc/variant_annotation/data/annovar')
    parser.add_argument('--build', default='hg19')
    args = parser.parse_args()
    annovar = Path(args.annovar)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.parent / f'{output.stem}.annovar'
    command = [
        'perl', str(annovar / 'table_annovar.pl'), args.input, str(annovar / 'humandb'),
        '-buildver', args.build, '-out', str(prefix), '-remove', '-protocol', 'dbnsfp42a',
        '-operation', 'f', '-nastring', '.', '-vcfinput', '-polish',
    ]
    subprocess.run(command, check=True)
    annotated = Path(f'{prefix}.{args.build}_multianno.txt')
    source = pd.read_csv(annotated, sep='\t', low_memory=False)
    result = source[['Chr', 'Start', 'Ref', 'Alt']].rename(columns={'Chr': 'chr', 'Start': 'pos', 'Ref': 'ref', 'Alt': 'alt'})
    for source_name, output_name in FEATURE_MAP.items():
        if source_name in source:
            result[output_name] = numeric(source[source_name])
    result.to_csv(output, index=False)
    print(f'{output}: {len(result)} variants, {len(result.columns) - 4} features')


if __name__ == '__main__':
    main()
