#!/usr/bin/env python3
import argparse
import os, sys, subprocess, gzip
import pandas as pd
import numpy as np
from collections import defaultdict
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=('case', 'control'), default='case')
parser.add_argument('--input')
parser.add_argument('--output')
parser.add_argument('--rmsk', default='/home/cc/variant_annotation/feature_data/rmsk.txt.gz')
parser.add_argument('--dbnsfp', default='/home/cc/variant_annotation/tools/annovar/humandb/hg19_dbnsfp42a.txt')
args = parser.parse_args()
MERGED = args.input or f"/mnt/c/Users/GCC/Desktop/case study/data/{args.dataset}_variants.csv"
RMSK = args.rmsk
DBNSFP = args.dbnsfp
OUT_CSV = args.output or f"/mnt/c/Users/GCC/Desktop/case study/feature/{args.dataset}_vcf_extracted/{args.dataset}_te_fitcons.csv"
RMSK_CHROM = 5
RMSK_START = 6
RMSK_END = 7
RMSK_CLASS = 11
TE_CLASSES = {'SINE', 'LINE', 'LTR', 'DNA', 'RC', 'Retroposon'}
FITCONS_TARGETS = {'GM12878_fitCons_score': 'GM12878_fitCons_score', 'GM12878_fitCons_score_rankscore': 'GM12878_fitCons_score_rankscore', 'H1-hESC_fitCons_score': 'H1-hESC_fitCons_score', 'H1-hESC_fitCons_score_rankscore': 'H1-hESC_fitCons_score_rankscore', 'HUVEC_fitCons_score': 'HUVEC_fitCons_score', 'HUVEC_fitCons_score_rankscore': 'HUVEC_fitCons_score_rankscore'}

def load_variants():
    df = pd.read_csv(MERGED, usecols=['chr', 'pos', 'ref', 'alt'])
    variants = []
    for _, r in df.iterrows():
        c = str(r['chr']).replace('chr', '')
        variants.append({'chr': c, 'pos': int(r['pos']), 'ref': str(r['ref']), 'alt': str(r['alt']), 'key': f"{c}_{r['pos']}_{r['ref']}/{r['alt']}"})
    return variants

def build_rmsk_index(variants):
    chroms_needed = set(('chr' + v['chr'] for v in variants))
    print(f"  需要染色体: {len(chroms_needed)}")
    intervals = defaultdict(list)
    n = 0
    with gzip.open(RMSK, 'rt') as f:
        for line in f:
            p = line.split('\t')
            if len(p) < 13:
                continue
            chrom = p[RMSK_CHROM]
            if chrom not in chroms_needed:
                continue
            te_class = p[RMSK_CLASS]
            if te_class not in TE_CLASSES:
                continue
            try:
                start = int(p[RMSK_START])
                end = int(p[RMSK_END])
            except:
                continue
            intervals[chrom].append((start, end))
            n += 1
    print(f"  加载 TE 区间: {n} 个")
    return intervals

def is_in_te(intervals, chrom, pos):
    pos0 = pos - 1
    key = f"chr{chrom}"
    for start, end in intervals.get(key, []):
        if start <= pos0 < end:
            return 1
    return 0

def get_dbnsfp_fitcons(variants):
    result = {v['key']: {} for v in variants}
    if not os.path.exists(DBNSFP):
        print('  dbNSFP 不存在，跳过')
        return result
    r = subprocess.run(['head', '-1', DBNSFP], capture_output=True, text=True, timeout=30)
    headers = r.stdout.strip().split('\t')
    col_idx = {}
    for i, h in enumerate(headers):
        if h in FITCONS_TARGETS:
            col_idx[h] = i
    print(f"  dbNSFP fitCons cell-line 列: {col_idx}")
    if not col_idx:
        print('  dbNSFP v4.2a 中没有 cell-line specific fitCons (只有 integrated)')
        return result
    for v in variants:
        chrom = v['chr']
        pos = v['pos']
        cmd = ['grep', '-m1', f"^{chrom}\t{pos}\t", DBNSFP]
        try:
            r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r2.stdout:
                fields = r2.stdout.strip().split('\n')[0].split('\t')
                for col, idx in col_idx.items():
                    if idx < len(fields):
                        val = fields[idx]
                        if val not in ('.', '', 'nan'):
                            try:
                                result[v['key']][col] = float(val.split(';')[0])
                            except:
                                pass
        except:
            pass
    return result

def main():
    print('=' * 60)
    print('TE / fitCons cell-line 特征提取')
    print('=' * 60)
    variants = load_variants()
    print(f"变异数: {len(variants)}")
    print(f"\n[1] 加载 rmsk.txt.gz: {RMSK}")
    intervals = build_rmsk_index(variants)
    print('\n[2] 计算 TE 特征...')
    te_vals = {}
    for v in variants:
        te_vals[v['key']] = is_in_te(intervals, v['chr'], v['pos'])
    n_te = sum(te_vals.values())
    print(f"  TE=1 的变异数: {n_te}/{len(variants)}")
    print(f"\n[3] 查询 dbNSFP fitCons cell-line: {DBNSFP}")
    fitcons_vals = get_dbnsfp_fitcons(variants)
    rows = []
    for v in variants:
        row = {'Variant19': v['key'], 'TE': te_vals.get(v['key'], np.nan)}
        fc = fitcons_vals.get(v['key'], {})
        for col in FITCONS_TARGETS.values():
            row[col] = fc.get(col, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n完成! 输出: {OUT_CSV}")
    print(f"维度: {df.shape}")
    for c in ['TE'] + list(FITCONS_TARGETS.values()):
        if c in df.columns:
            n = df[c].notna().sum()
            print(f"  {c}: {n}/{len(df)}")
if __name__ == '__main__':
    main()
