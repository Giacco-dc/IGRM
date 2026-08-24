#!/usr/bin/env python3
import argparse
import os, sys, re, subprocess
import numpy as np
import pandas as pd
from collections import defaultdict
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=('case', 'control'), default='case')
parser.add_argument('--input')
parser.add_argument('--output')
parser.add_argument('--fasta', default='/home/cc/data/hg19.fa')
parser.add_argument('--gtf', default='/home/cc/data/gencode.v19.annotation.gtf')
parser.add_argument('--silva', default='/home/cc/data/silva/silva-1.1.1')
args = parser.parse_args()
import pysam
MERGED_CSV = args.input or f"/mnt/c/Users/GCC/Desktop/case study/data/{args.dataset}_variants.csv"
OUTPUT_CSV = args.output or f"/mnt/c/Users/GCC/Desktop/case study/feature/{args.dataset}_vcf_extracted/{args.dataset}_silva_features.csv"
REF_FASTA = args.fasta
GENCODE_GTF = args.gtf
SILVA_DIR = args.silva
FAS_HEX = os.path.join(SILVA_DIR, 'src/features/fas-ess/fas-hex3.txt')
PESE_TXT = os.path.join(SILVA_DIR, 'src/features/pesx/pese262.txt')
PESS_TXT = os.path.join(SILVA_DIR, 'src/features/pesx/pess262.txt')
MAXENT5_PL = os.path.join(SILVA_DIR, 'src/features/maxent/score5.pl')
MAXENT3_PL = os.path.join(SILVA_DIR, 'data/score3.pl')

def load_variants():
    df = pd.read_csv(MERGED_CSV, usecols=['chr', 'pos', 'ref', 'alt'])
    variants = []
    for _, r in df.iterrows():
        c = str(r['chr']).replace('chr', '')
        variants.append({'chr': c, 'pos': int(r['pos']), 'ref': str(r['ref']).upper(), 'alt': str(r['alt']).upper(), 'key': f"{c}_{r['pos']}_{r['ref']}/{r['alt']}"})
    return variants

def load_exons_from_gtf():
    print('读取 GTF 外显子注释...')
    exons = defaultdict(list)
    with open(GENCODE_GTF) as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.split('\t')
            if len(p) < 9 or p[2] != 'exon':
                continue
            chrom = p[0].replace('chr', '')
            start = int(p[3])
            end = int(p[4])
            strand = p[6]
            m = re.search('transcript_id "([^"]+)"', p[8])
            tid = m.group(1) if m else '.'
            exons[chrom].append((start, end, strand, tid))
    print(f"  加载 {sum((len(v) for v in exons.values()))} 个外显子")
    return exons

def get_exon_context(v, exons, fasta, flank=50):
    chrom = v['chr']
    pos = v['pos']
    ref = v['ref']
    alt = v['alt']
    cands = exons.get(chrom, [])
    best = None
    for start, end, strand, tid in cands:
        if start <= pos <= end:
            best = (start, end, strand)
            break
    if best is None:
        return (None, None)
    ex_start, ex_end, strand = best
    refs_to_try = [chrom, f"chr{chrom}"]
    seq = None
    for c in refs_to_try:
        if c in fasta.references:
            try:
                seq = fasta.fetch(c, ex_start - 1, ex_end).upper()
            except:
                pass
            break
    if not seq:
        return (None, None)
    mut_offset = pos - ex_start
    if mut_offset < 0 or mut_offset >= len(seq):
        return (None, None)
    if seq[mut_offset] != ref:
        pass
    up_seq = ''
    dn_seq = ''
    up_c = refs_to_try[0]
    for c in refs_to_try:
        if c in fasta.references:
            up_c = c
            break
    try:
        if ex_start > 1:
            up_seq = fasta.fetch(up_c, max(0, ex_start - 1 - flank), ex_start - 1).upper()
        if ex_end < fasta.get_reference_length(up_c):
            dn_seq = fasta.fetch(up_c, ex_end, min(ex_end + flank, fasta.get_reference_length(up_c))).upper()
    except:
        pass
    pre = seq[:mut_offset]
    post = seq[mut_offset + 1:]
    mut_exon = f"{pre}[{ref}/{alt}]{post}"
    silva_seq = f"{up_seq}|{mut_exon}|{dn_seq}"
    premrna_len = len(up_seq) + len(seq) + len(dn_seq)
    mrna_len = len(seq)
    pre_pos = len(up_seq) + mut_offset
    post_pos = premrna_len - pre_pos - 1
    f_premrna = min(pre_pos, post_pos) / (pre_pos + post_pos + 1)
    pre_mrna = len(pre)
    post_mrna = len(post)
    f_mrna = min(pre_mrna, post_mrna) / (pre_mrna + post_mrna + 1)
    return (silva_seq, {'f_premrna': f_premrna, 'f_mrna': f_mrna, 'pre': pre, 'post': post, 'ref': ref, 'alt': alt, 'exon_seq': seq, 'up_seq': up_seq, 'dn_seq': dn_seq, 'strand': strand})

def load_hexamers(path):
    hexs = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) == 6:
                hexs.add(line.upper())
    print(f"  加载 hexamers: {len(hexs)} (from {path})")
    return hexs

def load_octamers(path):
    octs = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if len(line) == 8:
                octs.add(line.upper())
    print(f"  加载 octamers: {len(octs)} (from {path})")
    return octs

def subseq_hits(motif_set, seq, motif_len):
    found = set()
    for i in range(len(seq) - motif_len + 1):
        if seq[i:i + motif_len] in motif_set:
            found.add(i)
    return found

def safe_div(num, denom):
    if num + denom > 0:
        return num / (num + denom)
    return np.nan

def calc_fas6(hexs, pre, ref, alt, post):
    short_ref = (pre[-7:] if len(pre) >= 7 else pre) + ref + (post[:7] if len(post) >= 7 else post)
    short_alt = (pre[-7:] if len(pre) >= 7 else pre) + alt + (post[:7] if len(post) >= 7 else post)
    old_full = pre + ref + post
    hits_old_full = subseq_hits(hexs, old_full, 6)
    hits_old_win = subseq_hits(hexs, short_ref, 6)
    hits_new_win = subseq_hits(hexs, short_alt, 6)
    n_old = len(hits_old_full)
    n_lost = len(hits_old_win - hits_new_win)
    n_gain = len(hits_new_win - hits_old_win)
    return (safe_div(n_lost, n_old), safe_div(n_gain, n_old))

def calc_pesx(pese, pess, pre, ref, alt, post):
    short_ref = (pre[-7:] if len(pre) >= 7 else pre) + ref + (post[:7] if len(post) >= 7 else post)
    short_alt = (pre[-7:] if len(pre) >= 7 else pre) + alt + (post[:7] if len(post) >= 7 else post)
    old_full = pre + ref + post
    n_pese = len(subseq_hits(pese, old_full, 8))
    n_pess = len(subseq_hits(pess, old_full, 8))
    pese_old = subseq_hits(pese, short_ref, 8)
    pese_new = subseq_hits(pese, short_alt, 8)
    pess_old = subseq_hits(pess, short_ref, 8)
    pess_new = subseq_hits(pess, short_alt, 8)
    pese_lost = len(pese_old - pese_new)
    pese_gain = len(pese_new - pese_old)
    pess_lost = len(pess_old - pess_new)
    pess_gain = len(pess_new - pess_old)
    return (safe_div(pese_lost, n_pese), safe_div(pese_gain, n_pese), safe_div(pess_lost, n_pess), safe_div(pess_gain, n_pess))

def score_maxent5(seq9):
    score5_pl = os.path.join(SILVA_DIR, 'data/score5.pl')
    if not os.path.exists(score5_pl):
        return None
    try:
        r = subprocess.run(['perl', score5_pl, seq9], capture_output=True, text=True, timeout=5)
        line = r.stdout.strip()
        if line:
            return float(line.split()[-1])
    except:
        pass
    return None

def score_maxent3(seq23):
    score3_pl = os.path.join(SILVA_DIR, 'data/score3.pl')
    if not os.path.exists(score3_pl):
        return None
    try:
        r = subprocess.run(['perl', score3_pl, seq23], capture_output=True, text=True, timeout=5)
        line = r.stdout.strip()
        if line:
            return float(line.split()[-1])
    except:
        pass
    return None

def calc_mec(ctx):
    pre = ctx['pre']
    post = ctx['post']
    ref = ctx['ref']
    alt = ctx['alt']
    seq_ref = pre + ref + post
    seq_alt = pre + alt + post
    mut_pos = len(pre)
    GOOD_SCORE = 2.0

    def find_5ss(seq, mut_pos):
        hits = []
        for m in re.finditer('GT', seq):
            pos = m.start()
            if abs(pos - mut_pos) <= 9:
                win_start = pos - 3
                win_end = pos + 6
                if win_start >= 0 and win_end <= len(seq):
                    hits.append((pos, seq[win_start:win_end]))
        return hits

    def find_3ss(seq, mut_pos):
        hits = []
        for m in re.finditer('AG', seq):
            pos = m.start()
            if abs(pos - mut_pos) <= 23:
                win_start = pos - 20
                win_end = pos + 3
                if win_start >= 0 and win_end <= len(seq):
                    hits.append((pos, seq[win_start:win_end]))
        return hits
    sites5_ref = find_5ss(seq_ref, mut_pos)
    sites5_alt = find_5ss(seq_alt, mut_pos)
    mec_mc = 0
    mec_cs = 0
    for pos, win in sites5_ref:
        score = score_maxent5(win)
        if score is not None and score >= GOOD_SCORE:
            mec_cs = 1
    for pos, win in sites5_alt:
        score = score_maxent5(win)
        if score is not None and score >= GOOD_SCORE:
            mec_mc = 1
    if mec_cs != mec_mc:
        mec_mc = 1
    return (int(mec_mc), int(mec_cs))

def main():
    print('=' * 60)
    print('SILVA 特征提取 (FAS6/PESE/PESS/f_mrna/MEC)')
    print('=' * 60)
    variants = load_variants()
    print(f"变异数: {len(variants)}")
    print('\n加载数据库文件...')
    hexs = load_hexamers(FAS_HEX)
    pese = load_octamers(PESE_TXT)
    pess = load_octamers(PESS_TXT)
    print(f"\n加载 hg19: {REF_FASTA}")
    fasta = pysam.FastaFile(REF_FASTA)
    exons = load_exons_from_gtf()
    results = []
    matched = 0
    for i, v in enumerate(variants):
        feat = {'Variant19': v['key']}
        silva_seq, ctx = get_exon_context(v, exons, fasta)
        if ctx is not None:
            matched += 1
            pre = ctx['pre']
            post = ctx['post']
            ref = ctx['ref']
            alt = ctx['alt']
            feat['f_mrna'] = ctx['f_mrna']
            feat['f_premrna'] = ctx['f_premrna']
            fas6m, fas6p = calc_fas6(hexs, pre, ref, alt, post)
            feat['FAS6-'] = fas6m
            feat['FAS6+'] = fas6p
            pese_m, pese_p, pess_m, pess_p = calc_pesx(pese, pess, pre, ref, alt, post)
            feat['PESE-'] = pese_m
            feat['PESE+'] = pese_p
            feat['PESS-'] = pess_m
            feat['PESS+'] = pess_p
            mec_mc, mec_cs = calc_mec(ctx)
            feat['MEC-MC?'] = mec_mc
            feat['MEC-CS?'] = mec_cs
        else:
            for k in ['f_mrna', 'f_premrna', 'FAS6-', 'FAS6+', 'PESE-', 'PESE+', 'PESS-', 'PESS+', 'MEC-MC?', 'MEC-CS?']:
                feat[k] = np.nan
        results.append(feat)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(variants)}, exon匹配: {matched}")
    fasta.close()
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    n_filled = df['f_mrna'].notna().sum()
    print(f"\n完成! 匹配外显子: {matched}/{len(variants)}")
    print(f"输出: {OUTPUT_CSV}")
    print(f"维度: {df.shape}")
    print(f"f_mrna 非空: {n_filled}")
    for c in ['FAS6-', 'FAS6+', 'PESE-', 'PESE+', 'PESS-', 'PESS+', 'f_mrna', 'MEC-MC?', 'MEC-CS?']:
        if c in df.columns:
            print(f"  {c}: {df[c].notna().sum()}/{len(df)}")
if __name__ == '__main__':
    main()
