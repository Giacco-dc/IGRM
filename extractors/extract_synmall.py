#!/usr/bin/env python3
import argparse
import os, sys, math, re, gzip
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
SUITE_DIR = '/home/cc/data/feature_suite'
sys.path.insert(0, SUITE_DIR)
REF_FASTA = '/home/cc/data/hg19.fa'
GENCODE_GTF = '/home/cc/data/gencode.v19.annotation.gtf'
FEAT_DATA = '/home/cc/variant_annotation/feature_data'
SYNVEP_DIR = '/home/cc/variant_annotation/Feature/synVEP'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=('case', 'control'), default='case')
parser.add_argument('--input')
parser.add_argument('--output')
parser.add_argument('--fasta', default=REF_FASTA)
parser.add_argument('--gtf', default=GENCODE_GTF)
parser.add_argument('--feature-data', default=FEAT_DATA)
parser.add_argument('--synvep-dir', default=SYNVEP_DIR)
parser.add_argument('--suite-dir', default=SUITE_DIR)
args = parser.parse_args()
import pysam
SUITE_DIR = args.suite_dir
REF_FASTA = args.fasta
GENCODE_GTF = args.gtf
FEAT_DATA = args.feature_data
SYNVEP_DIR = args.synvep_dir
default_input = f"/mnt/c/Users/GCC/Desktop/case study/data/{args.dataset}_variants.csv"
default_output = f"/mnt/c/Users/GCC/Desktop/case study/feature/{args.dataset}_vcf_extracted/{args.dataset}_features_268.csv"
MERGED_CSV = args.input or default_input
OUTPUT_CSV = args.output or default_output

def load_variants():
    df = pd.read_csv(MERGED_CSV, usecols=['chr', 'pos', 'ref', 'alt'])
    variants = []
    for _, r in df.iterrows():
        key = f"{r['chr']}_{r['pos']}_{r['ref']}/{r['alt']}"
        variants.append({'key': key, 'chr': str(r['chr']), 'pos': int(r['pos']), 'ref': str(r['ref']), 'alt': str(r['alt'])})
    print(f"读取变异: {len(variants)} 个")
    return variants

def extract_dna_features(variants, fasta):
    print('\n[1] DNA 序列特征...')
    TRANSITIONS = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
    DINUCS = ['AA', 'AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    GRANTHAM = {('A', 'R'): 112, ('A', 'N'): 111, ('A', 'D'): 126, ('A', 'C'): 195, ('A', 'Q'): 91, ('A', 'E'): 107, ('A', 'G'): 60, ('A', 'H'): 86, ('A', 'I'): 94, ('A', 'L'): 96, ('A', 'K'): 106, ('A', 'M'): 84, ('A', 'F'): 113, ('A', 'P'): 27, ('A', 'S'): 99, ('A', 'T'): 58, ('A', 'W'): 148, ('A', 'Y'): 112, ('A', 'V'): 64, ('R', 'A'): 112, ('R', 'N'): 86, ('R', 'D'): 96, ('R', 'C'): 180, ('R', 'Q'): 43, ('R', 'E'): 54, ('R', 'G'): 125, ('R', 'H'): 29, ('R', 'I'): 97, ('R', 'L'): 102, ('R', 'K'): 26, ('R', 'M'): 91, ('R', 'F'): 97, ('R', 'P'): 103, ('R', 'S'): 110, ('R', 'T'): 71, ('R', 'W'): 101, ('R', 'Y'): 77, ('R', 'V'): 96, ('C', 'A'): 195, ('C', 'R'): 180, ('C', 'N'): 139, ('C', 'D'): 154, ('C', 'Q'): 154, ('C', 'E'): 170, ('C', 'G'): 159, ('C', 'H'): 174, ('C', 'I'): 198, ('C', 'L'): 198, ('C', 'K'): 202, ('C', 'M'): 196, ('C', 'F'): 205, ('C', 'P'): 169, ('C', 'S'): 112, ('C', 'T'): 149, ('C', 'W'): 215, ('C', 'Y'): 194, ('C', 'V'): 192}
    HYDRO = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
    POLAR = {'N', 'Q', 'S', 'T', 'Y', 'C', 'H'}
    CODON_AA = {'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L', 'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*', 'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'}
    results = []
    for i, v in enumerate(variants):
        feat = {}
        chrom = v['chr']
        pos = v['pos']
        ref = v['ref'].upper()
        alt = v['alt'].upper()
        seq = None
        for c in [chrom, f"chr{chrom}", chrom.replace('chr', '')]:
            if c in fasta.references:
                try:
                    start = max(0, pos - 101)
                    end = pos + 100
                    seq = fasta.fetch(c, start, end).upper()
                except:
                    pass
                break
        mid = min(100, pos - max(0, pos - 101) - 1)
        if seq and len(seq) >= 10:
            seq50 = seq[max(0, mid - 25):mid + 25]
            seq100 = seq[max(0, mid - 50):mid + 50]
            n50 = max(1, len(seq50))
            n100 = max(1, len(seq100))
            feat['GC_content_50bp'] = (seq50.count('G') + seq50.count('C')) / n50
            feat['GC_content_100bp'] = (seq100.count('G') + seq100.count('C')) / n100
            feat['CpG_density'] = seq100.count('CG') / max(1, n100 - 1)
            feat['GC_Content_Local'] = (seq.count('G') + seq.count('C')) / len(seq)
            for di in DINUCS:
                feat[f"dinuc_{di}_freq"] = seq100.count(di) / max(1, n100 - 1)
            base_counts = Counter(seq100)
            total = sum(base_counts.values())
            ent = -sum((c / total * math.log2(c / total) for c in base_counts.values() if c > 0))
            feat['sequence_entropy'] = ent
            feat['Seq_Entropy'] = ent
            kmers = {seq100[k:k + 3] for k in range(len(seq100) - 2)}
            feat['Sequence_Complexity'] = len(kmers) / min(64, max(1, len(seq100) - 2))
            max_homo = curr = 1
            for j in range(1, len(seq100)):
                if seq100[j] == seq100[j - 1]:
                    curr += 1
                    max_homo = max(max_homo, curr)
                else:
                    curr = 1
            feat['In_Homopolymer'] = int(max_homo >= 4)
            di_rep = 0
            for j in range(0, len(seq100) - 3, 2):
                if seq100[j:j + 2] == seq100[j + 2:j + 4]:
                    di_rep = 1
                    break
            feat['In_Dinuc_Repeat'] = di_rep
            local = seq[max(0, mid - 10):mid + 10]
            feat['Near_Palindrome'] = int(local == local[::-1]) if len(local) >= 4 else 0
            feat['G4_Potential'] = min(1.0, seq100.count('GGG') / 4.0) if seq100.count('GGG') >= 2 else 0
            g = seq100.count('G')
            c_ = seq100.count('C')
            feat['Seq_GC_Skew'] = (g - c_) / max(1, g + c_)
            ref_gc = int(ref in 'GC')
            alt_gc = int(alt in 'GC')
            feat['GC_Delta'] = alt_gc - ref_gc
            if 0 < mid < len(seq) - 1:
                ref_ctx = seq[mid - 1:mid + 2]
                alt_seq = seq[:mid] + alt + seq[mid + 1:]
                alt_ctx = alt_seq[mid - 1:mid + 2]
                feat['CpG_Delta'] = alt_ctx.count('CG') - ref_ctx.count('CG')
                feat['Ctx_CpG_Loss'] = int(ref_ctx.count('CG') > alt_ctx.count('CG'))
                trinuc = ref_ctx if len(ref_ctx) == 3 else ''
                feat['trinuc_context'] = trinuc
            cpb_ref = sum((1 for k in range(len(seq100) - 1) if seq100[k:k + 2] == 'CG')) / max(1, n100 - 1)
            alt_seq2 = seq[:mid] + alt + seq[mid + 1:]
            s2 = alt_seq2[max(0, mid - 50):mid + 50]
            cpb_alt = sum((1 for k in range(len(s2) - 1) if s2[k:k + 2] == 'CG')) / max(1, len(s2) - 1)
            feat['CPB_Delta'] = cpb_alt - cpb_ref
            k7_ref = {seq100[k:k + 7] for k in range(len(seq100) - 6)}
            alt_s = seq[:mid] + alt + seq[mid + 1:]
            s2b = alt_s[max(0, mid - 50):mid + 50]
            k7_alt = {s2b[k:k + 7] for k in range(len(s2b) - 6)}
            feat['miRNA_Site_Change'] = len(k7_ref.symmetric_difference(k7_alt))
        else:
            for k in ['GC_content_50bp', 'GC_content_100bp', 'CpG_density', 'GC_Content_Local', 'sequence_entropy', 'Seq_Entropy', 'Sequence_Complexity', 'In_Homopolymer', 'In_Dinuc_Repeat', 'Near_Palindrome', 'G4_Potential', 'Seq_GC_Skew', 'GC_Delta', 'CpG_Delta', 'Ctx_CpG_Loss', 'trinuc_context', 'CPB_Delta', 'miRNA_Site_Change']:
                feat[k] = np.nan
            for di in DINUCS:
                feat[f"dinuc_{di}_freq"] = np.nan
        is_trans = (ref, alt) in TRANSITIONS
        feat['is_transition'] = int(is_trans)
        feat['is_transversion'] = int(not is_trans)
        for key in ['ref_hydrophobic', 'alt_hydrophobic', 'hydrophobicity_change', 'ref_polar', 'alt_polar', 'polarity_change', 'ref_charged']:
            feat[key] = np.nan
        results.append(feat)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(variants)}")
    print(f"  完成: {len(results)} 个变异")
    return pd.DataFrame(results)

def extract_translation_features(variants, fasta):
    print('\n[2] 翻译/密码子特征...')
    sys.path.insert(0, SUITE_DIR)
    try:
        import translation_lib as tlib
        has_tlib = True
    except ImportError:
        has_tlib = False
        print('  translation_lib 不可用，跳过')
        return pd.DataFrame()
    results = []
    for v in variants:
        feat = {}
        chrom = v['chr']
        pos = v['pos']
        ref = v['ref'].upper()
        alt = v['alt'].upper()
        seq = None
        for c in [chrom, f"chr{chrom}"]:
            if c in fasta.references:
                try:
                    start = max(0, pos - 51)
                    end = pos + 50
                    seq = fasta.fetch(c, start, end).upper()
                except:
                    pass
                break
        if seq and len(seq) >= 6:
            mid = min(50, pos - max(0, pos - 51) - 1)
            codon_start = mid - mid % 3
            codon_ref = seq[codon_start:codon_start + 3] if codon_start + 3 <= len(seq) else None
            if codon_ref and len(codon_ref) == 3:
                codon_alt = codon_ref[:mid - codon_start] + alt + codon_ref[mid - codon_start + 1:]
                codon_alt = codon_alt[:3]
                feat['tRNA_Copy_Ref'] = tlib.TRNA_COPY.get(codon_ref, 0)
                feat['tRNA_Copy_Alt'] = tlib.TRNA_COPY.get(codon_alt, 0)
                feat['Delta_tRNA_Copy'] = feat['tRNA_Copy_Alt'] - feat['tRNA_Copy_Ref']
                feat['Wobble_Ref'] = tlib.WOBBLE.get(codon_ref, 0)
                feat['Wobble_Alt'] = tlib.WOBBLE.get(codon_alt, 0)
                feat['Delta_Wobble'] = feat['Wobble_Alt'] - feat['Wobble_Ref']
                feat['CAI_Ref'] = tlib.CAI_WEIGHTS.get(codon_ref, 0.5)
                feat['CAI_Alt'] = tlib.CAI_WEIGHTS.get(codon_alt, 0.5)
                feat['Delta_CAI'] = feat['CAI_Alt'] - feat['CAI_Ref']
                ctx = seq[max(0, codon_start - 9):codon_start + 12]
                rare = sum((1 for k in range(0, len(ctx) - 2, 3) if tlib.CAI_WEIGHTS.get(ctx[k:k + 3], 0.5) < 0.3))
                feat['Rare_Codon_Cluster'] = int(rare >= 2)
                feat['Rare_Codon_Count'] = rare
                aa_seq = ''.join((tlib.CODON_TO_AA.get(ctx[k:k + 3], 'X') for k in range(0, len(ctx) - 2, 3)))
                feat['Poly_Pro_Context'] = int('PP' in aa_seq)
                aa_ref = tlib.CODON_TO_AA.get(codon_ref, 'X')
                aa_alt = tlib.CODON_TO_AA.get(codon_alt, 'X')
                feat['AA_Hydro_Ref'] = tlib.AA_HYDROPHOBICITY.get(aa_ref, 0)
                feat['AA_Hydro_Alt'] = tlib.AA_HYDROPHOBICITY.get(aa_alt, 0)
                feat['Delta_Hydro'] = feat['AA_Hydro_Alt'] - feat['AA_Hydro_Ref']
            else:
                for k in ['tRNA_Copy_Ref', 'tRNA_Copy_Alt', 'Delta_tRNA_Copy', 'Wobble_Ref', 'Wobble_Alt', 'Delta_Wobble', 'CAI_Ref', 'CAI_Alt', 'Delta_CAI', 'Rare_Codon_Cluster', 'Rare_Codon_Count', 'Poly_Pro_Context', 'AA_Hydro_Ref', 'AA_Hydro_Alt', 'Delta_Hydro']:
                    feat[k] = np.nan
        else:
            for k in ['tRNA_Copy_Ref', 'tRNA_Copy_Alt', 'Delta_tRNA_Copy', 'Wobble_Ref', 'Wobble_Alt', 'Delta_Wobble', 'CAI_Ref', 'CAI_Alt', 'Delta_CAI', 'Rare_Codon_Cluster', 'Rare_Codon_Count', 'Poly_Pro_Context', 'AA_Hydro_Ref', 'AA_Hydro_Alt', 'Delta_Hydro']:
                feat[k] = np.nan
        results.append(feat)
    print(f"  完成: {len(results)} 个")
    return pd.DataFrame(results)

def extract_rbp_rnamod_features(variants, fasta):
    print('\n[3] RBP / m6A / RNA 修饰特征 (序列预测)...')
    RBP_MOTIFS = {'RBFOX2': ['GCAUG', 'UGCAUG'], 'HNRNPC': ['TTTTT', 'TTTTTT'], 'PTBP1': ['TCTT', 'TCTCT'], 'SRSF1': ['GGAGGA', 'GAAGAA'], 'HNRNPA1': ['TAGGGA', 'TAGGG'], 'TARDBP': ['TGTGT', 'GTGT'], 'FUS': ['GGTG', 'GTGGT'], 'ELAVL1': ['ATTTA', 'TATTT'], 'MBNL1': ['TGCT', 'GCTT'], 'CELF1': ['TGTT', 'TGTGTG']}
    RNA_MOD = {'m6A': '[AGT][AG]AC[ACT]', 'm1A': 'G[TU][TU]C[AG]A', 'm5C': '[CG]{3,}', 'pseudoU': 'G[TU][TU]C'}

    def count_motifs(seq):
        hits = 0
        for motifs in RBP_MOTIFS.values():
            for m in motifs:
                hits += seq.count(m.replace('U', 'T'))
        return hits
    results = []
    for v in variants:
        feat = {}
        chrom, pos, ref, alt = (v['chr'], v['pos'], v['ref'].upper(), v['alt'].upper())
        seq = None
        for c in [chrom, f"chr{chrom}"]:
            if c in fasta.references:
                try:
                    seq = fasta.fetch(c, max(0, pos - 51), pos + 50).upper()
                except:
                    pass
                break
        if seq and len(seq) >= 10:
            mid = min(50, pos - max(0, pos - 51) - 1)
            seq_alt = seq[:mid] + alt + seq[mid + 1:]
            ref_hits = count_motifs(seq)
            alt_hits = count_motifs(seq_alt)
            feat['RBP_Motif_Hits_Ref'] = ref_hits
            feat['RBP_Motif_Delta'] = alt_hits - ref_hits
            feat['RBP_Sites_Gained'] = max(0, alt_hits - ref_hits)
            feat['RBP_Sites_Lost'] = max(0, ref_hits - alt_hits)
            feat['RBP_Max_Change'] = abs(alt_hits - ref_hits)
            feat['RBP_Any_Change'] = int(alt_hits != ref_hits)
            feat['RBP_Max_Gain'] = feat['RBP_Sites_Gained']
            feat['RBP_Max_Loss'] = feat['RBP_Sites_Lost']
            feat['RBP_Mean_Abs_Change'] = abs(alt_hits - ref_hits)
            m6a_ref = len(re.findall(RNA_MOD['m6A'], seq))
            m6a_alt = len(re.findall(RNA_MOD['m6A'], seq_alt))
            feat['m6A_Motif_Ref'] = m6a_ref
            feat['m6A_Motif_Delta'] = m6a_alt - m6a_ref
            feat['m6A_Site_Disrupted'] = int(m6a_ref > m6a_alt)
            feat['m6A_Site_Created'] = int(m6a_alt > m6a_ref)
            feat['m1A_Motif'] = len(re.findall(RNA_MOD['m1A'], seq))
            feat['m5C_Motif'] = len(re.findall(RNA_MOD['m5C'], seq))
            feat['pseudoU_Motif'] = len(re.findall(RNA_MOD['pseudoU'], seq))
            feat['RNA_Mod_Total_Ref'] = feat['m6A_Motif_Ref'] + feat['m1A_Motif'] + feat['m5C_Motif'] + feat['pseudoU_Motif']
            feat['RNA_Mod_Delta'] = feat['m6A_Motif_Delta']
            PAUSE_CODONS = {'CGA': 3.0, 'CGG': 2.5, 'AGG': 2.0, 'AGA': 1.8, 'CCG': 1.5, 'GCG': 1.5, 'ACG': 1.3}

            def ribo_score(s):
                return sum((PAUSE_CODONS.get(s[k:k + 3], 0) for k in range(0, len(s) - 2, 3)))
            feat['Ribo_Pause_Score_Ref'] = ribo_score(seq)
            feat['Ribo_Pause_Delta'] = ribo_score(seq_alt) - feat['Ribo_Pause_Score_Ref']
            feat['Ribo_Consecutive_Pause'] = int(any((seq[k:k + 3] in PAUSE_CODONS and seq[k + 3:k + 6] in PAUSE_CODONS for k in range(0, len(seq) - 5, 3))))
            CAI_COMMON = {'CTG': 1.0, 'ATG': 1.0, 'GTG': 0.9, 'AAG': 0.9, 'GAG': 0.9, 'GAA': 0.8, 'AAA': 0.8}
            feat['Translation_Rate_Ref'] = sum((CAI_COMMON.get(seq[k:k + 3], 0.3) for k in range(0, len(seq) - 2, 3))) / max(1, len(seq) // 3)
            feat['Translation_Rate_Delta'] = sum((CAI_COMMON.get(seq_alt[k:k + 3], 0.3) for k in range(0, len(seq_alt) - 2, 3))) / max(1, len(seq_alt) // 3) - feat['Translation_Rate_Ref']
            feat['Rare_Codon_Count'] = sum((1 for k in range(0, len(seq) - 2, 3) if CAI_COMMON.get(seq[k:k + 3], 0.3) < 0.4))
        else:
            for k in ['RBP_Motif_Hits_Ref', 'RBP_Motif_Delta', 'RBP_Sites_Gained', 'RBP_Sites_Lost', 'RBP_Max_Change', 'RBP_Any_Change', 'RBP_Max_Gain', 'RBP_Max_Loss', 'RBP_Mean_Abs_Change', 'm6A_Motif_Ref', 'm6A_Motif_Delta', 'm6A_Site_Disrupted', 'm6A_Site_Created', 'm1A_Motif', 'm5C_Motif', 'pseudoU_Motif', 'RNA_Mod_Total_Ref', 'RNA_Mod_Delta', 'Ribo_Pause_Score_Ref', 'Ribo_Pause_Delta', 'Ribo_Consecutive_Pause', 'Translation_Rate_Ref', 'Translation_Rate_Delta', 'Rare_Codon_Count']:
                feat[k] = np.nan
        results.append(feat)
    print(f"  完成: {len(results)} 个")
    return pd.DataFrame(results)

def extract_fitcons_features(variants):
    print('\n[4] fitCons 特征 (BigWig)...')
    try:
        import pyBigWig
    except ImportError:
        print('  pyBigWig 未安装，跳过')
        return pd.DataFrame()
    bw_map = {'integrated_fitCons_score': os.path.join(FEAT_DATA, 'fitcons_integrated.bw'), 'H1-hESC_fitCons_score': os.path.join(FEAT_DATA, 'fitcons_h1hesc.bw'), 'HUVEC_fitCons_score': os.path.join(FEAT_DATA, 'fitcons_huvec.bw')}
    bws = {}
    for name, path in bw_map.items():
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                bws[name] = pyBigWig.open(path)
                print(f"  已加载: {name}")
            except Exception as e:
                print(f"  [跳过] {name}: {e}")
        else:
            print(f"  [跳过] {path} 不存在或为空")
    if not bws:
        return pd.DataFrame()
    results = []
    for v in variants:
        feat = {}
        chrom = v['chr']
        pos = v['pos']
        for chrom_try in [chrom, f"chr{chrom}"]:
            for name, bw in bws.items():
                try:
                    vals = bw.values(chrom_try, pos - 1, pos)
                    feat[name] = vals[0] if vals and vals[0] is not None else np.nan
                except:
                    feat.setdefault(name, np.nan)
            if any((not np.isnan(feat.get(n, np.nan)) for n in bws)):
                break
        results.append(feat)
    for bw in bws.values():
        bw.close()
    print(f"  完成: {len(results)} 个")
    return pd.DataFrame(results)

def extract_remm_features(variants):
    print('\n[5] ReMM_Score (awk 精准提取)...')
    remm_path = os.path.join(FEAT_DATA, 'ReMM.v0.3.1.tsv.gz')
    if not os.path.exists(remm_path):
        print(f"  [跳过] {remm_path} 不存在")
        return pd.DataFrame()
    pos_set = set()
    for v in variants:
        chrom = str(v['chr']).replace('chr', '')
        pos_set.add(f"{chrom}:{v['pos']}")
    pos_file = '/tmp/remm_query_positions.txt'
    with open(pos_file, 'w') as f:
        for p in pos_set:
            f.write(p + '\n')
    lookup = {}
    print('  使用 awk 极其高效地流式读取 ReMM...')
    try:
        import subprocess
        awk_cmd = f"""awk 'FNR==NR {{pos[$1]=1; next}} (($1":"$2) in pos)' '{pos_file}' <(zcat '{remm_path}')"""
        proc = subprocess.Popen(['bash', '-c', awk_cmd], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for line in proc.stdout:
            line = line.decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    chrom = parts[0]
                    pos = int(parts[1])
                    val = float(parts[2]) if parts[2] not in ('.', '') else np.nan
                    lookup[chrom, pos] = val
                except ValueError:
                    continue
        proc.wait()
    except Exception as e:
        print(f"  awk 读取失败: {e}")
    results = []
    matched = 0
    for v in variants:
        k = (str(v['chr']).replace('chr', ''), v['pos'])
        score = lookup.get(k, np.nan)
        if not score != score:
            matched += 1
        results.append({'ReMM_Score': score})
    print(f"  匹配: {matched}/{len(variants)}")
    return pd.DataFrame(results)

def extract_synvep_features(variants):
    print('\n[6] synVEP 特征...')
    lookup = {}
    for split in ['train', 'test1', 'test2']:
        path = os.path.join(SYNVEP_DIR, f"{split}_synvep_features_numeric.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            k = f"{row['chr']}_{row['pos']}_{row['ref']}/{row['alt']}"
            lookup[k] = {'synvep_score': row.get('synvep_score', np.nan), 'has_gene': row.get('has_gene', np.nan), 'has_dbsnp': row.get('has_dbsnp', np.nan)}
    results = []
    matched = 0
    for v in variants:
        rec = lookup.get(v['key'], {})
        if rec:
            matched += 1
        results.append({'synvep_score': rec.get('synvep_score', np.nan), 'has_gene': rec.get('has_gene', np.nan), 'has_dbsnp': rec.get('has_dbsnp', np.nan)})
    print(f"  匹配: {matched}/{len(variants)}")
    return pd.DataFrame(results)

def extract_exon_pos(variants):
    print('\n[7] Exon_Relative_Pos (GTF)...')
    if not os.path.exists(GENCODE_GTF):
        print(f"  [跳过] {GENCODE_GTF} 不存在")
        return pd.DataFrame()
    exons = defaultdict(list)
    print('  读取 GTF 外显子...')
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
            exons[chrom].append((start, end))
    results = []
    for v in variants:
        chrom = str(v['chr']).replace('chr', '')
        pos = v['pos']
        rel = np.nan
        for start, end in exons.get(chrom, []):
            if start <= pos <= end:
                length = end - start
                rel = (pos - start) / max(1, length)
                break
        results.append({'Exon_Relative_Pos': rel})
    matched = sum((1 for r in results if not np.isnan(r['Exon_Relative_Pos'])))
    print(f"  匹配外显子: {matched}/{len(variants)}")
    return pd.DataFrame(results)

def extract_phys_features(variants, fasta):
    print('\n[8] 物理特征 (弯曲/堆积能)...')
    BEND = {'AA': -0.26, 'AC': 0.77, 'AG': 0.14, 'AT': -0.73, 'CA': 0.76, 'CC': -0.31, 'CG': 0.61, 'CT': 0.14, 'GA': 0.14, 'GC': 0.26, 'GG': -0.31, 'GT': 0.77, 'TA': 0.67, 'TC': 0.14, 'TG': 0.76, 'TT': -0.26}
    STACK = {'AA': -1.0, 'AT': -0.88, 'TA': -0.58, 'CA': -1.45, 'GT': -1.44, 'CT': -1.28, 'GA': -1.3, 'CG': -2.17, 'GC': -2.24, 'GG': -1.84, 'AC': -1.44, 'TC': -1.28, 'AG': -1.3, 'TG': -1.45, 'CC': -1.84, 'TT': -1.0}

    def calc_bend(seq):
        return sum((BEND.get(seq[i:i + 2], 0) for i in range(len(seq) - 1))) / max(1, len(seq) - 1)

    def calc_stack(seq):
        return sum((STACK.get(seq[i:i + 2], -1.0) for i in range(len(seq) - 1))) / max(1, len(seq) - 1)
    results = []
    for v in variants:
        feat = {}
        chrom, pos, ref, alt = (v['chr'], v['pos'], v['ref'].upper(), v['alt'].upper())
        seq = None
        for c in [chrom, f"chr{chrom}"]:
            if c in fasta.references:
                try:
                    seq = fasta.fetch(c, max(0, pos - 11), pos + 10).upper()
                except:
                    pass
                break
        if seq and len(seq) >= 4:
            mid = min(10, pos - max(0, pos - 11) - 1)
            seq_alt = seq[:mid] + alt + seq[mid + 1:]
            feat['Phys_Delta_Bend'] = calc_bend(seq_alt) - calc_bend(seq)
            feat['Phys_Delta_Stack'] = calc_stack(seq_alt) - calc_stack(seq)
        else:
            feat['Phys_Delta_Bend'] = np.nan
            feat['Phys_Delta_Stack'] = np.nan
        results.append(feat)
    print(f"  完成: {len(results)} 个")
    return pd.DataFrame(results)

def extract_silva_features(variants, fasta):
    print('\n[9] Silva 特征 (PhyloP / RNAfold / RSCU)...')
    import subprocess
    phylop_bw = '/home/cc/data/phylop100way.bw'
    bw_phylop = None
    try:
        import pyBigWig
        if os.path.exists(phylop_bw) and os.path.getsize(phylop_bw) > 0:
            bw_phylop = pyBigWig.open(phylop_bw)
            print('  phylop100way.bw 已加载')
    except Exception as e:
        print(f"  PhyloP 加载失败: {e}")
    RSCU_TABLE = {'TTT': 0.77, 'TTC': 1.23, 'TTA': 0.46, 'TTG': 0.81, 'CTT': 0.79, 'CTC': 1.26, 'CTA': 0.43, 'CTG': 2.37, 'ATT': 0.99, 'ATC': 1.39, 'ATA': 0.46, 'ATG': 1.0, 'GTT': 0.72, 'GTC': 0.95, 'GTA': 0.48, 'GTG': 2.76, 'TCT': 1.13, 'TCC': 1.18, 'TCA': 0.82, 'TCG': 0.33, 'CCT': 1.13, 'CCC': 1.21, 'CCA': 1.03, 'CCG': 0.42, 'ACT': 0.95, 'ACC': 1.35, 'ACA': 0.98, 'ACG': 0.4, 'GCT': 1.05, 'GCC': 1.54, 'GCA': 0.88, 'GCG': 0.43, 'TAT': 0.85, 'TAC': 1.15, 'CAT': 0.83, 'CAC': 1.17, 'CAA': 0.54, 'CAG': 2.5, 'AAT': 0.94, 'AAC': 1.06, 'AAA': 0.83, 'AAG': 2.37, 'GAT': 0.93, 'GAC': 1.07, 'GAA': 0.85, 'GAG': 2.36, 'TGT': 0.91, 'TGC': 1.09, 'TGG': 1.0, 'CGT': 0.31, 'CGC': 0.65, 'CGA': 0.45, 'CGG': 0.82, 'AGT': 0.79, 'AGC': 1.43, 'AGA': 0.86, 'AGG': 0.85, 'GGT': 0.65, 'GGC': 1.32, 'GGA': 1.02, 'GGG': 1.0}
    ESE_HEXAMERS = {'GAAGAA', 'AAGAAG', 'AGAAGA', 'CAAGAA', 'AAGAAA', 'GAAAGA', 'GAAGAT', 'TGAAGA', 'GAAGAC', 'GGAAGA', 'AAGACG', 'GAAGAG', 'AAGACA', 'AGAAGC', 'AAGAAC'}

    def run_rnafold(seq):
        try:
            proc = subprocess.run(['RNAfold', '--noPS'], input=seq, capture_output=True, text=True, timeout=10)
            lines = proc.stdout.strip().split('\n')
            if len(lines) >= 2:
                m = re.search('\\((-?\\d+\\.?\\d*)\\)', lines[1])
                if m:
                    return float(m.group(1))
        except:
            pass
        return np.nan

    def delta_rscu(seq_ref, seq_alt, mid):
        codons_ref = [seq_ref[k:k + 3] for k in range(0, min(len(seq_ref) - 2, 30), 3)]
        codons_alt = [seq_alt[k:k + 3] for k in range(0, min(len(seq_alt) - 2, 30), 3)]
        d = [abs(RSCU_TABLE.get(cr, 1.0) - RSCU_TABLE.get(ca, 1.0)) for cr, ca in zip(codons_ref, codons_alt)]
        return max(d) if d else 0.0
    results = []
    for i, v in enumerate(variants):
        feat = {}
        chrom, pos, ref, alt = (v['chr'], v['pos'], v['ref'].upper(), v['alt'].upper())
        feat['Silva_PhyloP'] = np.nan
        if bw_phylop:
            for c in [f"chr{chrom}", chrom]:
                try:
                    vals = bw_phylop.values(c, pos - 1, pos)
                    if vals and vals[0] is not None:
                        feat['Silva_PhyloP'] = vals[0]
                        break
                except:
                    pass
        seq = None
        for c in [chrom, f"chr{chrom}"]:
            if c in fasta.references:
                try:
                    seq = fasta.fetch(c, max(0, pos - 31), pos + 30).upper()
                except:
                    pass
                break
        if seq and len(seq) >= 10:
            mid = min(30, pos - max(0, pos - 31) - 1)
            seq_alt_str = seq[:mid] + alt + seq[mid + 1:]
            mfe_ref = run_rnafold(seq)
            mfe_alt = run_rnafold(seq_alt_str)
            feat['Silva_Delta_MFE'] = mfe_alt - mfe_ref if not np.isnan(mfe_ref) and (not np.isnan(mfe_alt)) else np.nan
            feat['Silva_Delta_RSCU'] = delta_rscu(seq, seq_alt_str, mid)
            ese_ref = sum((1 for k in range(len(seq) - 5) if seq[k:k + 6] in ESE_HEXAMERS))
            ese_alt = sum((1 for k in range(len(seq_alt_str) - 5) if seq_alt_str[k:k + 6] in ESE_HEXAMERS))
            feat['Splicing_ESE_Delta'] = ese_alt - ese_ref
        else:
            feat['Silva_Delta_MFE'] = np.nan
            feat['Silva_Delta_RSCU'] = np.nan
            feat['Splicing_ESE_Delta'] = np.nan
        results.append(feat)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(variants)}")
    if bw_phylop:
        bw_phylop.close()
    print(f"  完成: {len(results)} 个")
    return pd.DataFrame(results)

def extract_splice_tfbs(variants, fasta):
    print('\n[10] Dist_Nearest_Splice / TFBS_Max_Delta / Ctx_Poly_Change...')
    if not os.path.exists(GENCODE_GTF):
        print(f"  [跳过] GTF 不存在")
        return pd.DataFrame()
    splice_sites = defaultdict(list)
    print('  读取 GTF 剪接位点...')
    with open(GENCODE_GTF) as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.split('\t')
            if len(p) < 9 or p[2] not in ('exon', 'CDS'):
                continue
            chrom = p[0].replace('chr', '')
            start = int(p[3])
            end = int(p[4])
            splice_sites[chrom].append(start)
            splice_sites[chrom].append(end)
    TFBS_MOTIFS = {'SP1': 'GGGCGG', 'AP1': 'TGASTCA', 'NF1': 'TTGGCN', 'CREB': 'TGACGTCA', 'E2F': 'TTTSGCGS', 'ETS': 'MGGAWG', 'GATA': 'WGATAR', 'HIF': 'RCGTG'}

    def calc_tfbs_score(seq):
        score = 0
        for motif in TFBS_MOTIFS.values():
            m = motif.replace('S', '[GC]').replace('N', '[ACGT]').replace('W', '[AT]').replace('R', '[AG]').replace('M', '[AC]').replace('G', 'G')
            try:
                score += len(re.findall(m, seq, re.IGNORECASE))
            except:
                pass
        return score
    results = []
    for v in variants:
        feat = {}
        chrom = str(v['chr']).replace('chr', '')
        pos = v['pos']
        ref = v['ref'].upper()
        alt = v['alt'].upper()
        sites = splice_sites.get(chrom, [])
        if sites:
            dist = min((abs(pos - s) for s in sites))
            feat['Dist_Nearest_Splice'] = dist
        else:
            feat['Dist_Nearest_Splice'] = np.nan
        seq = None
        for c in [v['chr'], f"chr{v['chr']}"]:
            if c in fasta.references:
                try:
                    seq = fasta.fetch(c, max(0, pos - 11), pos + 10).upper()
                except:
                    pass
                break
        if seq and len(seq) >= 6:
            mid = min(10, pos - max(0, pos - 11) - 1)
            seq_alt = seq[:mid] + alt + seq[mid + 1:]
            feat['TFBS_Max_Delta'] = calc_tfbs_score(seq_alt) - calc_tfbs_score(seq)
            max_poly_ref = max((sum((1 for _ in g)) for _, g in __import__('itertools').groupby(seq)), default=0)
            max_poly_alt = max((sum((1 for _ in g)) for _, g in __import__('itertools').groupby(seq_alt)), default=0)
            feat['Ctx_Poly_Change'] = max_poly_alt - max_poly_ref
        else:
            feat['TFBS_Max_Delta'] = np.nan
            feat['Ctx_Poly_Change'] = np.nan
        results.append(feat)
    matched = sum((1 for r in results if not np.isnan(r.get('Dist_Nearest_Splice', np.nan))))
    print(f"  剪接位点匹配: {matched}/{len(variants)}")
    return pd.DataFrame(results)

def extract_aa_features(variants, fasta):
    print('\n[11] 氨基酸特征 (ref/alt/hydrophobic/polar/charged)...')
    CODON_AA = {'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L', 'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'TAT': 'Y', 'TAC': 'Y', 'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'TGT': 'C', 'TGC': 'C', 'TGG': 'W', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'}
    HYDRO = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
    POLAR = {'N', 'Q', 'S', 'T', 'Y', 'C', 'H'}
    CHARGED = {'R', 'K', 'D', 'E', 'H'}
    HYDRO_THRESH = 0.0
    results = []
    for v in variants:
        feat = {}
        chrom, pos, ref, alt = (v['chr'], v['pos'], v['ref'].upper(), v['alt'].upper())
        seq = None
        for c in [chrom, f"chr{chrom}"]:
            if c in fasta.references:
                try:
                    seq = fasta.fetch(c, max(0, pos - 4), pos + 3).upper()
                except:
                    pass
                break
        aa = None
        if seq and len(seq) >= 3:
            mid = min(3, pos - max(0, pos - 4) - 1)
            for offset in range(3):
                codon_start = mid - (mid - offset) % 3
                if codon_start >= 0 and codon_start + 3 <= len(seq):
                    codon_ref = seq[codon_start:codon_start + 3]
                    codon_alt = codon_ref[:mid - codon_start] + alt + codon_ref[mid - codon_start + 1:]
                    if len(codon_alt) == 3:
                        aa_ref = CODON_AA.get(codon_ref)
                        aa_alt = CODON_AA.get(codon_alt)
                        if aa_ref and aa_alt:
                            aa = (aa_ref, aa_alt)
                            break
        if aa:
            aa_ref, aa_alt = aa
            feat['ref_hydrophobic'] = int(HYDRO.get(aa_ref, 0) > HYDRO_THRESH)
            feat['alt_hydrophobic'] = int(HYDRO.get(aa_alt, 0) > HYDRO_THRESH)
            feat['hydrophobicity_change'] = int(feat['ref_hydrophobic'] != feat['alt_hydrophobic'])
            feat['ref_polar'] = int(aa_ref in POLAR)
            feat['alt_polar'] = int(aa_alt in POLAR)
            feat['polarity_change'] = int(feat['ref_polar'] != feat['alt_polar'])
            feat['ref_charged'] = int(aa_ref in CHARGED)
        else:
            for k in ['ref_hydrophobic', 'alt_hydrophobic', 'hydrophobicity_change', 'ref_polar', 'alt_polar', 'polarity_change', 'ref_charged']:
                feat[k] = np.nan
        results.append(feat)
    filled = sum((1 for r in results if not np.isnan(r.get('ref_hydrophobic', np.nan))))
    print(f"  氨基酸特征填充: {filled}/{len(variants)}")
    return pd.DataFrame(results)

def main():
    print('=' * 60)
    print('Case Study 特征提取 (真实数据)')
    print('=' * 60)
    print(f"\n加载 hg19: {REF_FASTA}")
    if not os.path.exists(REF_FASTA):
        print('  [错误] hg19.fa 不存在！')
        sys.exit(1)
    fasta = pysam.FastaFile(REF_FASTA)
    print(f"  染色体数: {len(fasta.references)}")
    variants = load_variants()
    keys = [v['key'] for v in variants]
    df_dna = extract_dna_features(variants, fasta)
    df_trans = extract_translation_features(variants, fasta)
    df_rbp = extract_rbp_rnamod_features(variants, fasta)
    df_fit = extract_fitcons_features(variants)
    df_remm = extract_remm_features(variants)
    df_svep = extract_synvep_features(variants)
    df_exon = extract_exon_pos(variants)
    df_phys = extract_phys_features(variants, fasta)
    df_silva = extract_silva_features(variants, fasta)
    df_splice = extract_splice_tfbs(variants, fasta)
    df_aa = extract_aa_features(variants, fasta)
    fasta.close()
    result = pd.DataFrame({'Variant19': keys})
    for df in [df_dna, df_trans, df_rbp, df_fit, df_remm, df_svep, df_exon, df_phys, df_silva, df_splice, df_aa]:
        if df is not None and len(df) == len(variants):
            for col in df.columns:
                if col not in result.columns:
                    result[col] = df[col].values
                else:
                    result[col] = result[col].combine_first(pd.Series(df[col].values, index=result.index))
    result.to_csv(OUTPUT_CSV, index=False)
    n_filled = sum((1 for c in result.columns if c != 'Variant19' and result[c].notna().any()))
    print(f"\n输出: {OUTPUT_CSV}")
    print(f"维度: {result.shape}")
    print(f"有数据的列: {n_filled}/{len(result.columns) - 1}")
if __name__ == '__main__':
    main()
