#!/usr/bin/env python3
import argparse
import os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=('case', 'control'), default='case')
parser.add_argument('--vcf')
parser.add_argument('--input')
parser.add_argument('--output')
parser.add_argument('--gtf', default='/home/cc/data/gencode.v19.annotation.gtf')
parser.add_argument('--fasta', default='/home/cc/data/hg19.fa')
args = parser.parse_args()
VCF_IN = args.vcf or f"/home/cc/data/{args.dataset}_study_sorted.vcf.gz"
GTF_FILE = args.gtf
FASTA = args.fasta
MERGED = args.input or f"/mnt/c/Users/GCC/Desktop/case study/data/{args.dataset}_variants.csv"
OUT_CSV = args.output or f"/mnt/c/Users/GCC/Desktop/case study/feature/{args.dataset}_vcf_extracted/{args.dataset}_mmsplice_features.csv"
MMS_MAP = {'delta_logit_psi': 'MMS_delta_logit_psi', 'ref_acceptorIntron': 'MMS_ref_acceptorIntron', 'alt_exon': 'MMS_alt_exon', 'pathogenicity': 'MMS_pathogenicity'}

def build_variant19(row):
    c = str(row.get('chr', row.get('CHROM', ''))).replace('chr', '')
    return f"{c}_{row['pos']}_{row['ref']}/{row['alt']}"

def main():
    print('=' * 60)
    print('MMSplice 特征提取')
    print('=' * 60)
    df_merged = pd.read_csv(MERGED, usecols=['chr', 'pos', 'ref', 'alt'])
    df_merged['Variant19'] = df_merged.apply(lambda r: f"{str(r['chr']).replace('chr', '')}_{int(r['pos'])}_{r['ref']}/{r['alt']}", axis=1)
    print(f"目标变异数: {len(df_merged)}")
    from mmsplice import MMSplice
    from mmsplice.vcf_dataloader import SplicingVCFDataloader
    print(f"加载 GTF: {GTF_FILE}")
    print(f"加载 FASTA: {FASTA}")
    try:
        dl = SplicingVCFDataloader(GTF_FILE, FASTA, VCF_IN, split_seq=True, encode=True)
        print('SplicingVCFDataloader 初始化成功')
    except Exception as e:
        print(f"SplicingVCFDataloader 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    model = MMSplice()
    print('运行 MMSplice predict_all_table ...')
    from mmsplice import predict_all_table
    import re as _re
    df_pred = predict_all_table(model, dl, pathogenicity=True, splicing_efficiency=False)
    print(f"predict_all_table 结果: {df_pred.shape}, 列: {list(df_pred.columns)}")

    def id_to_v19(id_str):
        m = _re.match('(chr)?(\\S+):(\\d+):([ACGT]+)>([ACGT]+)', str(id_str))
        if m:
            return f"{m.group(2)}_{m.group(3)}_{m.group(4)}/{m.group(5)}"
        return ''
    df_pred['Variant19'] = df_pred['ID'].apply(id_to_v19)
    mms_cols = list(MMS_MAP.values())
    rename = {src: dst for src, dst in MMS_MAP.items() if src in df_pred.columns}
    df_pred = df_pred.rename(columns=rename)
    available_mms = [c for c in mms_cols if c in df_pred.columns]
    primary = 'MMS_delta_logit_psi'
    if primary in df_pred.columns:
        df_pred['_abs_psi'] = df_pred[primary].abs()
        df_pred = df_pred.sort_values('_abs_psi', ascending=False)
        df_out = df_pred.groupby('Variant19')[available_mms].first().reset_index()
    else:
        df_out = df_pred.groupby('Variant19')[available_mms].mean().reset_index()
    df_result = df_merged[['Variant19']].merge(df_out, on='Variant19', how='left')
    df_result.to_csv(OUT_CSV, index=False)
    print(f"\n完成! 输出: {OUT_CSV}")
    print(f"维度: {df_result.shape}")
    for c in mms_cols:
        if c in df_result.columns:
            print(f"  {c}: {df_result[c].notna().sum()}/{len(df_result)} 非空")
if __name__ == '__main__':
    main()
