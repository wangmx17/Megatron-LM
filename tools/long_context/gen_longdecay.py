"""
gen_longdecay.py  --  MiniCPM-5 longdecay 数据配比生成

Fork 自 /user/xuxiaoyue/datafactory/recipe_final/gen_minicpm5_32k.py，
128k 逻辑参考 /user/xuxiaoyue/datafactory/recipe_final/gen_law_128k.py。
主要改动：
  - 合并 32k / 128k 两种模式（通过 --mode 参数切换）
  - other 子类保持原始分桶权重比例（不上采样长文本）
  - CLASS_RECIPE 按 260301 longdecay 比例 + Other=10% + Gov 调整
  - MAX_TRAIN_TOKENS 按实际训练量设置（32k: 75B, 128k: 25B）

模式配置：
  32k:  阈值 16k, 短/长 = 3/8 : 5/8, 训练 75B token
  128k: 阈值 64k, 短/长 = 1/2 : 1/2, 训练 25B token

--------
  Step 1  加载 minicpm5.txt，根据 SUBCLASS_DATASETS 对每个数据集分类
  Step 2  (ADD=True) 对 detective/long/mixed_fin/mixed_gov 中缺失的数据集，
          从 sala.txt 补充路径；(ADD=False) 不补充，仅整个子类全部缺失时权重置 0
  Step 3  根据 CLASS_RECIPE / SUBCLASS_RECIPE / SUBCLASS_PORTION 目标配比，
          重新分配每个数据集的权重
  Step 4  调整每个数据集内部分桶的权重：
            - 去掉条数 <=1000 的分桶（设为 0）
            - 非 other 子类：按 SPLIT_THRESHOLD 分为短/长两组重新分配
            - other 子类：保持原始分桶权重比例（不上采样长文本）
  Step 5  对训练 N token 时可能超过 3epoch 的分桶进行裁剪
  Step 6  归一化所有权重使和为 1，按原始顺序写入输出文件

--------
"""

import os
import sys
import json
import argparse
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 是否补充 minicpm5.txt 中缺失的数据集（从 sala.txt 获取路径）
# 设为 False 时不添加新数据集，仅当某子类的数据集全部缺失时将该子类权重置为 0
ADD = True

# ---------------------------------------------------------------------------
# 相关设定
# ---------------------------------------------------------------------------

# 所有合法的长度分桶名称
LENGTH_BUCKETS = {
    '0k_4k', '4k_8k', '8k_16k', '16k_32k', '32k_64k',
    '64k_128k', '128k_256k', '256k_512k', '512k_gt', 'qa',
    'lt_4k', '4k_16k',
}

# 每个分桶的上界（token），用于根据 SPLIT_THRESHOLD 自动划分 短/长 两组
BUCKET_UPPER_BOUND = {
    '0k_4k': 4096,      'lt_4k': 4096,      '4k_8k': 8192,
    '4k_16k': 16384,    '8k_16k': 16384,    '16k_32k': 32768,
    '32k_64k': 65536,   '64k_128k': 131072, '128k_256k': 262144,
    '256k_512k': 524288, '512k_gt': 1048576, 'qa': 4096,
}

# ---- 32k / 128k 模式配置 ----
MODE_CONFIG = {
    '32k': {
        'split_threshold': 16384,           # 16k
        'weight_short': 0.375,              # 3/8
        'weight_long': 0.625,               # 5/8
        'max_train_tokens': 75_000_000_000, # 75B
        'output_suffix': '32k_longdecay',
    },
    '128k': {
        'split_threshold': 65536,           # 64k
        'weight_short': 0.5,               # 1/2
        'weight_long': 0.5,                # 1/2
        'max_train_tokens': 25_000_000_000, # 25B
        'output_suffix': '128k_longdecay',
    },
}

def _split_buckets(threshold: int) -> tuple[set[str], set[str]]:
    short = {b for b, ub in BUCKET_UPPER_BOUND.items() if ub <= threshold}
    long  = {b for b, ub in BUCKET_UPPER_BOUND.items() if ub >  threshold}
    return short, long

# 各分桶的中位数 token 数，用于估算总 token（中位数长度 × 数据条数）
BUCKET_MEDIAN_TOKENS = {
    '0k_4k': 2048,   'lt_4k': 2048,   '4k_8k': 6144,
    '4k_16k': 10240, '8k_16k': 12288, '16k_32k': 24576,
    '32k_64k': 49152, '64k_128k': 98304, '128k_256k': 196608,
    '256k_512k': 393216, '512k_gt': 524288, 'qa': 2048,
}

MAX_EPOCHS = 3                      # 单个分桶最多训练的 epoch 数
MIN_ENTRY_COUNT = 1000              # 分桶最少需要的数据条数

# 需要检查缺失的子类（ADD 时从 sala.txt 补充）
ADD_MISSING_SUBCLASSES = {'detective', 'long', 'mixed_fin', 'mixed_gov'}

# ---- 大类目标配比（和约等于 1） ----
# Other=10%，Gov 从 0.1other 取值，其余按 260301 比例等比缩放
CLASS_RECIPE = {
    'PDF': 0.176573,    'Paper': 0.171262,  'Knowledge': 0.064655,
    'Law': 0.087623,    'Gov': 0.092691,    'Finance': 0.016781,
    'Long': 0.001486,   'Novel': 0.0774,    'Detective': 0.002896,
    'Other': 0.1,       'Code': 0.211527,
}

# ---- 大类内部子类的配比（比值） ----
SUBCLASS_RECIPE = {
    'PDF':       {'zh_pdf': 1, 'en_pdf': 2.5},
    'Paper':     {'zh_paper': 1, 'en_paper': 2.5},
    'Finance':   {'zh_fin': 1, 'mixed_fin': 4},
    'Novel':     {'zh_novel': 1.0},
    'Knowledge': {'en_knowledge': 1.0},
    'Law':       {'zh_law': 1.0},
    'Gov':       {'mixed_gov': 1.0},
    'Detective': {'detective': 1.0},
    'Long':      {'long': 1.0},
    'Other':     {'other': 1.0},
    'Code':      {'code': 1.0},
}

# ---- 每个子类包含哪些数据集 ----
SUBCLASS_DATASETS = {
    'code': [
        'Config_DSL_DevOps', 'Config_DSL_DevOps_fim_v1', 'Config_DSL_DevOps_fim_v1_supp',
        'Functional_Formal', 'Functional_Formal_fim_v1', 'Functional_Formal_fim_v1_supp',
        'Hardware_Lowlevel', 'Hardware_Lowlevel_fim_v1', 'Hardware_Lowlevel_fim_v1_supp',
        'MultiPL_E', 'MultiPL_E_fim_v1',
        'Other_Programming', 'Other_Programming_fim_v1', 'Other_Programming_fim_v1_supp',
        'Text_Documentation', 'Web_Frontend', 'Web_Frontend_fim_v1',
        'RefineCode-code-corpus', 'stack_v2_full_not_code_transformed',
        'CCI4.0_cot_synthesis_code', 'CCI4.0_code', 'code_30w',
        'code_python_to_testbook_250427', 'code_to_strucutre_textbook_250407',
        'nvidia_Nemotron-Pretraining-SFT_v1_Nemotron-SFT-Code',
        'nvidia_Nemotron_Pretraining_Code_v1_Synthetic-Code',
        'opc_annealing_algorithmic_corpus', 'web_code_en_dedup_new', 'web_code_zh_dedup_new',
        'Nemotron-Pretraining-Scientific-Coding',
        'Nemotron-Pretraining-Code-v2_synthetic-code-review',
        'Nemotron-Pretraining-Code-v2_synthetic-question-answering',
        'Nemotron-Pretraining-Code-v2_ynthetic-student-teacher',
        'Nemotron-Pretraining-Code-v2_synthetic-transpilation',
        'Nemotron-Pretraining-Code-v2_synthetic-rewriting',
        'code_0', 'code_feedback_241230', 'evol_code_clean_dedup_whoru',
        'functioncalling_chatml', 'gcode_0813', 'glaive',
        'humanevallike_clean_dedup', 'lccf_apps_ultracode_aug_data_0818',
        'leetcode', 'leetcode_pass_code', 'leetcode_pass_code_0125',
        'opc_annealing_synthetic_code_snippet_corpus', 'opc_annealing_synthetic_qa_corpus',
        'parallel_functioncall', 'task_oriented', 'tinycode_sciphi', 'xlam_function_calling_60k',
        'humaneval_like_qa_250423_dedup_with_unit_test_pass_0shot_2x',
        'mbpp_like_qa_250423_with_unit_test_pass_0shot_2x',
        'mbpp_like_qa_250423_with_unit_test_pass_fewshot_2x',
        'humaneval_like_old_data_ut_pass_0shot_2x',
        'deep_think_sft_drop_think', 'nemotron_sft_code',
        'choices', 'codeinterpreter_0720', 'dscodefb_0720',
    ],
    'detective': [
        'detective_novels_en_longcontent_yangjiawei',
        'detective_novels_zh_longcontent_yangjiawei',
    ],
    'long': ['dailyscript', 'congress', 'gaogov', 'imsdb'],
    'mixed_fin': [
        'finance_en_longcontent_yangjiawei',
        'finance_zh_longcontent_yangjiawei',
    ],
    'mixed_gov': [
        'gov_en_longcontent_yangjiawei',
        'gov_zh_longcontent_yangjiawei',
    ],
    'zh_fin': ['Chinese_Finance_merge'],
    'zh_law': ['Chinese_Law_merge', 'mnbvc_law'],
    'zh_novel': ['Chinese_Novel_merge'],
    'zh_pdf': [
        'Chinese_Book_large_merge', 'Chinese_Book_small_merge', 'annas_book_zh',
        'CCI4.0-M2-Extra-v1-new_books_zh', 'chinese_patent_volume',
        'finepdfs_edu_zh', 'finepdfs_zh', 'patent_zh', 'hangyeyanbao_zh',
    ],
    'en_pdf': [
        'English_Book_merge', 'annas_book_en', 'annas_en_pdf_2500w',
        'CCI4.0-M2-Extra-v1-new_books_en', 'finepdfs_edu_en', 'finepdfs_en',
        'proquest', 'patent_en', 'hangyeyanbao_en',
    ],
    'zh_paper': ['hi138_data', 'zhiwangshuobo_pipeline', 'zhiwangshuobo_vlm', 'ChineseJournal'],
    'en_paper': [
        'English_Academic_merge', 'google-scholar_merge', 'annas_qikan', 'arxiv_ocr',
        'CCI4.0-M2-Extra-v1-new_arxiv', 'CCI4.0_cot_synthesis_arxiv',
        'arxiv_tex', 'annas_qikan_1000w',
    ],
    'en_knowledge': [
        'fandom_by_purchasing', 'finewiki_en', 'olympics_wiki_corpus',
        'wikivoyage_english', 'wikibooks_new_english',
        'Nemotron-Pretraining-Wiki-Rewrite', 'CCI4.0_cot_synthesis_wiki',
        'finewiki_en_synthetic_conversation', 'finewiki_en_synthetic_fact_qa',
        'finewiki_en_synthetic_textbook', 'chegg-text',
    ],
    # other 不在这里列出，所有不属于上述子类的数据集自动归入 other
}

# ---- 各子类内每个数据集的占比 ----
# 只列出 ADD_MISSING_SUBCLASSES 中的缺失数据集
# 已存在于 minicpm5.txt 的数据集会用当前权重来计算占比，不需要此表
MISSING_DATASET_PORTION = {
    # detective: 两个数据集都缺失
    'detective_novels_en_longcontent_yangjiawei': 0.6,
    'detective_novels_zh_longcontent_yangjiawei': 0.4,
    # long: dailyscript 和 imsdb 缺失，congress/gaogov 已存在
    'dailyscript': 0.05,
    'imsdb':       0.10,
    # mixed_fin: 两个数据集都缺失
    'finance_en_longcontent_yangjiawei': 0.6,
    'finance_zh_longcontent_yangjiawei': 0.4,
    # mixed_gov: 两个数据集都缺失
    'gov_en_longcontent_yangjiawei': 0.8,
    'gov_zh_longcontent_yangjiawei': 0.2,
}

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def extract_dataset_and_bucket(path: str) -> tuple[str, str]:
    """从路径中提取 (数据集名, 分桶名)。

    路径最后一级是分桶（如 0k_4k, qa），数据集名是 _merge/_sample/ 等关键词
    后面的第一级目录名。

    特殊情况：merge_small 是一个父目录，其下 260113.stable1 / 260113.stable2 /
    260126.decay 才是真正的数据集，需要再取一级。
    """
    parts = path.rstrip('/').split('/')
    bucket = parts[-1]
    if bucket not in LENGTH_BUCKETS:
        return parts[-1], 'unknown'

    for marker in ('_sample/', '_merge/'):
        idx = path.find(marker)
        if idx != -1:
            after = path[idx + len(marker):]
            segments = after.split('/')
            dataset_name = segments[0]
            if dataset_name == 'merge_small' and len(segments) >= 2:
                dataset_name = segments[1]
            return dataset_name, bucket

    return parts[-2] if len(parts) >= 2 else parts[0], bucket


def build_class_mappings() -> tuple[dict[str, str], dict[str, str]]:
    """构建两个映射：

    dataset_to_subclass : 数据集名 -> 子类名
    subclass_to_class   : 子类名   -> 大类名
    """
    dataset_to_subclass: dict[str, str] = {}
    subclass_to_class: dict[str, str] = {}

    for cls, subclasses in SUBCLASS_RECIPE.items():
        for subclass in subclasses:
            subclass_to_class[subclass] = cls
            for ds in SUBCLASS_DATASETS.get(subclass, []):
                dataset_to_subclass[ds] = subclass

    return dataset_to_subclass, subclass_to_class


def load_entry_counts(path: str) -> dict[str, dict[str, int]]:
    """加载 minicpm5_entry_count.json -> {数据集: {分桶: 条数}}。"""
    counts: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ds_name, _total, bucket_dict = rec[0], rec[1], rec[2]
            counts[ds_name] = bucket_dict
    return counts


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MiniCPM-5 longdecay 配比生成')
    parser.add_argument('--mode', choices=['32k', '128k'], default='32k',
                        help='生成模式: 32k (阈值16k, 3/8:5/8, 75B) 或 128k (阈值64k, 1/2:1/2, 25B)')
    args = parser.parse_args()

    cfg = MODE_CONFIG[args.mode]
    SPLIT_THRESHOLD = cfg['split_threshold']
    WEIGHT_SHORT    = cfg['weight_short']
    WEIGHT_LONG     = cfg['weight_long']
    MAX_TRAIN_TOKENS = cfg['max_train_tokens']
    BUCKETS_SHORT, BUCKETS_LONG = _split_buckets(SPLIT_THRESHOLD)

    print(f'模式: {args.mode}  阈值: {SPLIT_THRESHOLD//1024}k  短/长: {WEIGHT_SHORT:.0%}/{WEIGHT_LONG:.0%}  训练: {MAX_TRAIN_TOKENS/1e9:.0f}B')

    recipe_path = os.path.join(SCRIPT_DIR, 'minicpm5.txt')
    sala_path   = os.path.join(SCRIPT_DIR, 'sala.txt')
    entry_count_path = os.path.join(SCRIPT_DIR, 'minicpm5_entry_count.json')
    output_path = os.path.join(SCRIPT_DIR, f'minicpm5_{cfg["output_suffix"]}.txt')

    # ===================================================================
    # STEP 1 : 加载 minicpm5.txt 并分类
    # ===================================================================
    print('=' * 72)
    print('STEP 1: 加载配比文件并对数据集分类')
    print('=' * 72)

    dataset_to_subclass, subclass_to_class = build_class_mappings()

    # 每条记录: (weight, path, dataset_name, bucket)
    entries: list[tuple[float, str, str, str]] = []
    # 记录每条原始权重，用于 Step 4 判断哪些分桶需要冻结为 0
    original_weights: dict[int, float] = {}
    datasets_seen: set[str] = set()

    with open(recipe_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            weight = float(parts[0])
            path = parts[1]
            ds_name, bucket = extract_dataset_and_bucket(path)
            original_weights[len(entries)] = weight
            entries.append((weight, path, ds_name, bucket))
            datasets_seen.add(ds_name)

    # 统计每个数据集当前的总权重
    dataset_weight: dict[str, float] = defaultdict(float)
    for w, _p, ds, _b in entries:
        dataset_weight[ds] += w

    # 给每个数据集分配子类（不在任何类中的归入 other）
    ds_subclass_map: dict[str, str] = {}
    for ds in datasets_seen:
        ds_subclass_map[ds] = dataset_to_subclass.get(ds, 'other')

    # 按大类/子类汇总当前权重
    class_weight: dict[str, float] = defaultdict(float)
    subclass_weight_current: dict[str, float] = defaultdict(float)
    for ds, w in dataset_weight.items():
        sc = ds_subclass_map[ds]
        cls = subclass_to_class.get(sc, 'Other')
        class_weight[cls] += w
        subclass_weight_current[sc] += w

    total_w = sum(dataset_weight.values())
    print(f'\n共加载 {len(entries)} 行, {len(datasets_seen)} 个数据集, 总权重 = {total_w:.6f}')

    # >>> 监控输出：每个大类/子类下有多少数据集 <<<
    print('\n各大类当前权重及数据集数量:')
    subclass_ds_count: dict[str, int] = defaultdict(int)
    for ds in datasets_seen:
        sc = ds_subclass_map[ds]
        subclass_ds_count[sc] += 1

    print(f'  {"大类":15s}  {"子类":20s}  {"数据集数":>8s}  {"当前权重":>12s}')
    print(f'  {"-"*15}  {"-"*20}  {"-"*8}  {"-"*12}')
    for cls in sorted(CLASS_RECIPE.keys()):
        cls_ds_total = 0
        for sc in sorted(SUBCLASS_RECIPE.get(cls, {}).keys()):
            cnt = subclass_ds_count.get(sc, 0)
            sw = subclass_weight_current.get(sc, 0.0)
            print(f'  {cls:15s}  {sc:20s}  {cnt:8d}  {sw:12.6f}')
            cls_ds_total += cnt
        cw = class_weight.get(cls, 0.0)
        print(f'  {cls:15s}  {"[SUM]":20s}  {cls_ds_total:8d}  {cw:12.6f}  ({cw/total_w*100:.2f}%)')

    # ===================================================================
    # STEP 2 : 补充缺失数据集（仅 ADD=True 时执行）
    # ===================================================================
    print('\n' + '=' * 72)
    print(f'STEP 2: 补充缺失数据集 (ADD={ADD})')
    print('=' * 72)

    # 找出每个待检查子类中哪些数据集缺失
    missing_datasets_by_subclass: dict[str, list[str]] = defaultdict(list)
    for subclass in ADD_MISSING_SUBCLASSES:
        for ds in SUBCLASS_DATASETS.get(subclass, []):
            if ds not in datasets_seen:
                missing_datasets_by_subclass[subclass].append(ds)

    # 判断哪些子类完全缺失（子类内所有数据集都不在 minicpm5.txt 中）
    fully_missing_subclasses: set[str] = set()
    for subclass in ADD_MISSING_SUBCLASSES:
        all_ds = SUBCLASS_DATASETS.get(subclass, [])
        if all_ds and all(ds not in datasets_seen for ds in all_ds):
            fully_missing_subclasses.add(subclass)

    for sc, ds_list in sorted(missing_datasets_by_subclass.items()):
        tag = '(整个子类缺失)' if sc in fully_missing_subclasses else '(部分缺失)'
        print(f'  子类 {sc} {tag}: 缺失 {ds_list}')

    added_count = 0
    if ADD:
        # 预加载 sala.txt，按数据集名索引
        sala_lines: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
        with open(sala_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                sw = float(parts[0])
                sp = parts[1]
                sds, sbkt = extract_dataset_and_bucket(sp)
                sala_lines[sds].append((sw, sp, sbkt))

        for subclass in sorted(ADD_MISSING_SUBCLASSES):
            for ds in missing_datasets_by_subclass.get(subclass, []):
                if ds not in sala_lines:
                    print(f'  [警告] {ds} 在 sala.txt 中未找到，跳过')
                    continue
                for _sw, sp, sbkt in sala_lines[ds]:
                    entries.append((0.0, sp, ds, sbkt))
                    added_count += 1
                datasets_seen.add(ds)
                ds_subclass_map[ds] = subclass
                dataset_weight[ds] = 0.0
                print(f'  添加 {ds} ({len(sala_lines[ds])} 个分桶) -> 子类={subclass}')

        print(f'\n共补充 {added_count} 行')
    else:
        print(f'\n  ADD=False，不补充新数据集')
        if fully_missing_subclasses:
            print(f'  以下子类完全缺失，其目标权重将被置为 0: {sorted(fully_missing_subclasses)}')

    # ===================================================================
    # STEP 3 : 按目标配比重新分配权重
    # ===================================================================
    print('\n' + '=' * 72)
    print('STEP 3: 按 CLASS_RECIPE / SUBCLASS_RECIPE 重新分配权重')
    print('=' * 72)

    # 计算每个子类的目标权重
    # ADD=False 时，仅当子类完全缺失才将其权重置 0
    target_subclass_weight: dict[str, float] = {}
    zeroed_subclasses: list[str] = []
    for cls, subclasses in SUBCLASS_RECIPE.items():
        cls_target = CLASS_RECIPE[cls]
        ratio_sum = sum(subclasses.values())
        for sc, ratio in subclasses.items():
            if not ADD and sc in fully_missing_subclasses:
                target_subclass_weight[sc] = 0.0
                zeroed_subclasses.append(sc)
            else:
                target_subclass_weight[sc] = cls_target * ratio / ratio_sum

    if zeroed_subclasses:
        print(f'\n  以下子类目标权重已置为 0: {zeroed_subclasses}')

    # 按子类分组数据集，按当前权重比例（或 MISSING_DATASET_PORTION）分配目标权重
    target_dataset_weight: dict[str, float] = {}

    subclass_datasets_grouped: dict[str, list[str]] = defaultdict(list)
    for ds in datasets_seen:
        sc = ds_subclass_map[ds]
        subclass_datasets_grouped[sc].append(ds)

    for sc, ds_list in subclass_datasets_grouped.items():
        sc_target = target_subclass_weight.get(sc, 0.0)
        if sc == 'other':
            sc_target = CLASS_RECIPE['Other']

        missing_in_sc = set(missing_datasets_by_subclass.get(sc, []))
        new_ds = [ds for ds in ds_list if ds in missing_in_sc]
        existing_ds = [ds for ds in ds_list if ds not in missing_in_sc]

        # 新增数据集按 MISSING_DATASET_PORTION 写定的比例（子类内绝对占比）
        new_share = sum(MISSING_DATASET_PORTION.get(ds, 0.0) for ds in new_ds)
        for ds in new_ds:
            target_dataset_weight[ds] = sc_target * MISSING_DATASET_PORTION.get(ds, 0.0)

        # 已有数据集按当前 weight 比例瓜分剩余份额
        remaining_share = 1.0 - new_share
        # print(f'remaining_share: {sc} {remaining_share}')
        current_total = sum(dataset_weight[ds] for ds in existing_ds)
        for ds in existing_ds:
            if current_total > 0:
                target_dataset_weight[ds] = sc_target * remaining_share * (dataset_weight[ds] / current_total)
            elif existing_ds:
                target_dataset_weight[ds] = sc_target * remaining_share / len(existing_ds)
            else:
                target_dataset_weight[ds] = 0.0

    # 输出目标 vs 当前的大类对比
    target_class_weight: dict[str, float] = defaultdict(float)
    for ds, tw in target_dataset_weight.items():
        sc = ds_subclass_map[ds]
        cls = subclass_to_class.get(sc, 'Other')
        target_class_weight[cls] += tw

    print('\n目标配比 vs 当前配比:')
    print(f'  {"大类":15s}  {"目标":>10s}  {"当前":>10s}')
    for cls in sorted(CLASS_RECIPE.keys()):
        print(f'  {cls:15s}  {target_class_weight.get(cls,0):.6f}  {class_weight.get(cls,0):.6f}')

    # ===================================================================
    # STEP 4 : 调整数据集内部分桶权重（16k-/16k+ 拆分）
    # ===================================================================
    print('\n' + '=' * 72)
    threshold_label = f'{SPLIT_THRESHOLD // 1024}k'
    print(f'STEP 4: 调整分桶权重（过滤条数不足、{threshold_label} 短/长拆分 {WEIGHT_SHORT:.0%}/{WEIGHT_LONG:.0%}，other 保持原始比例）')
    print('=' * 72)

    entry_counts = load_entry_counts(entry_count_path)

    # 按数据集分组，记录每个数据集对应的 entries 索引
    ds_bucket_indices: dict[str, list[int]] = defaultdict(list)
    for i, (w, p, ds, bkt) in enumerate(entries):
        ds_bucket_indices[ds].append(i)

    dropped_low_entry = 0
    dropped_details: list[tuple[str, str, int]] = []

    for ds, indices in ds_bucket_indices.items():
        ds_target = target_dataset_weight.get(ds, 0.0)
        ds_sc = ds_subclass_map.get(ds, 'other')

        # 过滤条数不足的分桶
        ds_has_counts = ds in entry_counts
        active_indices = []
        for i in indices:
            _, p, _, bkt = entries[i]
            if ds_has_counts:
                ec = entry_counts[ds].get(bkt, 0)
                if ec <= MIN_ENTRY_COUNT:
                    entries[i] = (0.0, p, ds, bkt)
                    dropped_low_entry += 1
                    dropped_details.append((ds, bkt, ec))
                    continue
            else:
                print(f'  [警告] {ds} 在 entry_count 中未找到，跳过')
                continue
            active_indices.append(i)

        if not active_indices:
            continue

        if ds_sc == 'other':
            # other 子类：保持原始分桶权重比例（不上采样长文本）
            orig_total = sum(original_weights.get(i, 0.0) for i in active_indices)
            for i in active_indices:
                _, p, _, bkt = entries[i]
                if orig_total > 0:
                    ratio = original_weights.get(i, 0.0) / orig_total
                else:
                    ratio = 1.0 / len(active_indices)
                entries[i] = (ds_target * ratio, p, ds, bkt)
        else:
            # 非 other 子类：按短/长拆分重新分配
            short_indices = [i for i in active_indices if entries[i][3] in BUCKETS_SHORT]
            long_indices  = [i for i in active_indices if entries[i][3] in BUCKETS_LONG]

            if short_indices and long_indices:
                w_short = WEIGHT_SHORT
                w_long  = WEIGHT_LONG
            elif short_indices:
                w_short = 1.0
                w_long  = 0.0
            else:
                w_short = 0.0
                w_long  = 1.0

            if short_indices:
                per_bucket = ds_target * w_short / len(short_indices)
                for i in short_indices:
                    _, p, _, bkt = entries[i]
                    entries[i] = (per_bucket, p, ds, bkt)
            if long_indices:
                per_bucket = ds_target * w_long / len(long_indices)
                for i in long_indices:
                    _, p, _, bkt = entries[i]
                    entries[i] = (per_bucket, p, ds, bkt)

    # >>> 监控输出：条数不足被丢弃的分桶 <<<
    print(f'\n因条数 <= {MIN_ENTRY_COUNT} 被丢弃的分桶（共 {dropped_low_entry} 个）:')
    for ds, bkt, ec in dropped_details:
        print(f'  {ds:50s}  桶={bkt:12s}  条数={ec}')

    # ===================================================================
    # STEP 5 : 3-epoch 上限裁剪（20B token）
    # ===================================================================
    print('\n' + '=' * 72)
    print(f'STEP 5: 训练 {MAX_TRAIN_TOKENS/1e9:.0f}B token 下的 {MAX_EPOCHS}-epoch 上限裁剪')
    print('=' * 72)

    clipped_count = 0
    clipped_details: list[tuple[str, str, float, float]] = []
    for i, (w, p, ds, bkt) in enumerate(entries):
        if w <= 0.0:
            continue
        ec = entry_counts.get(ds, {}).get(bkt, 0)
        if ec == 0:
            continue
        median_tokens = BUCKET_MEDIAN_TOKENS.get(bkt, 2048)
        estimated_tokens = median_tokens * ec
        max_weight = MAX_EPOCHS * estimated_tokens / MAX_TRAIN_TOKENS
        if w > max_weight:
            clipped_details.append((ds, bkt, w, max_weight))
            entries[i] = (max_weight, p, ds, bkt)
            clipped_count += 1

    print(f'\n被裁剪的分桶（共 {clipped_count} 个）:')
    for ds, bkt, old_w, new_w in clipped_details:
        print(f'  {ds:50s}  桶={bkt:12s}  {old_w:.9f} -> {new_w:.9f}')

    # ===================================================================
    # STEP 6 : 归一化并保存
    # ===================================================================
    print('\n' + '=' * 72)
    print('STEP 6: 归一化并保存输出')
    print('=' * 72)

    total = sum(w for w, _, _, _ in entries)
    if total <= 0:
        print('[错误] 总权重为 0，无法保存。')
        return

    # 写入输出文件，同时统计最终的大类、子类、分桶、数据集权重
    final_class_weight: dict[str, float] = defaultdict(float)
    final_subclass_weight: dict[str, float] = defaultdict(float)
    final_bucket_weight: dict[str, float] = defaultdict(float)
    final_dataset_weight: dict[str, float] = defaultdict(float)

    with open(output_path, 'w') as f:
        for w, p, ds, bkt in entries:
            scaled = w / total
            sc = ds_subclass_map.get(ds, 'other')
            cls = subclass_to_class.get(sc, 'Other')
            final_class_weight[cls] += scaled
            final_subclass_weight[sc] += scaled
            final_bucket_weight[bkt] += scaled
            final_dataset_weight[ds] += scaled
            f.write(f'{scaled:.9f} {p}\n')

    print(f'\n已保存 {len(entries)} 行到 {output_path}')
    print(f'归一化前总权重: {total:.9f}')

    # >>> 监控输出：最终每个大类的权重 <<<
    print('\n' + '-' * 72)
    print('最终各大类权重:')
    print(f'  {"大类":15s}  {"目标":>10s}  {"实际":>10s}  {"差值":>10s}')
    for cls in sorted(CLASS_RECIPE.keys()):
        target = CLASS_RECIPE[cls]
        actual = final_class_weight.get(cls, 0.0)
        print(f'  {cls:15s}  {target:10.6f}  {actual:10.6f}  {actual-target:+10.6f}')

    # >>> 监控输出：最终每个子类的权重 <<<
    print('\n最终各子类权重:')
    print(f'  {"大类":15s}  {"子类":20s}  {"权重":>12s}')
    for cls in sorted(CLASS_RECIPE.keys()):
        for sc in sorted(SUBCLASS_RECIPE.get(cls, {}).keys()):
            sw = final_subclass_weight.get(sc, 0.0)
            print(f'  {cls:15s}  {sc:20s}  {sw:12.6f}')

    # >>> 监控输出：分桶权重调整前后对比 <<<
    original_total = sum(dataset_weight.values())
    original_bucket_weight: dict[str, float] = defaultdict(float)
    for w, _p, _ds, bkt in [(original_weights[i], entries[i][1], entries[i][2], entries[i][3]) for i in range(len(entries)) if i in original_weights]:
        original_bucket_weight[bkt] += w

    bucket_order = ['0k_4k', 'lt_4k', '4k_8k', '4k_16k', '8k_16k',
                    '16k_32k', '32k_64k', '64k_128k', '128k_256k',
                    '256k_512k', '512k_gt', 'qa', 'unknown']
    print('\n各分桶权重占比（调整前 vs 调整后）:')
    print(f'  {"分桶":15s}  {"调整前":>10s}  {"调整后":>10s}  {"变化":>10s}')
    print(f'  {"-"*15}  {"-"*10}  {"-"*10}  {"-"*10}')
    for bkt in bucket_order:
        before = original_bucket_weight.get(bkt, 0.0) / original_total if original_total > 0 else 0.0
        after = final_bucket_weight.get(bkt, 0.0)
        if before > 0 or after > 0:
            diff = after - before
            print(f'  {bkt:15s}  {before:10.6f}  {after:10.6f}  {diff:+10.6f}')

    # >>> 监控输出：每个数据集调整前后的权重占比 <<<
    all_ds = set(list(dataset_weight.keys()) + list(final_dataset_weight.keys()))
    ds_diffs = []
    for ds in all_ds:
        before = dataset_weight.get(ds, 0.0) / original_total if original_total > 0 else 0.0
        after = final_dataset_weight.get(ds, 0.0)
        ds_diffs.append((ds, before, after, after - before))
    ds_diffs.sort(key=lambda x: abs(x[3]), reverse=True)

    print('\n' + '-' * 72)
    print('各数据集权重占比（调整前 vs 调整后，按变化绝对值降序）:')
    print(f'  {"数据集":50s}  {"子类":15s}  {"调整前":>10s}  {"调整后":>10s}  {"变化":>10s}')
    print(f'  {"-"*50}  {"-"*15}  {"-"*10}  {"-"*10}  {"-"*10}')
    for ds, before, after, diff in ds_diffs:
        sc = ds_subclass_map.get(ds, 'other')
        print(f'  {ds:50s}  {sc:15s}  {before:10.6f}  {after:10.6f}  {diff:+10.6f}')

    final_total = sum(final_class_weight.values())
    print(f'\n最终总权重 = {final_total:.9f}')
    print(f'被裁剪的分桶数: {clipped_count}')
    print(f'因条数不足丢弃的分桶数: {dropped_low_entry}')
    print('完成。')


if __name__ == '__main__':
    main()
