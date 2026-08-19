#!/bin/bash

# Runs Mixtral 8x7B model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
# GPUS_PER_NODE=1
# export CUDA_VISIBLE_DEVICES=0
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6002"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

# CHECKPOINT_PATH=$1
# TOKENIZER_MODEL=$2
# DATA_PATH=$3

# DATA_PATH="0.133926750 /user/traindoc/text/by_minicpm3_tokenizer.74k_241216/fineweb_hqdata_v5
# 0.127760108 /user/traindoc/text/by_minicpm3_tokenizer.74k/the_stack_v2_rule_filtered_transformed
# 0.057951320 /user/traindoc/text/by_minicpm3_tokenizer.74k_241216/RefineCode-code-corpus
# 0.075227407 /user/traindoc/text/by_minicpm3_tokenizer.74k_241216/Nemotron_quality_high_kind_actual_kind2_actual
# 0.050243922 /user/traindoc/text/by_minicpm3_tokenizer.74k/code_dedup_new_rule_filtered_v1_transformed
# 0.011653421 /datasets/the_stack_deduped_v2_all_fim_sstable
# 0.046128198 /user/traindoc/text/by_minicpm3_tokenizer.74k_241216/Nemotron_quality_high_kind_synthetic_kind2_diverse_qa_pairs
# 0.077184379 /user/caijie/sstable/chinese-fineweb-edu-v2
# 0.037015142 /user/traindoc/text/by_minicpm3_tokenizer.74k_241216/Nemotron_quality_high_kind_synthetic_kind2_wrap_medium
# 0.029936928 /user/traindoc/text/by_minicpm3_tokenizer.74k_241216/Nemotron_quality_high_kind_synthetic_kind2_extract_knowledge
# 0.000000000 /user/zhangyixuan/data/sstable/hqdata_zh_exp_v3_merge/zh_seed_exp_v3_rc2_eqpos_eqneg
# 0.027846398 /user/traindoc/text/by_minicpm3_tokenizer.74k/cwp_clean_head_pos_transformed_new
# 0.019921757 /user/caijie/sstable/finemath-3plus
# 0.020587167 /user/traindoc/text/by_minicpm3_tokenizer.74k_241216/Nemotron_quality_high_kind_synthetic_kind2_knowledge_list
# 0.019543705 /user/traindoc/text/by_minicpm3_tokenizer.74k/jupyter_notebook_markdown_transformed
# 0.019031360 /user/traindoc/text/by_minicpm3_tokenizer.74k/stack_v2_full_not_code_transformed
# 0.016388770 /user/traindoc/text/by_minicpm3_tokenizer.74k_241216/Nemotron_quality_high_kind_synthetic_kind2_distill
# 0.014273354 /user/traindoc/text/by_minicpm3_tokenizer.74k/cc_math_transformed
# 0.013774528 /user/traindoc/text/by_minicpm3_tokenizer.74k/proof-pile-arxiv_transformed
# 0.012415772 /user/caijie/sstable/infiwebmath-3plus
# 0.012056183 /user/traindoc/text/by_minicpm3_tokenizer.74k/cwp_clean_mid_pos_transformed_new
# 0.008963510 /user/caijie/sstable/fineweb2_left
# 0.000000000 /datasets/minicpm3_4b_sft_all/k12_full_new_0712
# 0.010777465 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.chineseall_epub
# 0.007842096 /user/traindoc/text/by_minicpm3_tokenizer.74k/proof-pile-open-web-math_transformed
# 0.007625125 /user/traindoc/text/by_minicpm3_tokenizer.74k/peS2o
# 0.008980551 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh_common_crawl_pos_transformed
# 0.008934610 /user/traindoc/text/by_minicpm3_tokenizer.74k/cc_hm_clean_pos_transformed
# 0.007082892 /user/caijie/sstable/law_pretrain/law_sst
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/dm_math_clean
# 0.006692057 /user/caijie/sstable/fineweb2_rus
# 0.006110222 /user/traindoc/text/by_minicpm3_tokenizer.74k/nllb
# 0.006144271 /user/traindoc/text/by_minicpm3_tokenizer.74k/cwp_clean_tail_pos_transformed_new
# 0.004546562 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.zhihu-article
# 0.006107642 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.sj_txt
# 0.003851072 /user/traindoc/text/by_minicpm3_tokenizer.74k/proof-pile-algebraic-stack_transformed
# 0.007692121 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh_cwp_head_transformed
# 0.004726532 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.epubee
# 0.004721710 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.zp_ebook
# 0.003196671 /user/traindoc/text/by_minicpm3_tokenizer.74k/yayi_pos_transformed
# 0.002351324 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.zhihu-answer_0321
# 0.002325393 /user/traindoc/text/by_minicpm3_tokenizer.74k/arxiv
# 0.002278673 /user/traindoc/text/by_minicpm3_tokenizer.74k/books3_long_context
# 0.002165583 /user/caijie/sstable/fineweb2_deu
# 0.002156097 /user/traindoc/text/by_minicpm3_tokenizer.74k/wechat_account
# 0.004353468 /user/zhangyixuan/data/sstable/zh_source/tele
# 0.001907781 /user/caijie/sstable/fineweb2_jpn
# 0.001817142 /user/caijie/sstable/fineweb2_spa
# 0.003847063 /user/zhangyixuan/data/sstable/zh_source/cci3_all
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/leetcode
# 0.001722726 /user/traindoc/text/by_minicpm3_tokenizer.74k/zhihu_qa
# 0.003591983 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh_cwp_mid_transformed
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/eq_1103
# 0.001532091 /user/caijie/sstable/fineweb2_fra
# 0.002898237 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.novel.c
# 0.000000000 /datasets/minicpm3_4b_sft_all/math_webinstructsub
# 0.001175267 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.zhihu_0320
# 0.001163448 /user/traindoc/text/by_minicpm3_tokenizer.74k/shenqing_yuqing
# 0.002481017 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh_cc_hm
# 0.002323008 /user/zhangyixuan/data/sstable/zh_source/wanjuan_chinese_web
# 0.001029447 /user/caijie/sstable/fineweb2_ita
# 0.000915649 /user/traindoc/text/by_minicpm3_tokenizer.74k/stack_overflow
# 0.001929793 /user/caijie/sstable/nemotron-medium202
# 0.001912150 /user/caijie/sstable/nemotron-medium201
# 0.001876880 /user/zhangyixuan/data/sstable/chinese_cc_pretrain_dataset/mb_zh_250113
# 0.000000000 /datasets/minicpm3_4b_sft_all/classical_chinese_full_new_0712
# 0.000814103 /user/caijie/sstable/fineweb2_por
# 0.000000000 /user/zhangyixuan/data/sstable/sft_250106/smoltalk_chinese_all
# 0.000000000 /datasets/minicpm3_4b_sft_all/math_numina
# 0.001703971 /user/traindoc/text/by_minicpm3_tokenizer.74k/novel8080_long_context
# 0.000755146 /user/caijie/sstable/law_pretrain/dffl_clean_sst
# 0.000000000 /datasets/minicpm3_4b_sft_all/math_merge_meta_instruct_plus
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/size_math
# 0.000000000 /datasets/minicpm3_4b_sft_all/math_k12_cn
# 0.000890720 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.magazine.1
# 0.000887206 /user/traindoc/text/by_minicpm3_tokenizer.74k/qikan_chinese
# 0.000000000 /user/zhangyixuan/data/sstable/sft_241230/mmlu_knowledge_merge_v0v1v2
# 0.000000000 /user/zhouge/decay_en_sstable/openhermes2_5
# 0.000000000 /datasets/minicpm3_4b_sft_all/couplet_full_new_0712
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/math_k12_1103
# 0.000000000 /datasets/minicpm3_4b_sft_all/math_k12_knowledge_wenku
# 0.000539664 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.baijiahao
# 0.000535358 /user/traindoc/text/by_minicpm3_tokenizer.74k/baike_chinese_new_all
# 0.000000000 /user/tc_agi/llm/sstable_datasets/libretext
# 0.000937959 /user/caijie/sstable/nemotron-medium-high
# 0.000000000 /user/zhangyixuan/data/sstable/sft_241230/cmmlu_cot_random_style_5x
# 0.000858222 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.novel.zzdxss
# 0.000000000 /datasets/minicpm3_4b_sft_all/math_k12_trans-to-en
# 0.000000000 /datasets/minicpm3_4b_sft_all/decay_en_if_081602_chatml
# 0.000343448 /user/traindoc/text/by_minicpm3_tokenizer.74k/stack_exchange_qa
# 0.000720524 /user/traindoc/text/by_minicpm3_tokenizer.74k/qinkan_long_context
# 0.000326157 /user/traindoc/text/by_minicpm3_tokenizer.74k/wikipedia
# 0.000703609 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.yayi
# 0.000000000 /user/zhouge/decay_en_sstable/magpie
# 0.000000000 /user/tc_agi/llm/megatron_datasets/web_code_zh_dedup_new
# 0.000649360 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.novel.ijjxsw
# 0.000000000 /user/zhangyixuan/data/sstable/sft_241230/chinese_knowledge_v1v2
# 0.000288730 /user/traindoc/text/by_minicpm3_tokenizer.74k/ultratextbook
# 0.000000000 /datasets/minicpm3_4b_sft_all/evol_code_clean_dedup_whoru
# 0.000000000 /datasets/minicpm3_4b_sft_all/ultrainteract_0813
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/humanevallike_clean_dedup
# 0.000000000 /user/zhangyixuan/data/sstable/sft_241230/mmlu_cot_random_style_5x
# 0.000213564 /user/traindoc/text/by_minicpm3_tokenizer.74k/douban
# 0.000000000 /datasets/minicpm3_4b_sft_all/gcode_0813
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/souyun
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/math_cal_easy
# 0.000000000 /user/zhouge/decay_en_sstable/close_llm_instruct
# 0.000000000 /user/zhangyixuan/data/sstable/sft_241230/mmlu_like_random_style_5x
# 0.000000000 /datasets/minicpm3_4b_sft_all/math_college_en
# 0.000000000 /user/zhouge/decay_en_sstable/reasoning_0730
# 0.000191869 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.weibo_0320
# 0.000184428 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.q4_news_10_100
# 0.000000000 /datasets/minicpm3_4b_sft_all/math_k12en_numina_stepwise
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/arithemetic
# 0.000000000 /datasets/minicpm3_4b_sft_all/glaive
# 0.000000000 /user/zhouge/decay_en_sstable/magpie_gp
# 0.000000000 /datasets/minicpm3_4b_sft_all/math_trans-from-en
# 0.000155540 /user/caijie/sstable/law_pretrain/8_xingzhengchufa_sst
# 0.000204591 /user/traindoc/text/by_minicpm3_tokenizer.74k/github_repo
# 0.000207355 /user/traindoc/text/by_minicpm3_tokenizer.74k/ebook_kindle
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/dm_math_annotate
# 0.000000000 /datasets/minicpm3_4b_sft_all/mtbench_like_no_sys
# 0.000149599 /user/traindoc/text/by_minicpm3_tokenizer.74k/mt_lab
# 0.000000000 /datasets/minicpm3_4b_sft_all/choices
# 0.000000000 /datasets/minicpm3_4b_sft_all/math_college_cnq_trans-from-enq
# 0.000195992 /user/traindoc/text/by_minicpm3_tokenizer.74k/ebook_epubee
# 0.000138217 /user/traindoc/text/by_minicpm3_tokenizer.74k/books1_long_context
# 0.000184052 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.boke_kindle_mobi
# 0.000133398 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.hdf_0320
# 0.000179802 /user/traindoc/text/by_minicpm3_tokenizer.74k/ebook_sobooks
# 0.000000000 /user/zhangyixuan/data/sstable/sft_241230/cmmlu_like_random_style_5x
# 0.000000000 /user/tc_agi/llm/megatron_datasets/web_code_en_dedup_new
# 0.000000000 /datasets/minicpm3_4b_sft_all/math_primary_mwp
# 0.000000000 /user/zhouge/decay_en_sstable/CoT_collection_en
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/code_30w
# 0.000000000 /datasets/minicpm3_4b_sft_all/lccf_apps_ultracode_aug_data_0818
# 0.000110977 /user/traindoc/text/by_minicpm3_tokenizer.74k/hi138_data
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/weikao
# 0.000000000 /user/wf/train_data/sstable/codefeedback_0624_4k_new
# 0.000000000 /datasets/minicpm3_4b_sft_all/sft_old_rewrite_no_sys
# 0.000000000 /user/zhouge/decay_en_sstable/multifaceted_collection_sft
# 0.000099801 /user/traindoc/text/by_minicpm3_tokenizer.74k/people_daily
# 0.000000000 /user/zhangyixuan/data/sstable/sft_241230/ceval_cot_random_style_5x
# 0.000195273 /user/traindoc/text/by_minicpm3_tokenizer.74k/novel.17k
# 0.000194630 /user/traindoc/text/by_minicpm3_tokenizer.74k/novel.yd_baidu
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/10w_why
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/tinycode_sciphi
# 0.000080514 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.q4_news_0_10
# 0.000074832 /user/traindoc/text/by_minicpm3_tokenizer.74k/csdn_parse_filter
# 0.000000000 /datasets/minicpm3_4b_sft_all/xlam_function_calling_60k
# 0.000000000 /datasets/minicpm3_4b_sft_all/fbdata_v40
# 0.000071663 /user/traindoc/text/by_minicpm3_tokenizer.74k/shenqing_news_20231109
# 0.000068817 /user/caijie/sstable/law_pretrain/3_book_sst
# 0.000067316 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.q4_news_gt_100
# 0.000091055 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.ebook_wode
# 0.000000000 /datasets/minicpm3_4b_sft_all/k12_icl_no_sys
# 0.000060524 /user/traindoc/text/by_minicpm3_tokenizer.74k/shenqing_xingguang
# 0.000075858 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.boke_kindle_epub
# 0.000073028 /user/traindoc/text/by_minicpm3_tokenizer.74k/ebook_ebooks_1_to_6
# 0.000000000 /datasets/minicpm3_4b_sft_all/knowledge_rewrite_no_sys
# 0.000048347 /user/traindoc/text/by_minicpm3_tokenizer.74k/360_doc_personal_library
# 0.000101932 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.novel.pan_jinjiang_2022_2023_10_30__2023_12_01_17
# 0.000000000 /datasets/minicpm3_4b_sft_all/code_0
# 0.000097897 /user/traindoc/text/by_minicpm3_tokenizer.74k/starting_point_chinese_website
# 0.000043618 /user/caijie/sstable/law_pretrain/law_sst_flqk
# 0.000088264 /user/traindoc/text/by_minicpm3_tokenizer.74k/xiaoxiang_academy_data
# 0.000000000 /datasets/minicpm3_4b_sft_all/math_k12en_numina_stepwise_error_recovery
# 0.000054253 /user/traindoc/text/by_minicpm3_tokenizer.74k/ebook_panda_reading
# 0.000000000 /datasets/minicpm3_4b_sft_all/ab_like2_filter
# 0.000000000 /user/tc_agi/zhouge/decay/num_format_task_0715_280k_wo_sys
# 0.000031673 /user/caijie/sstable/law_pretrain/law_sst_cpws
# 0.000067462 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.novel.pan_baidu
# 0.000029476 /user/traindoc/text/by_minicpm3_tokenizer.74k/refined-anime-text
# 0.000000000 /user/zhangyixuan/data/sstable/sft_241230/ceval_like_random_style_5x
# 0.000060415 /user/traindoc/text/by_minicpm3_tokenizer.74k/feilu
# 0.000024843 /user/caijie/sstable/law_pretrain/law_sst_zyfl
# 0.000023219 /user/traindoc/text/by_minicpm3_tokenizer.74k/chn_dic
# 0.000022021 /user/traindoc/text/by_minicpm3_tokenizer.74k/en.wikihow
# 0.000020452 /user/traindoc/text/by_minicpm3_tokenizer.74k/ctrip_trave_v1_clean
# 0.000020043 /user/traindoc/text/by_minicpm3_tokenizer.74k/mbalib_data_clean
# 0.000041003 /user/traindoc/text/by_minicpm3_tokenizer.74k/zxcs_long_context
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/whys
# 0.000034181 /user/traindoc/text/by_minicpm3_tokenizer.74k/zongheng_chinese_network_v1_clean
# 0.000000000 /datasets/minicpm3_4b_sft_all/dscodefb_0720
# 0.000014338 /user/caijie/sstable/law_pretrain/law_sst_flsy
# 0.000013795 /user/traindoc/text/by_minicpm3_tokenizer.74k/xuexiqiangguo
# 0.000000000 /datasets/minicpm3_4b_sft_all/task_oriented
# 0.000000000 /datasets/minicpm3_4b_sft_all/functioncalling_chatml
# 0.000028003 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.kara_jinjiang_literature_city
# 0.000011999 /user/traindoc/text/by_minicpm3_tokenizer.74k/xiaohongshu
# 0.000000000 /datasets/minicpm3_4b_sft_all/logi_merge_no_sys
# 0.000011387 /user/traindoc/text/by_minicpm3_tokenizer.74k/gov_safety
# 0.000000000 /datasets/minicpm3_4b_sft_all/bbh_improve_0829_v9
# 0.000023888 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.novel.taobao
# 0.000010750 /user/traindoc/text/by_minicpm3_tokenizer.74k/moshadong
# 0.000010300 /user/traindoc/text/by_minicpm3_tokenizer.74k/mt_book_dic
# 0.000009832 /user/caijie/sstable/law_pretrain/4_faxinma_sst
# 0.000009801 /user/traindoc/text/by_minicpm3_tokenizer.74k/tianya_forum
# 0.000000000 /datasets/minicpm3_4b_sft_all/parallel_functioncall
# 0.000007515 /user/traindoc/text/by_minicpm3_tokenizer.74k/ancient_chinese_poetry_clean
# 0.000000000 /user/zhouge/decay_en_sstable/openassistant_best_replies_train
# 0.000007144 /user/traindoc/text/by_minicpm3_tokenizer.74k/jianshu
# 0.000006944 /user/traindoc/text/by_minicpm3_tokenizer.74k/what_is_worth_buying
# 0.000000000 /datasets/minicpm3_4b_sft_all/align_no_sys
# 0.000006768 /user/traindoc/text/by_minicpm3_tokenizer.74k/poem
# 0.000005635 /user/caijie/sstable/law_pretrain/law_sst_twsy
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/leetcode_pass_code
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/logi
# 0.000004622 /user/caijie/sstable/law_pretrain/law_sst_lf
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/leetcode_pass_code_0125
# 0.000004299 /user/caijie/sstable/law_pretrain/2_qa_sst
# 0.000008570 /user/traindoc/text/by_minicpm3_tokenizer.74k/aiqu_27txt
# 0.000003500 /user/traindoc/text/by_minicpm3_tokenizer.74k/economic_information_daily_clean
# 0.000003389 /user/traindoc/text/by_minicpm3_tokenizer.74k/en.boke_kindle_mobi
# 0.000000000 /user/traindoc/text/by_minicpm3_tokenizer.74k/wordproblem_1013
# 0.000002496 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh_kara_netease_cloud_reading
# 0.000001813 /user/caijie/sstable/law_pretrain/law_sst_sf
# 0.000001693 /user/traindoc/text/by_minicpm3_tokenizer.74k/shenqingbaike_sq
# 0.000000000 /user/zhangyixuan/data/sstable/sft_250106/waijiaobu_qa
# 0.000000000 /user/zhangyixuan/data/sstable/sft_250106/zhengfu_qa
# 0.000001389 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.linux_cn
# 0.000001244 /user/traindoc/text/by_minicpm3_tokenizer.74k/kara_go_kitchen_clean
# 0.000000000 /user/zhangyixuan/data/sstable/sft_250106/guowuyuan_qa
# 0.000001114 /user/caijie/sstable/law_pretrain/law_sst_gjty
# 0.000001084 /user/traindoc/text/by_minicpm3_tokenizer.74k/zh.wikihow
# 0.000001043 /user/traindoc/text/by_minicpm3_tokenizer.74k/1905_film
# 0.000000000 /user/zhangyixuan/data/sstable/sft_250106/gaokao_essay
# 0.000000932 /user/caijie/sstable/law_pretrain/law_sst_twk
# 0.000000789 /user/caijie/sstable/law_pretrain/law_sst_xgk
# 0.000000000 /datasets/minicpm3_4b_sft_all/codeinterpreter_0720
# 0.000000670 /user/caijie/sstable/law_pretrain/5_falvyange_sst
# 0.000000530 /user/caijie/sstable/law_pretrain/6_tiantongma_sst
# 0.000000453 /user/caijie/sstable/law_pretrain/law_sst_amk
# 0.000000402 /user/caijie/sstable/law_pretrain/alk_sst"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --vocab-size 130560
    --make-vocab-size-divisible-by 1
    --disable-bias-linear
    --seq-length 131072
    --max-position-embeddings 131072
    --num-layers 42
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 2
    --kv-channels 128
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 5000000
    --norm-epsilon 1e-6
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model /user/daizhenning2781/tokenizer/backup/tokenizer_v9.6/
    # --data-path $DATA_PATH
    --split 99990,8,2
    --use-modelbest-sdk
    # --no-load-data-state
    --dataloader-type external
)

TRAINING_ARGS=(
    # --micro-batch-size 16
    # --global-batch-size 32
    # --lr 2e-4
    # --train-iters 40000
    # --lr-decay-iters 40000
    # --lr-wsd-decay-iters 3000
    --lr-decay-style WSD
    --lr-wsd-decay-style exponential
    # --min-lr 2e-5
    --weight-decay 0.1
    # --lr-warmup-iters 400
    --clip-grad 1.0
    --bf16
    --optimizer muon
    --adam-beta1 0.9
    --adam-beta2 0.95
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 12
    --decoder-last-pipeline-num-layers 6
    --use-distributed-optimizer
    --sequence-parallel
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

LOGGING_ARGS=(
    --log-interval 1 \
    # --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH \
    # --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --log-task-loss-interval 100 \
    --async-save \
    --ckpt-format torch_dist \
    --dist-ckpt-strictness log_all \
    # --use-persistent-ckpt-worker
)

set -ex
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_minicpm.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    $@
