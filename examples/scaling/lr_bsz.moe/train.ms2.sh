#!/bin/bash

# Runs Mixtral 8x7B model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

# CHECKPOINT_PATH=$1
# TOKENIZER_MODEL=$2
# DATA_PATH=$3

# DATA_PATH="0.075648453 /user/zhangyixuan/data/pretrain/sstable_251103/en_hqdata_v5
# 0.130105581 /user/zhangyixuan/data/pretrain/sstable_251103/the_stack_v2_rule_filtered_transformed
# 0.098201317 /user/zhangyixuan/data/pretrain/sstable_251103/RefineCode-code-corpus
# 0.042492161 /user/zhangyixuan/data/pretrain/sstable_251103/Nemotron_quality_high_kind_actual_kind2_actual
# 0.038246825 /user/zhangyixuan/data/pretrain/sstable_251103/code_dedup_new_rule_filtered_v1_transformed
# 0.011354690 /user/zhangyixuan/data/pretrain/sstable_251103/stack_v2_fim
# 0.039083232 /user/zhangyixuan/data/pretrain/sstable_251103/Nemotron_quality_high_kind_synthetic_kind2_diverse_qa_pairs
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/chinese_fineweb_edu_v2
# 0.031361975 /user/zhangyixuan/data/pretrain/sstable_251103/Nemotron_quality_high_kind_synthetic_kind2_wrap_medium
# 0.025364786 /user/zhangyixuan/data/pretrain/sstable_251103/Nemotron_quality_high_kind_synthetic_kind2_extract_knowledge
# 0.045745804 /user/zhangyixuan/data/pretrain/sstable_251103/zh_hqdata_v3
# 0.019661279 /user/zhangyixuan/data/pretrain/sstable_251103/cwp_clean_head_pos_transformed_new
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/finemath-3plus
# 0.017442976 /user/zhangyixuan/data/pretrain/sstable_251103/Nemotron_quality_high_kind_synthetic_kind2_knowledge_list
# 0.012544604 /user/zhangyixuan/data/pretrain/sstable_251103/jupyter_notebook_markdown_transformed
# 0.012529390 /user/zhangyixuan/data/pretrain/sstable_251103/stack_v2_full_not_code_transformed
# 0.013885782 /user/zhangyixuan/data/pretrain/sstable_251103/Nemotron_quality_high_kind_synthetic_kind2_distill
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/cc_math_transformed
# 0.001823563 /user/zhangyixuan/data/pretrain/sstable_251103/proof-pile-arxiv_transformed
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/infiwebmath-3plus
# 0.008512411 /user/zhangyixuan/data/pretrain/sstable_251103/cwp_clean_mid_pos_transformed_new
# 0.002847957 /user/zhangyixuan/data/pretrain/sstable_251103/fineweb2_left
# 0.031156881 /user/zhangyixuan/data/pretrain/sstable_251103/k12_full_new_0712
# 0.001426792 /user/zhangyixuan/data/pretrain/sstable_251103/zh.chineseall_epub
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/proof-pile-open-web-math_transformed
# 0.004037858 /user/zhangyixuan/data/pretrain/sstable_251103/peS2o
# 0.006340824 /user/zhangyixuan/data/pretrain/sstable_251103/zh_common_crawl_pos_transformed
# 0.006308387 /user/zhangyixuan/data/pretrain/sstable_251103/cc_hm_clean_pos_transformed
# 0.001875360 /user/zhangyixuan/data/pretrain/sstable_251103/law_sst
# 0.027175700 /user/zhangyixuan/data/pretrain/sstable_251103/dm_math_clean
# 0.002126253 /user/zhangyixuan/data/pretrain/sstable_251103/fineweb2_rus
# 0.001617823 /user/zhangyixuan/data/pretrain/sstable_251103/nllb
# 0.004338235 /user/zhangyixuan/data/pretrain/sstable_251103/cwp_clean_tail_pos_transformed_new
# 0.001203808 /user/zhangyixuan/data/pretrain/sstable_251103/zh.zhihu-article
# 0.000808570 /user/zhangyixuan/data/pretrain/sstable_251103/zh.sj_txt
# 0.000509830 /user/zhangyixuan/data/pretrain/sstable_251103/proof-pile-algebraic-stack_transformed
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/zh_cwp_head_transformed
# 0.000625730 /user/zhangyixuan/data/pretrain/sstable_251103/zh.epubee
# 0.000625091 /user/zhangyixuan/data/pretrain/sstable_251103/zh.zp_ebook
# 0.002257047 /user/zhangyixuan/data/pretrain/sstable_251103/yayi_pos_transformed
# 0.000622568 /user/zhangyixuan/data/pretrain/sstable_251103/zh.zhihu-answer_0321
# 0.001231404 /user/zhangyixuan/data/pretrain/sstable_251103/arxiv_ocr
# 0.000603332 /user/zhangyixuan/data/pretrain/sstable_251103/books3_long_context
# 0.000688066 /user/zhangyixuan/data/pretrain/sstable_251103/fineweb2_deu
# 0.000570877 /user/zhangyixuan/data/pretrain/sstable_251103/wechat_account
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/tele
# 0.000606155 /user/zhangyixuan/data/pretrain/sstable_251103/fineweb2_jpn
# 0.000577357 /user/zhangyixuan/data/pretrain/sstable_251103/fineweb2_spa
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/cci3_all
# 0.003497046 /user/zhangyixuan/data/pretrain/sstable_251103/leetcode
# 0.000456132 /user/zhangyixuan/data/pretrain/sstable_251103/zhihu_qa
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/zh_cwp_mid_transformed
# 0.004782444 /user/zhangyixuan/data/pretrain/sstable_251103/eq_1103
# 0.000486788 /user/zhangyixuan/data/pretrain/sstable_251103/fineweb2_fra
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/zh.novel.c
# 0.007669248 /user/zhangyixuan/data/pretrain/sstable_251103/math_webinstructsub
# 0.000311179 /user/zhangyixuan/data/pretrain/sstable_251103/zh.zhihu_0320
# 0.000308050 /user/zhangyixuan/data/pretrain/sstable_251103/shenqing_yuqing
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/zh_cc_hm
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/wanjuan_chinese_web
# 0.000327084 /user/zhangyixuan/data/pretrain/sstable_251103/fineweb2_ita
# 0.000242439 /user/zhangyixuan/data/pretrain/sstable_251103/stack_overflow
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/mb_zh_250113
# 0.001301817 /user/zhangyixuan/data/pretrain/sstable_251103/classical_chinese_full_new_0712
# 0.000258663 /user/zhangyixuan/data/pretrain/sstable_251103/fineweb2_por
# 0.004725578 /user/zhangyixuan/data/pretrain/sstable_251103/math_numina
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/novel8080
# 0.000199942 /user/zhangyixuan/data/pretrain/sstable_251103/dffl_clean_sst
# 0.003980306 /user/zhangyixuan/data/pretrain/sstable_251103/math_merge_meta_instruct_plus
# 0.003494325 /user/zhangyixuan/data/pretrain/sstable_251103/size_math
# 0.003954213 /user/zhangyixuan/data/pretrain/sstable_251103/math_k12_cn
# 0.000117919 /user/zhangyixuan/data/pretrain/sstable_251103/zh.magazine.1
# 0.000117454 /user/zhangyixuan/data/pretrain/sstable_251103/qikan_chinese
# 0.001180072 /user/zhangyixuan/data/sstable/sft_241230/mmlu_knowledge_merge_v0v1v2
# 0.002359388 /user/zhangyixuan/data/pretrain/sstable_251103/openhermes2_5
# 0.000927556 /user/zhangyixuan/data/pretrain/sstable_251103/couplet_full_new_0712
# 0.002303398 /user/zhangyixuan/data/pretrain/sstable_251103/math_k12_1103
# 0.003267873 /user/zhangyixuan/data/pretrain/sstable_251103/math_k12_knowledge_wenku
# 0.000142889 /user/zhangyixuan/data/pretrain/sstable_251103/zh.baijiahao
# 0.000141748 /user/zhangyixuan/data/pretrain/sstable_251103/baike_chinese_new_all
# 0.000568026 /user/zhangyixuan/data/pretrain/sstable_251103/libretext
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/nemotron-medium-high
# 0.000000000 /user/zhangyixuan/data/sstable/sft_241230/cmmlu_cot_random_style_5x
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/zh.novel.zzdxss
# 0.002237075 /user/zhangyixuan/data/pretrain/sstable_251103/math_k12_trans-to-en
# 0.001414798 /user/zhangyixuan/data/pretrain/sstable_251103/decay_en_if_081602_chatml
# 0.000090936 /user/zhangyixuan/data/pretrain/sstable_251103/stack_exchange_qa
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/qinkan_long_context
# 0.000086358 /user/zhangyixuan/data/pretrain/sstable_251103/wikipedia
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/zh.yayi
# 0.001279094 /user/zhangyixuan/data/pretrain/sstable_251103/magpie
# 0.000622693 /user/zhangyixuan/data/pretrain/sstable_251103/web_code_zh_dedup_new
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/zh.novel.ijjxsw
# 0.000966933 /user/zhangyixuan/data/sstable/sft_241230/chinese_knowledge_v1v2
# 0.000152896 /user/zhangyixuan/data/pretrain/sstable_251103/ultratextbook
# 0.000562573 /user/zhangyixuan/data/pretrain/sstable_251103/evol_code_clean_dedup_whoru
# 0.001101149 /user/zhangyixuan/data/pretrain/sstable_251103/ultrainteract_0813
# 0.000069662 /user/zhangyixuan/data/pretrain/sstable_251103/humanevallike_clean_dedup
# 0.001284118 /user/zhangyixuan/data/sstable/sft_241230/mmlu_cot_random_style_5x
# 0.000056546 /user/zhangyixuan/data/pretrain/sstable_251103/douban
# 0.000411749 /user/zhangyixuan/data/pretrain/sstable_251103/gcode_0813
# 0.000282712 /user/zhangyixuan/data/pretrain/sstable_251103/souyun
# 0.000784146 /user/zhangyixuan/data/pretrain/sstable_251103/close_llm_instruct
# 0.001036857 /user/zhangyixuan/data/sstable/sft_241230/mmlu_like_random_style_5x
# 0.000960518 /user/zhangyixuan/data/pretrain/sstable_251103/math_college_en
# 0.000766523 /user/zhangyixuan/data/pretrain/sstable_251103/reasoning_0730
# 0.000050802 /user/zhangyixuan/data/pretrain/sstable_251103/zh.weibo_0320
# 0.000048832 /user/zhangyixuan/data/pretrain/sstable_251103/zh.q4_news_10_100
# 0.000884030 /user/zhangyixuan/data/pretrain/sstable_251103/math_k12en_numina_stepwise
# 0.001046311 /user/zhangyixuan/data/pretrain/sstable_251103/arithemetic
# 0.000346582 /user/zhangyixuan/data/pretrain/sstable_251103/glaive
# 0.000657651 /user/zhangyixuan/data/pretrain/sstable_251103/magpie_gp
# 0.000947395 /user/zhangyixuan/data/pretrain/sstable_251103/math_trans-from-en
# 0.000041183 /user/zhangyixuan/data/pretrain/sstable_251103/8_xingzhengchufa_sst
# 0.000162511 /user/zhangyixuan/data/pretrain/sstable_251103/github_repo
# 0.000027451 /user/zhangyixuan/data/pretrain/sstable_251103/ebook_kindle
# 0.000747153 /user/zhangyixuan/data/pretrain/sstable_251103/dm_math_annotate
# 0.000594810 /user/zhangyixuan/data/pretrain/sstable_251103/mtbench_like_no_sys
# 0.000039610 /user/zhangyixuan/data/pretrain/sstable_251103/mt_lab
# 0.000289493 /user/zhangyixuan/data/pretrain/sstable_251103/choices
# 0.000720960 /user/zhangyixuan/data/pretrain/sstable_251103/math_college_cnq_trans-from-enq
# 0.000025947 /user/zhangyixuan/data/pretrain/sstable_251103/ebook_epubee
# 0.000036596 /user/zhangyixuan/data/pretrain/sstable_251103/books1_long_context
# 0.000024366 /user/zhangyixuan/data/pretrain/sstable_251103/zh.boke_kindle_mobi
# 0.000023803 /user/zhangyixuan/data/pretrain/sstable_251103/ebook_sobooks
# 0.000501631 /user/zhangyixuan/data/sstable/sft_241230/cmmlu_like_random_style_5x
# 0.000247553 /user/zhangyixuan/data/pretrain/sstable_251103/web_code_en_dedup_new
# 0.000729216 /user/zhangyixuan/data/pretrain/sstable_251103/math_primary_mwp
# 0.000474908 /user/zhangyixuan/data/pretrain/sstable_251103/CoT_collection_en
# 0.000231020 /user/zhangyixuan/data/pretrain/sstable_251103/code_30w
# 0.000221620 /user/zhangyixuan/data/pretrain/sstable_251103/lccf_apps_ultracode_aug_data_0818
# 0.000058768 /user/zhangyixuan/data/pretrain/sstable_251103/hi138_data
# 0.000146876 /user/zhangyixuan/data/pretrain/sstable_251103/weikao
# 0.000204429 /user/zhangyixuan/data/pretrain/sstable_251103/code_feedback_241230
# 0.000408134 /user/zhangyixuan/data/pretrain/sstable_251103/sft_old_rewrite_no_sys
# 0.000403916 /user/zhangyixuan/data/pretrain/sstable_251103/multifaceted_collection_sft
# 0.000026425 /user/zhangyixuan/data/pretrain/sstable_251103/people_daily
# 0.000000000 /user/zhangyixuan/data/sstable/sft_241230/ceval_cot_random_style_5x
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/novel.17k
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/novel.yd_baidu
# 0.000125918 /user/zhangyixuan/data/pretrain/sstable_251103/10w_why
# 0.000167956 /user/zhangyixuan/data/pretrain/sstable_251103/tinycode_sciphi
# 0.000021318 /user/zhangyixuan/data/pretrain/sstable_251103/zh.q4_news_0_10
# 0.000019814 /user/zhangyixuan/data/pretrain/sstable_251103/csdn_parse_filter
# 0.000147624 /user/zhangyixuan/data/pretrain/sstable_251103/xlam_function_calling_60k
# 0.000626463 /user/zhangyixuan/data/pretrain/sstable_251103/fbdata_v40
# 0.000018975 /user/zhangyixuan/data/pretrain/sstable_251103/shenqing_news_20231109
# 0.000018221 /user/zhangyixuan/data/pretrain/sstable_251103/3_book_sst
# 0.000017824 /user/zhangyixuan/data/pretrain/sstable_251103/zh.q4_news_gt_100
# 0.000012054 /user/zhangyixuan/data/pretrain/sstable_251103/zh.ebook_wode
# 0.000260386 /user/zhangyixuan/data/pretrain/sstable_251103/k12_icl_no_sys
# 0.000016025 /user/zhangyixuan/data/pretrain/sstable_251103/shenqing_xingguang_new
# 0.000010043 /user/zhangyixuan/data/pretrain/sstable_251103/zh.boke_kindle_epub
# 0.000009668 /user/zhangyixuan/data/pretrain/sstable_251103/ebook_ebooks_1_to_6
# 0.000191186 /user/zhangyixuan/data/pretrain/sstable_251103/knowledge_rewrite_no_sys
# 0.000012801 /user/zhangyixuan/data/pretrain/sstable_251103/360_doc_personal_library
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/zh.novel.pan_jinjiang
# 0.000090638 /user/zhangyixuan/data/pretrain/sstable_251103/code_0
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/starting_point_chinese_website
# 0.000011549 /user/zhangyixuan/data/pretrain/sstable_251103/law_sst_flqk
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/xiaoxiang_academy_data
# 0.000200263 /user/zhangyixuan/data/pretrain/sstable_251103/math_k12en_numina_stepwise_error_recovery
# 0.000007182 /user/zhangyixuan/data/pretrain/sstable_251103/ebook_panda_reading
# 0.000153672 /user/zhangyixuan/data/for_sync/ab_like2_filter
# 0.000146647 /user/zhangyixuan/data/pretrain/sstable_251103/num_format_task_0715_280k_wo_sys
# 0.000008386 /user/zhangyixuan/data/pretrain/sstable_251103/law_sst_cpws
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/zh.novel.pan_baidu
# 0.000007804 /user/zhangyixuan/data/pretrain/sstable_251103/refined-anime-text
# 0.000110458 /user/zhangyixuan/data/sstable/sft_241230/ceval_like_random_style_5x
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/feilu
# 0.000006578 /user/zhangyixuan/data/pretrain/sstable_251103/law_sst_zyfl
# 0.000024591 /user/zhangyixuan/data/pretrain/sstable_251103/chn_dic
# 0.000005830 /user/zhangyixuan/data/pretrain/sstable_251103/en.wikihow
# 0.000005415 /user/zhangyixuan/data/pretrain/sstable_251103/ctrip_trave_v1_clean
# 0.000005307 /user/zhangyixuan/data/pretrain/sstable_251103/mbalib_data_clean
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/zxcs_long_context
# 0.000004245 /user/zhangyixuan/data/pretrain/sstable_251103/whys
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/zongheng_chinese_network
# 0.000029383 /user/zhangyixuan/data/for_sync/dscodefb_0720
# 0.000003796 /user/zhangyixuan/data/pretrain/sstable_251103/law_sst_flsy
# 0.000007305 /user/zhangyixuan/data/pretrain/sstable_251103/xuexiqiangguo
# 0.000026677 /user/zhangyixuan/data/pretrain/sstable_251103/task_oriented
# 0.000026466 /user/zhangyixuan/data/pretrain/sstable_251103/functioncalling_chatml
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/zh.kara_jinjiang_literature_city
# 0.000003177 /user/zhangyixuan/data/pretrain/sstable_251103/xiaohongshu
# 0.000046455 /user/zhangyixuan/data/pretrain/sstable_251103/logi_merge_no_sys
# 0.000006030 /user/zhangyixuan/data/pretrain/sstable_251103/gov_safety
# 0.000070317 /user/zhangyixuan/data/pretrain/sstable_251103/bbh_improve_0829_v9
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/zh.novel.taobao
# 0.000011385 /user/zhangyixuan/data/pretrain/sstable_251103/moshadong
# 0.000002727 /user/zhangyixuan/data/pretrain/sstable_251103/mt_book_dic
# 0.000002603 /user/zhangyixuan/data/pretrain/sstable_251103/4_faxinma_sst
# 0.000002595 /user/zhangyixuan/data/pretrain/sstable_251103/tianya_forum
# 0.000017519 /user/zhangyixuan/data/pretrain/sstable_251103/parallel_functioncall
# 0.000007959 /user/zhangyixuan/data/pretrain/sstable_251103/ancient_chinese_poetry_clean
# 0.000029064 /user/zhangyixuan/data/pretrain/sstable_251103/openassistant_best_replies_train
# 0.000001892 /user/zhangyixuan/data/pretrain/sstable_251103/jianshu
# 0.000001839 /user/zhangyixuan/data/pretrain/sstable_251103/what_is_worth_buying
# 0.000038958 /user/zhangyixuan/data/pretrain/sstable_251103/align_no_sys
# 0.000007168 /user/zhangyixuan/data/for_sync/poem
# 0.000001492 /user/zhangyixuan/data/pretrain/sstable_251103/law_sst_twsy
# 0.000009210 /user/zhangyixuan/data/pretrain/sstable_251103/leetcode_pass_code
# 0.000006515 /user/zhangyixuan/data/pretrain/sstable_251103/logi
# 0.000001224 /user/zhangyixuan/data/pretrain/sstable_251103/law_sst_lf
# 0.000008695 /user/zhangyixuan/data/pretrain/sstable_251103/leetcode_pass_code_0125
# 0.000001138 /user/zhangyixuan/data/pretrain/sstable_251103/2_qa_sst
# 0.000000000 /user/zhangyixuan/data/pretrain/sstable_251103/aiqu_27txt
# 0.000000927 /user/zhangyixuan/data/pretrain/sstable_251103/economic_information_daily_clean
# 0.000000897 /user/zhangyixuan/data/pretrain/sstable_251103/en.boke_kindle_mobi
# 0.000016608 /user/zhangyixuan/data/pretrain/sstable_251103/wordproblem_1013
# 0.000000661 /user/zhangyixuan/data/pretrain/sstable_251103/zh_kara_netease_cloud_reading
# 0.000000480 /user/zhangyixuan/data/pretrain/sstable_251103/law_sst_sf
# 0.000000448 /user/zhangyixuan/data/pretrain/sstable_251103/shenqingbaike_sq
# 0.000003222 /user/zhangyixuan/data/pretrain/sstable_251103/waijiaobu_qa
# 0.000003103 /user/zhangyixuan/data/pretrain/sstable_251103/zhengfu_qa
# 0.000000368 /user/zhangyixuan/data/pretrain/sstable_251103/zh.linux_cn
# 0.000002225 /user/zhangyixuan/data/pretrain/sstable_251103/guowuyuan_qa
# 0.000000295 /user/zhangyixuan/data/pretrain/sstable_251103/law_sst_gjty
# 0.000000144 /user/zhangyixuan/data/pretrain/sstable_251103/zh.wikihow
# 0.000000276 /user/zhangyixuan/data/pretrain/sstable_251103/1905_film
# 0.000001857 /user/zhangyixuan/data/pretrain/sstable_251103/gaokao_essay
# 0.000000247 /user/zhangyixuan/data/pretrain/sstable_251103/law_sst_twk
# 0.000000209 /user/zhangyixuan/data/pretrain/sstable_251103/law_sst_xgk
# 0.000001534 /user/zhangyixuan/data/for_sync/codeinterpreter_0720
# 0.000000177 /user/zhangyixuan/data/pretrain/sstable_251103/5_falvyange_sst
# 0.000000140 /user/zhangyixuan/data/pretrain/sstable_251103/6_tiantongma_sst
# 0.000000120 /user/zhangyixuan/data/pretrain/sstable_251103/law_sst_amk
# 0.000000106 /user/zhangyixuan/data/pretrain/sstable_251103/alk_sst
# 0.001167803 /user/zhangyixuan/data/pretrain/sstable_251103/opc_annealing_synthetic_qa_corpus
# 0.002224143 /user/zhangyixuan/data/pretrain/sstable_251103/opc_annealing_synthetic_code_snippet_corpus
# 0.003564717 /user/zhangyixuan/data/pretrain/sstable_251103/opc_annealing_algorithmic_corpus
# 0.029070116 /user/zhangyixuan/data/pretrain/sstable_251103/code_to_strucutre_textbook_250407
# 0.018290878 /user/zhangyixuan/data/pretrain/sstable_251103/code_python_to_testbook_250427
# 0.000002456 /user/zhangyixuan/data/sft_processed/sstbale/humaneval_like_qa_250423_dedup_with_unit_test_pass_0shot_2x
# 0.000918245 /user/zhangyixuan/data/sft_processed/sstbale/mbpp_like_qa_250423_with_unit_test_pass_0shot_2x
# 0.000766310 /user/zhangyixuan/data/sft_processed/sstbale/mbpp_like_qa_250423_with_unit_test_pass_fewshot_2x
# 0.000913353 /user/zhangyixuan/data/sft_processed/sstbale/humaneval_like_old_data_ut_pass_0shot_2x
# 0.000825916 /user/zhangyixuan/data/pretrain/sstable_251103/finemath-4plus
# 0.000738833 /user/zhangyixuan/data/pretrain/sstable_251103/infiwebmath-4plus
# 0.010581912 /user/zhangyixuan/data/pretrain/sstable_251103/finemath_4plus_gen_mind_8_role_all
# 0.025307661 /user/zhangyixuan/data/pretrain/sstable_251103/math_sft_gen_textbook_250304
# 0.033224203 /user/zhangyixuan/data/pretrain/sstable_251103/infinemath4plus_gen_textbook_part_250324
# 0.009721546 /user/zhangyixuan/data/pretrain/sstable_251103/finemath_4plus_gen_middle_school_qa_dedup_250408
# 0.002592502 /user/zhangyixuan/data/pretrain/sstable_251103/finemath_4plus_gen_grade_school_qa_dedup_250408
# 0.007169452 /user/zhangyixuan/data/pretrain/sstable_251103/opc_fineweb_math_corpus
# 0.001425651 /user/zhangyixuan/data/pretrain/sstable_251103/megamath_qa_qwen_25
# 0.005769471 /user/zhangyixuan/data/pretrain/sstable_251103/megamath_text_code_block
# 0.003639687 /user/zhangyixuan/data/pretrain/sstable_251103/megamath_translated_code
# 0.015796360 /user/zhangyixuan/data/pretrain/sstable_251103/megamath_web_pro
# 0.000000000 /user/zhangyixuan/data/sft_processed/sstbale/math_sft_old_rewrite_250429_1x
# 0.000463791 /user/zhangyixuan/data/chinese_sft_250423/sst/ceval_merge_3_style_each_2x_dedup
# 0.002068995 /user/zhangyixuan/data/chinese_sft_250423/sst/cmmlu_merge_3_style_each_2x_dedup
# 0.002017123 /user/zhangyixuan/data/sft_processed/sstbale/deep_think_sft_drop_think
# 0.000412957 /user/zhangyixuan/data/sft_processed/sstbale/nemotron_sft_code
# 0.002675135 /user/zhangyixuan/data/sft_processed/sstbale/math_sft_old_rewrite_250505_1x
# 0.000005067 /user/zhangyixuan/data/sft_processed/sstbale/math500_train_instruct_evol_v0
# 0.000013760 /user/zhangyixuan/data/sft_processed/sstbale/math500_train_instruct_evol_v1
# 0.000363673 /user/zhangyixuan/data/sft_processed/sstbale/nemotron_sft_math
# 0.002675304 /user/zhangyixuan/data/sft_processed/sstbale/jiuzhang_grade_school_250514_1x
# 0.003511593 /user/zhangyixuan/data/sft_processed/sstbale/response_diversification_250514_1x
# 0.004866068 /user/zhangyixuan/data/for_sync/n_gram_dedup_PretrainGuwenData_nltk_10_test1_2_simplify
# 0.000064161 /user/zhangyixuan/data/sft_processed/sstbale/math500_trainset_qwen3_32B_gen_rm_emoji_1x
# 0.000036061 /user/zhangyixuan/data/sft_processed/sstbale/math_sft_qwen3_32b_wo_think_correct_answer
# 0.000035223 /user/zhangyixuan/data/sft_processed/sstbale/math_sft_qwen3_32b_w_think_correct_answer
# 0.000031484 /user/zhangyixuan/data/sft_processed/sstbale/gsm8k_qwen3_30B_gen_rm_emoji_1x
# 0.000087459 /user/zhangyixuan/data/sft_processed/sstbale/survey_250514
# 0.001275544 /user/zhangyixuan/data/sft_processed/sstbale/nemotron_sft_science
# 0.000025499 /user/zhangyixuan/data/sft_processed/sstbale/qwen_sft_checked_code
# 0.000548640 /user/zhangyixuan/data/sft_processed/sstbale/qwen_sft_code
# 0.000240552 /user/zhangyixuan/data/sft_processed/sstbale/qwen_sft_math
# 0.000060095 /user/zhangyixuan/data/sft_processed/sstbale/qwen_sft_fries
# 0.000000209 /user/zhangyixuan/data/sft_processed/sstbale/qwen_sft_misc_tasks
# 0.000005409 /user/zhangyixuan/data/sft_processed/sstbale/qwen_sft_if
# 0.000582206 /user/zhangyixuan/data/sft_processed/sstbale/ifeval_like"


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
    --seq-length 4096
    --max-position-embeddings 4096
    --num-layers 10
    --hidden-size 768
    --ffn-hidden-size 3072
    --num-attention-heads 12
    --group-query-attention
    --num-query-groups 1
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
    --rotary-base 10000
    --norm-epsilon 1e-6
    --elementwise-attn-output-gate
)

MOE_ARGS=(
    --num-experts 160
    --moe-router-topk 16
    --moe-ffn-hidden-size 192
    --moe-shared-expert-intermediate-size 192
    # --moe-shared-expert-overlap
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-4
    --moe-grouped-gemm
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend hybridep
    --moe-permute-fusion
    --moe-layer-freq [0]+[1]*9
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0.001
    --moe-router-pre-softmax
    --moe-router-topk-scaling-factor 3.66
    --moe-router-dtype fp32
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model /user/yanhui/utils/tokenizer_v9.6/
    # --data-path $DATA_PATH
    --split 99990,8,2
    --use-modelbest-sdk
    # --no-load-data-state
    --dataloader-type external
)

TRAINING_ARGS=(
    # --micro-batch-size 16
    # --global-batch-size 128
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
    --muon-split-qkv 
    --muon-coefficient-type quintic
    --adam-beta1 0.9
    --adam-beta2 0.95
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1 \
    # --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    # --log-throughput \
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH \
    # --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --log-task-loss-interval 100 \
    --async-save \
    --ckpt-format torch_dist \
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
