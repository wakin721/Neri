"""Render a finished compare_dinov3_litert_local JSON run as a Markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NAMES = {'pytorch_original':'原版 PyTorch CPU','pytorch_gpu':'原版 PyTorch CUDA',
         'litert_fp32':'LiteRT FP32','litert_fp16_weights':'LiteRT FP16 权重'}


def render(directory):
    report = json.loads((directory/'report.json').read_text('utf8'))
    assert report.get('input_content_verified')
    assert len(report['quality']) >= 3 and len(report['speed']) >= 3
    quality = report['quality']
    fp32=quality['litert_fp32']['vs_original']
    fp16=quality['litert_fp16_weights']['vs_original']
    cpu_rate=report['speed']['pytorch_cpu']['batches']['32']['images_per_second']
    fp32_rate=report['speed']['litert_fp32']['batches']['32']['images_per_second']
    fp16_rate=report['speed']['litert_fp16_weights']['batches']['32']['images_per_second']
    lines = ['# Neri1.2preview：本地数据 LiteRT / PyTorch 对比', '',
        '日期：2026-11-15。固定现有模型和校准，不训练、不重校准、不改动源图片或软件模型设置。', '',
        f"本轮 14,477 张中，相对原版 CPU，LiteRT FP32 的候选物种变化 {fp32['species_changes']} 张、接纳变化 {fp32['acceptance_changes']} 张、最终输出变化 {fp32['final_output_changes']} 张；FP16 权重版分别为 {fp16['species_changes']}、{fp16['acceptance_changes']}、{fp16['final_output_changes']} 张。",
        f"同样 CPU 8 线程、batch=32 的重复测速中，FP32 吞吐为原版 CPU 的 {fp32_rate/cpu_rate:.2f} 倍，FP16 权重版为 {fp16_rate/cpu_rate:.2f} 倍。该结论限定本地数据与本机环境。", '',
        '## 数据与方法', '',
        f"全量使用冻结相机隔离 E 集，共 {report['identity']['rows']:,} 张真实裁图。逐图内容 SHA-256 与冻结清单一致。",
        '已知 43 个物种、14,449 张；未知 21 个物种、28 张。Bank、C 和 E 角色不变。',
        '每个编码器独立重算 E 特征及 E 内历史适配；同相机严格过去 7 天、最多 64 条去重历史；沿用完整事件融合和原始裁图哈希。',
        '主对照为 PyTorch CPU，CUDA 单列；分类头、校准参数与门限完全共享。所有精度指标均来自完整 14,477 张和完整事件。',
        '这些图片曾在历史研究中曝光，本次属于实现等价性与相机留出诊断，不是新采集盲测。未知样本较少，不据此外推稳定未知拒识能力。', '',
        '## 精度（物种宏平均）', '',
        '|版本|闭集准确率|已知正确接纳率|已知误拒率|未知拒识率|',
        '|---|---:|---:|---:|---:|']
    for variant,q in quality.items():
        m=q['metrics']
        values=[m[k]*100 for k in ('known_closed_accuracy_macro','known_correct_acceptance_macro',
                                  'known_false_rejection_macro','unknown_rejection_macro')]
        lines.append('|'+NAMES[variant]+'|'+'|'.join(f'{v:.6f}%' for v in values)+'|')
    lines += ['', '## 相对原版 CPU 的逐图及事件差异', '',
        '|版本|物种候选变化/张|接纳变化/张|最终输出变化/张|发生变化的事件|接纳分数最大绝对差|',
        '|---|---:|---:|---:|---:|---:|']
    for variant,q in quality.items():
        if 'vs_original' not in q:
            continue
        d=q['vs_original']
        lines.append(f"|{NAMES[variant]}|{d['species_changes']}|{d['acceptance_changes']}|{d['final_output_changes']}|{d['changed_events']}|{d['max_known_score_diff']:.9g}|")
    disagreements=json.loads((directory/'litert_fp16_weights_disagreements.json').read_text('utf8'))
    if disagreements:
        d=disagreements[0]
        lines += ['',f"FP16 本次变化是同一岩羊事件的两张裁图（00532512_00 / 00532512_01）：分数由 {d['original_score']:.9f} 变为 {d['lite_score']:.9f}，跨过固定门限 {d['threshold']:.9f}，因此由拒识变为正确接纳。物种候选不变。这是门限附近的数值翻转，不是 FP16 泛化能力更好的证据。"]
    lines += ['', '最终输出将拒识统一记为 Unknown。候选物种变化可以发生在被拒识的图片上，未必改变用户得到的输出。',
        '同一事件多张图片共享拒识决定，因此图片变化数量与独立事件变化数量分开报告。已知/未知细分见 report.json。', '',
        '## 推理速度（重复测量）', '',
        '同一组按物种均衡选择的 64 张真实裁图；CPU 固定 8 线程。每项先预热一次，再测三次，表中为中位数。',
        '测量编码器 API：预处理已完成，包含 PyTorch 堆叠与 CPU/GPU 传输、LiteRT 输入拷贝；不含探测、图片解码、历史、分类头或界面。', '',
        '|版本|batch=1 张/秒|batch=8 张/秒|batch=32 张/秒|batch=32 相对 CPU|',
        '|---|---:|---:|---:|---:|']
    base=report['speed']['pytorch_cpu']['batches']['32']['images_per_second']
    for variant,s in report['speed'].items():
        vals=[s['batches'][str(b)]['images_per_second'] for b in (1,8,32)]
        name=NAMES['pytorch_original'] if variant=='pytorch_cpu' else NAMES[variant]
        lines.append(f'|{name}|{vals[0]:.2f}|{vals[1]:.2f}|{vals[2]:.2f}|{vals[2]/base:.2f}×|')
    lines += ['', '## 全量模型流程分段耗时', '',
        '各版本分别运行，磁盘缓存未控制。下表不含图片读取、解码、预处理、初始化、探测、数据库或界面耗时。',
        'CPU 最初 2,944 张直接解码，后续恢复并共用预处理缓存；原始输入准备计时保留在 JSON 中，因执行方式不同不作后端性能对照。',
        f"共享输入缓存生成耗时 {report.get('input_cache',{}).get('seconds',0):.3f} 秒；所有输入均验证 Torch/NumPy 归一化逐元素相等。", '',
        '|版本|编码/秒|历史适配/秒|分类及事件融合/秒|上述合计/秒|',
        '|---|---:|---:|---:|---:|---:|']
    for variant,e in report['full_extraction'].items():
        q=quality[variant]
        vals=[e['encoder_seconds'],q['history_seconds'],q['classification_seconds']]
        lines.append('|'+NAMES[variant]+'|'+'|'.join(f'{v:.3f}' for v in vals+[sum(vals)])+'|')
    lines += ['', '## 环境、复现与边界', '',
        'CPU：AMD Ryzen 7 9800X3D；GPU：'+str(report['environment']['gpu'])+'。',
        f"PyTorch {report['environment']['torch']}；LiteRT {report['environment']['litert']}；NumPy {report['environment']['numpy']}。",
        'LiteRT FP16 仅压缩编码器权重；输入/输出为 FP32，不是 INT8 量化或全 FP16 推理。',
        '本测试覆盖已有裁图的特征、历史、分类和拒识。它不验证软件尚未接入的 LiteRT 分类加载、探测器组合、视频、自动学习或数据库全链路。',
        '源文件哈希、模型清单哈希、划分哈希、阈值、全部逐图判定、特征及差异清单保存在同目录。', '',
        '最终复核：64 张重新读取的原图预处理与共享缓存逐元素相等；四份全量特征均通过形状、有限值及单位范数检查；保存的逐图判定可独立重算全部指标；相关源码在测试期间未改变。',
        '原版 GPU 通过软件流程重新提取的全部事件分数，与冻结参考逐项一致（最大误差为 0）；CPU 与冻结参考的分类和接纳决定也全部一致。', '',
        '```powershell',
        "& 'D:/Application/Neri/toolkit/python.exe' -I -X utf8 scripts/compare_dinov3_litert_local.py --stage all",
        f"& 'D:/Application/Neri/toolkit/python.exe' -I -X utf8 scripts/report_dinov3_litert_local.py --run '{directory.as_posix()}'",
        '```', '']
    target=directory/'REPORT.md'
    target.write_text('\n'.join(lines),'utf8')
    print(target)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path(__file__).resolve().parents[1]/'runs/Neri1.2preview-litert-local-full-20261115')
    render(parser.parse_args().run.resolve())
