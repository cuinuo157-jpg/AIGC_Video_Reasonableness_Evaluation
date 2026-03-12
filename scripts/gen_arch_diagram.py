"""
多模态生成精度异常判定 - 右半部分架构图生成
输出 PNG 图片，可直接插入 PPT
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# ── 中文字体 ──
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 颜色（简约蓝灰色系） ──
C_BG = "#FAFBFC"
C_PRIMARY = "#2C3E50"       # 深蓝灰（主色）
C_PRIMARY_L = "#5D6D7E"     # 中蓝灰（边框/次级）
C_PRIMARY_XL = "#EBF0F4"    # 浅蓝灰（卡片底色）
C_WHITE = "#FFFFFF"
C_DARK = "#2C3E50"
C_GRAY = "#7F8C8D"
C_L1 = "#34495E"            # 三级检测统一深色
C_L2 = "#5D6D7E"            # 三级检测统一中色
C_L3 = "#7F8C8D"            # 三级检测统一浅色
C_ADV_BG = "#EBF0F4"        # 优势卡片统一底色
C_ADV_TC = "#2C3E50"        # 优势卡片统一标题色
C_ARROW = "#5D6D7E"


def draw_box(ax, x, y, w, h, text, facecolor, edgecolor="none",
             fontsize=10, fontcolor="white", bold=True, sub_lines=None,
             sub_fontsize=7.5, sub_color=None, linewidth=1.5):
    """绘制圆角矩形节点，紧凑排列"""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor if edgecolor != "none" else facecolor,
        linewidth=linewidth,
        zorder=2,
    )
    ax.add_patch(box)

    weight = "bold" if bold else "normal"
    if not sub_lines:
        # 无子行 → 标题居中
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=fontcolor, fontweight=weight, zorder=3)
    else:
        # 有子行 → 标题靠上，子行紧跟
        n_sub = len(sub_lines)
        line_gap = 0.28
        total_sub_h = n_sub * line_gap
        # 标题位置
        title_y = y + h - 0.22
        ax.text(x + w / 2, title_y, text, ha="center", va="center",
                fontsize=fontsize, color=fontcolor, fontweight=weight, zorder=3)

        sc = sub_color or C_GRAY
        start_y = title_y - 0.32
        for i, line in enumerate(sub_lines):
            if isinstance(line, tuple):
                line_text, line_color, line_bold = line
            else:
                line_text, line_color, line_bold = line, sc, False
            ly = start_y - i * line_gap
            ax.text(x + w / 2, ly, line_text, ha="center", va="center",
                    fontsize=sub_fontsize, color=line_color,
                    fontweight="bold" if line_bold else "normal", zorder=3)
    return box


def draw_arrow(ax, x, y_start, y_end):
    """绘制向下箭头，zorder 高于所有盒子"""
    ax.annotate(
        "", xy=(x, y_end), xytext=(x, y_start),
        arrowprops=dict(
            arrowstyle="->,head_width=0.25,head_length=0.12",
            color=C_ARROW, lw=2.5,
        ),
        zorder=5,
    )


def draw_adv_card(ax, x, y, w, h, title, desc, bg_color, title_color):
    """绘制优势小卡片"""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06",
        facecolor=bg_color,
        edgecolor="#BDBDBD",
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
            fontsize=8, color=title_color, fontweight="bold", zorder=3,
            linespacing=1.3)
    ax.text(x + w / 2, y + h * 0.22, desc, ha="center", va="center",
            fontsize=6, color=C_GRAY, zorder=3, linespacing=1.3)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(10, 10.5))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-1.2, 11.5)
    ax.axis("off")

    # ── 标题 ──
    ax.text(5, 11.1, "检测系统方案（图像 + 视频双模态）",
            ha="center", va="center", fontsize=15, color=C_PRIMARY,
            fontweight="bold")

    bx = 0.3
    bw = 9.4
    cx = bx + bw / 2
    arrow_gap = 0.5   # 箭头区域高度（盒子间距）

    # ── 1. 输入层 ──
    h1 = 0.5
    y1 = 10.3
    draw_box(ax, bx, y1, bw, h1,
             "输入层：图像 / 视频帧序列",
             facecolor=C_PRIMARY, fontsize=11, fontcolor=C_WHITE)

    draw_arrow(ax, cx, y1 - 0.12, y1 - arrow_gap + 0.12)

    # ── 2. 特征提取层 ──
    h2 = 1.5
    y2 = y1 - arrow_gap - h2
    draw_box(ax, bx, y2, bw, h2,
             "共享特征提取层（Feature Hub）",
             facecolor=C_PRIMARY_XL, edgecolor=C_PRIMARY_L,
             fontsize=10, fontcolor=C_PRIMARY,
             sub_lines=[
                 "关键点检测（MediaPipe Face / Hand / Body）",
                 "人脸嵌入（InsightFace ArcFace 512-d）",
                 "深度估计（MiDaS）  |  光流分析（RAFT，仅视频）",
                 "纹理特征  |  频域分析  |  颜色直方图",
             ],
             sub_fontsize=7.5, sub_color=C_GRAY)

    draw_arrow(ax, cx, y2 - 0.12, y2 - arrow_gap + 0.12)

    # ── 3. 三级检测引擎 ──
    h3 = 1.5
    y3 = y2 - arrow_gap - h3
    draw_box(ax, bx, y3, bw, h3,
             "多级异常检测引擎（三级递进）",
             facecolor=C_WHITE, edgecolor=C_PRIMARY_L,
             fontsize=10, fontcolor=C_PRIMARY, linewidth=2,
             sub_lines=[
                 ("L1 快速规则筛查：几何约束（手指数量/关节角度/骨骼比例）、生理指标（EAR/MAR）", C_L1, True),
                 ("L2 结构化分析：骨架拓扑验证、纹理一致性、时序平滑度（图像+视频）", C_L2, True),
                 ("L3 MLLM 语义推理：大模型视觉理解兜底，覆盖长尾异常与复合型缺陷", C_L3, True),
             ],
             sub_fontsize=7.5)

    draw_arrow(ax, cx, y3 - 0.12, y3 - arrow_gap + 0.12)

    # ── 4. 六维度评估矩阵 ──
    h4 = 1.65
    y4 = y3 - arrow_gap - h4
    draw_box(ax, bx, y4, bw, h4,
             "多维度评估矩阵（可扩展）",
             facecolor=C_PRIMARY_XL, edgecolor=C_PRIMARY_L,
             fontsize=10, fontcolor=C_PRIMARY,
             sub_lines=[
                 "D1 人脸一致性（CSIM 余弦相似度）       D2 表情自然度（AU + FACS 规则）",
                 "D3 生物结构异常（眼/手/口/躯体）       D4 运动逻辑（光流加速度平滑性）",
                 "D5 物理常识（像素漂移+重力一致性）     D6 背景一致性（深度时序+特征匹配）",
                 ("D(n) 更多维度（语义连贯性 / 光照一致性 / 音画同步 / ...）", C_GRAY, False),
             ],
             sub_fontsize=7.5, sub_color=C_DARK)

    draw_arrow(ax, cx, y4 - 0.12, y4 - arrow_gap + 0.12)

    # ── 5. 输出层 ──
    h5 = 0.7
    y5 = y4 - arrow_gap - h5
    draw_box(ax, bx, y5, bw, h5,
             "输出层",
             facecolor=C_PRIMARY, fontsize=10, fontcolor=C_WHITE,
             sub_lines=[
                 "异常类型分类 + 置信度评分  |  异常帧精准定位  |  多维度加权综合评分",
             ],
             sub_fontsize=7.5, sub_color="#BBDEFB")

    # ── 6. 优势卡片 ──
    adv_h = 1.0
    adv_y = y5 - 0.25 - adv_h
    adv_w = 2.9
    adv_gap = 0.35

    advs = [
        ("图像+视频\n双模态覆盖", "统一特征提取\n兼顾静态异常与动态时序异常", C_ADV_BG, C_ADV_TC),
        ("三级递进检测", "规则→结构→语义\n兼顾效率与覆盖率", C_ADV_BG, C_ADV_TC),
        ("MLLM\n兜底长尾", "大模型语义理解\n覆盖规则难以枚举的异常", C_ADV_BG, C_ADV_TC),
    ]
    for i, (title, desc, bg, tc) in enumerate(advs):
        ax_x = bx + i * (adv_w + adv_gap)
        draw_adv_card(ax, ax_x, adv_y, adv_w, adv_h, title, desc, bg, tc)

    # ── 保存 ──
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "arch_diagram_anomaly_detection.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"架构图已生成: {out_path}")


if __name__ == "__main__":
    main()
