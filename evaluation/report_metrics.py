"""
evaluation/report_metrics.py
-----------------------------
Tổng hợp bảng `ai_tutor_metrics` (được ghi tự động qua `database.track_metric`
/ `database.record_metric`) thành báo cáo:
  - Nhóm 2 (hiệu năng & độ tin cậy): latency trung bình/p50/p95 và tỷ lệ lỗi
    theo từng operation (`hybrid_search`, `index_chunks`, `ask_agent`,
    `cohere_rerank`...).
  - Nhóm 3 (chi phí/lượt gọi): số lượt gọi các dịch vụ tính phí/giới hạn
    quota (Gemini qua `ask_agent`, Cohere qua `cohere_rerank`), đối chiếu với
    hạn mức bạn tự khai báo qua biến môi trường.

Chạy: python -m evaluation.report_metrics [--days 7]

KHÔNG hard-code hạn mức free-tier trong script này vì các nhà cung cấp
(Google, Cohere) thường xuyên thay đổi quota — hãy tự kiểm tra hạn mức hiện
tại của bạn (tại ai.google.dev/gemini-api/docs/rate-limits và
dashboard.cohere.com) rồi khai báo qua biến môi trường:

    GEMINI_DAILY_QUOTA=1000   # số request/ngày cho model bạn đang dùng
    COHERE_DAILY_QUOTA=33     # ước tính = hạn mức tháng / 30

Không đặt các biến trên vẫn chạy được — script chỉ bỏ qua phần so sánh %.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Cho phép chạy trực tiếp `python evaluation/report_metrics.py` từ thư mục gốc
# lẫn `python -m evaluation.report_metrics`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import fetch_recent_metrics  # noqa: E402

# Operation nào tương ứng với lượt gọi API bên ngoài có tính quota (khác với
# hybrid_search/index_chunks — chạy embedding LOCAL bằng bge-m3, miễn phí).
QUOTA_ENV_MAP = {
    "ask_agent": "GEMINI_DAILY_QUOTA",
    "cohere_rerank": "COHERE_DAILY_QUOTA",
}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(int(len(values) * pct), len(values) - 1)
    return values[idx]


def print_performance_report(rows: list[tuple[str, float, bool]]) -> None:
    """Nhóm 2: hiệu năng & độ tin cậy — latency + tỷ lệ lỗi theo operation."""
    by_op: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for operation, duration, success in rows:
        by_op[operation].append((duration, success))

    print("=" * 78)
    print("NHÓM 2 — HIỆU NĂNG & ĐỘ TIN CẬY")
    print("=" * 78)
    if not by_op:
        print("(Chưa có dữ liệu metric nào trong khoảng thời gian này.)")
        return

    header = (
        f"{'Operation':<18}{'Số lượt':>9}{'Tỷ lệ lỗi':>12}"
        f"{'Avg (s)':>10}{'p50 (s)':>10}{'p95 (s)':>10}"
    )
    print(header)
    print("-" * len(header))
    for operation, entries in sorted(by_op.items()):
        durations = [d for d, _ in entries]
        error_rate = 1 - (sum(1 for _, ok in entries if ok) / len(entries))
        print(
            f"{operation:<18}{len(entries):>9}{error_rate:>11.1%} "
            f"{statistics.mean(durations):>9.2f} "
            f"{_percentile(durations, 0.5):>9.2f} "
            f"{_percentile(durations, 0.95):>9.2f}"
        )
        if error_rate >= 0.1:
            print(f"  ⚠️  Tỷ lệ lỗi '{operation}' khá cao ({error_rate:.0%}) — đáng kiểm tra log chi tiết.")


def print_cost_report(rows: list[tuple[str, float, bool]], days: int) -> None:
    """Nhóm 3: chi phí/lượt gọi — đối chiếu quota nếu người dùng có khai báo."""
    counts: dict[str, int] = defaultdict(int)
    for operation, _duration, _success in rows:
        counts[operation] += 1

    print()
    print("=" * 78)
    print("NHÓM 3 — CHI PHÍ / LƯỢT GỌI (ước tính quota)")
    print("=" * 78)
    print(
        f"(Tổng hợp {days} ngày gần nhất. LƯU Ý: 1 lượt 'ask_agent' có thể tương ứng "
        f"NHIỀU HƠN 1 lệnh gọi Gemini thật, vì vòng lặp ReAct có thể gọi lại LLM "
        f"nhiều bước cho 1 câu hỏi — số liệu dưới đây là CẬN DƯỚI, không phải số "
        f"lệnh gọi API tuyệt đối.)\n"
    )

    for operation, quota_env in QUOTA_ENV_MAP.items():
        count = counts.get(operation, 0)
        avg_per_day = count / days if days else 0
        line = f"  {operation:<16}{count:>6} lượt  (~{avg_per_day:.1f}/ngày)"

        quota_raw = os.getenv(quota_env)
        if quota_raw:
            try:
                quota_val = float(quota_raw)
                pct = (avg_per_day / quota_val * 100) if quota_val else 0
                line += f"  ~ {pct:.0f}% của quota {quota_val:.0f}/ngày ({quota_env})"
                if pct >= 80:
                    line += "  ⚠️ GẦN CHẠM QUOTA"
            except ValueError:
                line += f"  (giá trị {quota_env}='{quota_raw}' không hợp lệ, bỏ qua)"
        else:
            line += f"  (đặt biến {quota_env} để so sánh với hạn mức free-tier của bạn)"
        print(line)

    other_ops = set(counts) - set(QUOTA_ENV_MAP)
    if other_ops:
        print("\n  Operation khác (chạy LOCAL, không tốn quota API bên ngoài):")
        for op in sorted(other_ops):
            print(f"    {op:<16}{counts[op]:>6} lượt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Báo cáo hiệu năng (nhóm 2) & chi phí/lượt gọi (nhóm 3) từ ai_tutor_metrics."
    )
    parser.add_argument("--days", type=int, default=7, help="Số ngày gần nhất để phân tích (mặc định: 7)")
    args = parser.parse_args()

    rows = fetch_recent_metrics(args.days)
    print_performance_report(rows)
    print_cost_report(rows, args.days)


if __name__ == "__main__":
    main()
