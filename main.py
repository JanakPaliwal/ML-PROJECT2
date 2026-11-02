"""
main.py
All-in-One CLI & Execution Entrypoint for Short-Term Mandi Price Forecasting.
Commodities: Onion & Tomato
Mandis: Lasalgaon (MH), Kolar (KA), Azadpur (Delhi), Pimpalgaon (MH)
SDGs: SDG 2 (Zero Hunger) & SDG 12 (Responsible Consumption and Production)

Usage:
  python main.py              # Runs full pipeline and outputs benchmark report
  python main.py --serve      # Runs pipeline and starts web dashboard at http://localhost:8080
"""

import sys
import os
import argparse
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train_pipeline import run_pipeline

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def print_banner():
    banner = """
========================================================================================
   AGRIMANDI AI: SHORT-TERM MANDI PRICE FORECASTING (ONION & TOMATO)
   Aligned with UN SDG 2 (Zero Hunger) & SDG 12 (Responsible Consumption & Production)
   Dataset: Agmarknet (DMI, MoA&FW) via OGD Platform India (GODL-India License)
========================================================================================
"""
    print(banner)

def main():
    parser = argparse.ArgumentParser(description="Short-term mandi price forecasting for Onion & Tomato.")
    parser.add_argument("--serve", action="store_true", help="Launch local HTTP dashboard server after training")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve dashboard on (default: 8080)")
    args = parser.parse_args()

    print_banner()
    start_time = time.time()

    # 1. Execute end-to-end ML pipeline
    print("[1/3] Running Data Ingestion, Outlier Treatment & Feature Engineering...")
    dashboard_payload = run_pipeline()

    elapsed = round(time.time() - start_time, 2)
    print(f"\n[OK] Pipeline execution finished in {elapsed}s.")

    # 2. Display Formatted Benchmark Summary Table in Terminal
    print("\n" + "=" * 95)
    print("                      MODEL PERFORMANCE BENCHMARK LEADERBOARD (180-Day Holdout)")
    print("=" * 95)
    header = f"{'Commodity':<8} | {'Mandi':<12} | {'Model Architecture':<22} | {'RMSE (Rs)':<9} | {'MAE (Rs)':<8} | {'MAPE':<7} | {'R2':<7} | {'Spike Recall'}"
    print(header)
    print("-" * 95)

    summary_items = dashboard_payload.get("benchmark_summary", [])
    for row in summary_items:
        comm = row["Commodity"]
        mandi = row["Mandi"]
        model = row["Model"]
        rmse = f"Rs.{row['Holdout_RMSE']:.1f}"
        mae = f"Rs.{row['Holdout_MAE']:.1f}"
        mape = f"{row['Holdout_MAPE']:.1f}%"
        r2 = f"{row['Holdout_R2']:.3f}"
        rec = f"{row['Spike_Recall']*100:.1f}%"
        print(f"{comm:<8} | {mandi:<12} | {model:<22} | {rmse:<9} | {mae:<8} | {mape:<7} | {r2:<7} | {rec}")

    print("=" * 95)

    # 3. Early Warning Advisories
    print("\n" + "=" * 95)
    print("                      CURRENT 7-DAY AHEAD EARLY-WARNING VOLATILITY SIGNALS")
    print("=" * 95)
    for comm in ["Onion", "Tomato"]:
        comm_dict = dashboard_payload["commodities_data"][comm]
        for mandi, m_info in comm_dict["mandi_details"].items():
            alert = m_info["alert"]
            spot = m_info["latest_price"]
            pred = alert["predicted_price"]
            chg = alert["expected_change_pct"]
            best_m = m_info["best_model"]
            print(f"[{comm.upper()} - {mandi}] Best Model: {best_m}")
            print(f"  Spot Price: Rs.{spot:,.1f}/qtl  -->  7-Day Forecast: Rs.{pred:,.1f}/qtl ({'+' if chg>=0 else ''}{chg:.1f}%)")
            print(f"  Alert Level: {alert['level']}  |  Signal: {alert['recommendation']}")
            print("-" * 95)

    print("\n[OK] Results exported to:")
    print(f"  - Model comparison CSV : {os.path.join(PROJECT_ROOT, 'data', 'processed', 'overall_model_comparison.csv')}")
    print(f"  - Dashboard payload    : {os.path.join(PROJECT_ROOT, 'web', 'forecast_data.json')}")
    print(f"  - Research Paper Draft : {os.path.join(PROJECT_ROOT, 'paper', 'manuscript_short_communication.md')}")

    # 4. Optional Dashboard launch
    if args.serve:
        import webbrowser
        from run_demo import start_server
        port = start_server(args.port)
        url = f"http://localhost:{port}"
        print(f"\n[✓] Web Dashboard running at: {url}")
        print("Opening browser...")
        webbrowser.open(url)
        print("Press Ctrl+C to terminate.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down.")

if __name__ == "__main__":
    main()
