import configs.default as cfg
from src.backtest.engine import BacktestEngine
from src.backtest.report import print_summary, plot_results

if __name__ == "__main__":
    print("Running Options Market Maker Backtest...")
    engine  = BacktestEngine(cfg, seed=42)
    results = engine.run()
    print_summary(results)
    plot_results(results, save_path="backtest_results.png")
