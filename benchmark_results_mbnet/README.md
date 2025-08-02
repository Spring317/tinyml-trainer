# ONNX Runtime Benchmark Results

## Plotting Instructions

### Using Python + Matplotlib

```python
import pandas as pd
import matplotlib.pyplot as plt

# 1. Plot raw inference times
df = pd.read_csv('inference_times.csv')
plt.figure(figsize=(12, 6))
plt.plot(df['Iteration'], df['InferenceTime_ms'])
plt.xlabel('Iteration')
plt.ylabel('Inference Time (ms)')
plt.title('Inference Time per Iteration')
plt.grid(True)
plt.savefig('inference_times.png', dpi=300)

# 2. Plot histogram
hist_df = pd.read_csv('histogram.csv')
centers = (hist_df['BinStart_ms'] + hist_df['BinEnd_ms']) / 2
plt.figure(figsize=(10, 6))
plt.bar(centers, hist_df['Count'], width=(hist_df['BinEnd_ms'] - hist_df['BinStart_ms']).iloc[0])
plt.xlabel('Inference Time (ms)')
plt.ylabel('Frequency')
plt.title('Distribution of Inference Times')
plt.grid(True)
plt.savefig('histogram.png', dpi=300)

# 3. Plot percentiles
perc_df = pd.read_csv('percentiles.csv')
plt.figure(figsize=(10, 6))
plt.plot(perc_df['Percentile'], perc_df['Time_ms'])
plt.xlabel('Percentile')
plt.ylabel('Inference Time (ms)')
plt.title('Inference Time Percentiles')
plt.grid(True)
plt.savefig('percentiles.png', dpi=300)

# 4. Plot rolling average
rolling_df = pd.read_csv('rolling_average.csv')
plt.figure(figsize=(12, 6))
plt.plot(rolling_df['Iteration'], rolling_df['RollingAvg_ms'])
plt.xlabel('Iteration')
plt.ylabel('50-iteration Rolling Average (ms)')
plt.title('Inference Time Rolling Average')
plt.grid(True)
plt.savefig('rolling_average.png', dpi=300)

# 5. Plot cache impact comparison
cache_df = pd.read_csv('cache_impact.csv')
plt.figure(figsize=(8, 6))
plt.bar(cache_df['Scenario'], cache_df['Time_ms'])
plt.ylabel('Inference Time (ms)')
plt.title('Cache Impact on Inference Time')
plt.grid(True)
plt.savefig('cache_impact.png', dpi=300)
```

## Summary Statistics

See `summary.csv` for key performance metrics.
