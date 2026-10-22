"""
Optimized SQL queries with window functions for inference and latency statistics.
"""

# Query 1: Latency statistics with moving averages and cumulative percentiles using window functions
LATENCY_ANALYTICS_QUERY = """
SELECT 
    model_id,
    inference_timestamp,
    latency_ms,
    AVG(latency_ms) OVER (
        PARTITION BY model_id 
        ORDER BY inference_timestamp 
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) AS moving_avg_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) OVER (
        PARTITION BY model_id
    ) AS p95_latency_ms,
    RANK() OVER (
        PARTITION BY model_id 
        ORDER BY latency_ms DESC
    ) as latency_rank
FROM model_inferences
WHERE inference_timestamp >= NOW() - INTERVAL '7 days';
"""

# Query 2: Inference throughput and success/error rate trends using window aggregations
INFERENCE_STATS_QUERY = """
SELECT 
    DATE_TRUNC('hour', inference_timestamp) AS inference_hour,
    model_id,
    status,
    COUNT(*) AS total_inferences,
    SUM(COUNT(*)) OVER (
        PARTITION BY model_id, DATE_TRUNC('day', inference_timestamp)
    ) AS daily_model_total,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY model_id, DATE_TRUNC('hour', inference_timestamp)), 
        2
    ) AS status_percentage
FROM model_inferences
GROUP BY DATE_TRUNC('hour', inference_timestamp), model_id, status
ORDER BY inference_hour DESC, model_id;
"""
