-- This Hive SQL creates windowed running totals of a metric across some grouping key. This also handles the issue of missing values for some dates in the time period
-- Translated Problem: Created the running total of total_trips_2D across all mega_regions
WITH 
dates AS -- This can be scaled from a dates table
(
    SELECT '2021-01-01' AS datestr
    UNION ALL
    SELECT '2021-01-02' AS datestr
    UNION ALL 
    SELECT '2021-01-03' AS datestr
),
metric_trend AS -- This is the actual metric data
(
    SELECT '2021-01-01' AS datestr, 'a' AS mega_region, 10 AS trips
    UNION ALL 
    SELECT '2021-01-02' AS datestr, 'a' AS mega_region, 50 AS trips
    UNION ALL 
    SELECT '2021-01-01' AS datestr, 'b' AS mega_region, 20 AS trips
),
dates_mega_region_cartesian AS -- Creates cartesian on dates x mega_regions to create placeholders for mega_regions which do not have any values for a few dates
(
    SELECT
        DISTINCT
            dates.datestr, 
            metric_trend.mega_region
    FROM dates, metric_trend
),
norm_database AS -- Corrects the missing values and adds 0 for the dates when a mega region is not present in the data
(
    SELECT 
        dates_mega_region_cartesian.datestr,
        dates_mega_region_cartesian.mega_region,
        COALESCE(metric_trend.trips, 0) AS trips
    FROM dates_mega_region_cartesian
    LEFT JOIN metric_trend
    ON dates_mega_region_cartesian.datestr = metric_trend.datestr
        AND dates_mega_region_cartesian.mega_region = metric_trend.mega_region
)

-- Standard running total using WINDOW and ROWS BETWEEN
SELECT
    mega_region,
    datestr,
    trips,
    SUM(trips) OVER(
        PARTITION BY mega_region 
        ORDER BY datestr
        ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
    ) AS running_sum_2D
FROM norm_database
ORDER BY mega_region, datestr