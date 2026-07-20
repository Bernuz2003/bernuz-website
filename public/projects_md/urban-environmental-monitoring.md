# Urban Environmental Monitoring

**Polyglot data architecture for city-scale environmental monitoring.**

This project simulates an end-to-end urban monitoring platform that combines time-series telemetry, spatial context, threshold-based alerting, live ingestion, historical aggregation, and an interactive Grafana dashboard. The goal is not only to visualize environmental data, but to support operational monitoring, prioritization of critical zones, and strategic urban analysis.

**Main technologies:** InfluxDB · PostgreSQL/PostGIS · Apache Kafka · Telegraf · Grafana · Docker · Flux · SQL

**Project type:** Academic case study · Data Engineering · Data Architecture · Time-Series Monitoring · Dashboarding

---

## 1. Overview

The project models a smart-city environmental monitoring scenario. The system collects and processes telemetry related to air quality, noise, weather, and traffic conditions across urban grid zones.

The dashboard is designed as a **metric-centric decision-support tool**. The user can select a specific environmental metric, such as `pm25`, `pm10`, `no2`, `o3`, or `noise_db`, and then filter the analysis by borough, zone type, and time range.

The system supports both:

- **operational monitoring**, through recent measurements, alerts, critical zones, and spatial risk maps;
- **analytical exploration**, through trends, distributions, zone-type comparisons, and green infrastructure analysis;
- **data reliability checks**, through dataset availability, bucket-level records, thresholds, and data freshness indicators.

The core architectural idea is **polyglot persistence**: different types of data are stored in different databases according to their access patterns and modeling requirements.

InfluxDB is used for high-volume time-series data and derived alert events, while PostgreSQL/PostGIS stores stable contextual and spatial information, such as grid metadata, sensor metadata, geographic attributes, and threshold profiles.

---

## 2. Problem

Urban environmental monitoring involves heterogeneous data sources and different access patterns.

The system needs to answer questions such as:

- Which areas are currently critical for a selected environmental metric?
- How are warnings and critical alerts distributed across boroughs?
- Which zones should be prioritized for operational intervention?
- How has a metric evolved over the last few days or months?
- Are certain urban zone types more exposed to environmental pressure?
- Is there a relationship between green-space availability and environmental conditions?
- Are the underlying data fresh, complete, and reliable enough to support the dashboard?

A single database model is not ideal for all these requirements.

Time-series measurements require efficient writes, retention policies, temporal aggregation, and time-window queries. Spatial and contextual data require relational modeling, stable attributes, and geospatial support. Alert generation requires combining dynamic measurements with static threshold definitions.

For this reason, the project separates responsibilities across multiple components rather than forcing all data into one storage model.

---

## 3. Architecture

The architecture follows a layered data pipeline.

```text
Synthetic Data Generator
        |
        | historical dataset / live feed
        v
Apache Kafka  --->  Telegraf  --->  InfluxDB urban_raw
                                      |
                                      | InfluxDB Task: raw_to_hourly
                                      v
                               InfluxDB urban_hourly
                                      |
                         +------------+-------------+
                         |                          |
                         | hourly_to_alerts          | hourly_to_daily
                         v                          v
                    InfluxDB urban_alerts     InfluxDB urban_daily

PostgreSQL/PostGIS
        |
        | grid context, sensor context, thresholds, spatial metadata
        v
Grafana dashboard  <---  InfluxDB + PostgreSQL/PostGIS
```

The main components are:

| Component | Role |
|---|---|
| **Synthetic data generator** | Produces historical telemetry and live records for the urban monitoring scenario. |
| **Kafka** | Simulates a streaming ingestion layer and decouples producers from consumers. |
| **Telegraf** | Consumes live records and writes them into InfluxDB. |
| **InfluxDB** | Stores raw, hourly, daily, and alert time-series data. |
| **PostgreSQL/PostGIS** | Stores static urban context, spatial attributes, sensor metadata, and threshold profiles. |
| **InfluxDB Tasks** | Materialize hourly aggregates, daily aggregates, and derived alert events. |
| **Grafana** | Provides the interactive dashboard and decision-support interface. |

The architecture separates **observed telemetry**, **static context**, **derived aggregates**, and **derived alert events**. This makes the system easier to reason about and keeps each technology aligned with the type of data it handles best.

---

## 4. Data model

The data model is split between PostgreSQL/PostGIS and InfluxDB.

### PostgreSQL/PostGIS

PostgreSQL/PostGIS stores stable contextual information. This data changes slowly and is naturally relational or spatial.

#### `grid_context`

Represents urban grid zones.

| Field | Description |
|---|---|
| `grid_id` | Unique identifier of the urban grid zone. |
| `cell_code` | Human-readable grid cell code. |
| `display_name` | Display name used in dashboards or reports. |
| `city` | City name. |
| `borough` | Borough or administrative area. |
| `zone_type` | Urban zone type, such as residential, commercial, industrial, park, or mixed. |
| `center_lat` | Latitude of the zone centroid. |
| `center_lon` | Longitude of the zone centroid. |
| `green_space_pct` | Percentage of green space in the zone. |
| `estimated_population` | Estimated population associated with the zone. |
| `cell_area_km2` | Approximate area of the zone. |
| `geometry` / spatial fields | Geospatial representation of the zone, managed through PostGIS. |

#### `sensor_context`

Represents the sensor network.

| Field | Description |
|---|---|
| `sensor_id` | Unique identifier of the sensor. |
| `grid_id` | Grid zone where the sensor is located. |
| `sensor_lat` | Sensor latitude. |
| `sensor_lon` | Sensor longitude. |
| `sensor_type` | Type of sensor. |
| `active` | Whether the sensor is active. |

#### `threshold_profile`

Stores the operational thresholds used to derive warning and critical alerts.

| Field | Description |
|---|---|
| `metric` | Environmental metric, such as `pm25`, `pm10`, `no2`, `o3`, or `noise_db`. |
| `unit` | Unit of measurement. |
| `warning_threshold` | Warning-level threshold. |
| `critical_threshold` | Critical-level threshold. |
| `description` | Description of the threshold meaning. |
| `source` | Source or rationale for the threshold. |

### InfluxDB

InfluxDB stores time-dependent data. The main measurement is `urban_telemetry` for telemetry and aggregates, while `urban_alerts` is used for derived alert events.

#### Bucket: `urban_raw`

Stores raw telemetry records.

**Measurement:** `urban_telemetry`

Expected tags:

- `city`
- `grid_id`
- `cell_code`
- `borough`
- `zone_type`
- `sensor_id`
- `sensor_type`

Expected fields:

- `pm25`
- `pm10`
- `no2`
- `o3`
- `co`
- `noise_db`
- `traffic_density`
- `temperature_c`
- `humidity_pct`
- `wind_speed_ms`

This layer preserves the highest granularity and is useful for recent telemetry, debugging, live ingestion checks, and raw data freshness.

#### Bucket: `urban_hourly`

Stores hourly aggregates derived from `urban_raw`.

**Measurement:** `urban_telemetry`

Expected tags:

- `city`
- `grid_id`
- `cell_code`
- `borough`
- `zone_type`

Expected fields:

- `pm25`
- `pm10`
- `no2`
- `o3`
- `co`
- `noise_db`
- `traffic_density`
- `temperature_c`
- `humidity_pct`
- `wind_speed_ms`

This layer is the main operational layer for dashboard trends, maps, zone comparisons, and alert generation.

#### Bucket: `urban_daily`

Stores daily aggregates derived from `urban_hourly`.

**Measurement:** `urban_telemetry`

Expected tags:

- `city`
- `grid_id`
- `cell_code`
- `borough`
- `zone_type`

Expected fields:

- `pm25`
- `pm10`
- `no2`
- `o3`
- `co`
- `noise_db`
- `traffic_density`
- `temperature_c`
- `humidity_pct`
- `wind_speed_ms`

This layer supports long-term historical analysis and trend visualization.

#### Bucket: `urban_alerts`

Stores derived alert events generated from hourly aggregates and threshold profiles.

**Measurement:** `urban_alerts`

Expected tags:

- `city`
- `grid_id`
- `cell_code`
- `borough`
- `zone_type`
- `metric`

Expected fields:

- `observed_value`
- `threshold_value`
- `severity_level`

The `severity_level` field identifies the alert type:

- `1` = warning
- `2` = critical

An alert represents a specific metric, in a specific zone, during a specific hourly window, exceeding an operational threshold.

---

## 5. Ingestion workflow

The ingestion workflow supports both historical data loading and live data ingestion.

### Historical loading

Historical line protocol files are generated externally and loaded into InfluxDB. These populate the raw, hourly, and daily layers depending on the dataset generation and loading procedure.

After historical loading, alert backfilling can be executed to derive historical alerts from `urban_hourly` using the threshold profile stored in PostgreSQL.

The historical workflow is:

```text
Generate historical dataset
        |
Install dataset in the runtime repository
        |
Load data into InfluxDB
        |
Run alert backfill from hourly aggregates
        |
Visualize historical data and alerts in Grafana
```

### Live ingestion

Live ingestion is simulated through Kafka and Telegraf.

The live workflow is:

```text
Live data producer
        |
        v
Kafka topic
        |
        v
Telegraf consumer
        |
        v
InfluxDB urban_raw
```

From there, InfluxDB Tasks materialize derived layers.

### InfluxDB Tasks

The project uses InfluxDB Tasks to transform and derive data.

#### `raw_to_hourly`

Aggregates raw telemetry into hourly averages.

Conceptually:

```text
urban_raw  --->  urban_hourly
```

The task processes closed hourly windows and writes one hourly point per metric and zone. The timestamp of the hourly point corresponds to the hourly window, not to the task execution time.

#### `hourly_to_alerts`

Generates warning and critical alerts from hourly data.

Conceptually:

```text
urban_hourly + threshold_profile  --->  urban_alerts
```

This task joins hourly measurements with the corresponding threshold profile. If the observed value exceeds the warning threshold, an alert is generated. If it exceeds the critical threshold, the alert severity is set to critical.

A key design property is **idempotence**. Even if the task is executed frequently, the alert is written using the timestamp of the hourly measurement window, not `now()`. Therefore, if the same alert is generated again for the same metric, zone, and hour, InfluxDB updates the same logical point instead of appending duplicate events.

The logical identity of an alert is approximately:

```text
measurement + tag set + field key + timestamp
```

For this reason, repeated task executions do not inflate alert counts as long as the timestamp and tag set remain stable.

#### `hourly_to_daily`

Aggregates hourly telemetry into daily values.

Conceptually:

```text
urban_hourly  --->  urban_daily
```

This task supports long-term historical charts and reduces the cost of querying long time windows.

---

## 6. Dashboard

The Grafana dashboard is organized as a metric-centric decision-support interface. The user selects a metric, borough, zone type, and time range, and the panels update accordingly.

### Global filters

The dashboard exposes three main variables:

| Variable | Description |
|---|---|
| `metric` | Selected environmental metric, such as `pm25`, `pm10`, `no2`, `o3`, or `noise_db`. |
| `borough` | One or more selected boroughs. |
| `zone_type` | One or more selected urban zone types. |

These filters allow the same dashboard to be reused for different environmental dimensions and spatial scopes.

### City Command Center

The first section provides an immediate operational overview.

Main panels:

- **Critical Zones**: number of distinct grid zones with at least one critical alert for the selected metric.
- **Criticities**: number of critical alert events.
- **Warnings**: number of warning alert events.
- **Raw Data Age**: freshness indicator for the latest raw telemetry point.
- **Risk Geomap**: spatial distribution of alert intensity and severity.
- **Top Critical Zones**: top zones ranked by critical alert count.
- **Top Warning Zones**: top zones ranked by warning alert count.
- **Latest Telemetry Records**: latest raw telemetry records received for the selected metric.

This section answers the operational question:

> What is happening now, where is it happening, and which zones should be prioritized?

### Alert Operations

This section analyzes alert distribution and recent alert events.

Main panels:

- **Alerts by Borough and Severity**: stacked comparison of warning and critical alerts across boroughs.
- **Latest Alerts**: table of recent alerts with observed value, threshold, exceedance percentage, severity, zone, borough, and zone type.

This section helps reconstruct what happened and where, moving from spatial prioritization to event-level analysis.

### Environmental Trends

This section focuses on the continuous behavior of the selected metric.

Main panels:

- **Trend by Borough**: hourly trend by borough over a recent time window.
- **Long-term Daily Trend**: daily trend over a longer historical window.
- **Average by Zone Type**: average metric value grouped by urban zone type.
- **Distribution over Time**: heatmap showing temporal distribution, variability, and outliers.

This section separates alert events from the underlying metric behavior. It helps determine whether a problem is isolated, persistent, localized, or part of a broader temporal pattern.

### Urban Resilience & Green Infrastructure

This section connects environmental measurements with static urban context.

Main panels:

- **Green Infrastructure vs Metric Map**: spatial view combining green-space concentration with warning and critical pressure markers.
- **Green Space vs Metric Scatter Plot**: relationship between green-space percentage and average metric value.
- **Green Band Performance**: grouped comparison of low, medium, and high green-space zones.

The goal is not to prove causality, but to provide descriptive evidence that can support urban planning decisions. For example, the dashboard can help identify whether low-green areas tend to experience higher environmental pressure for selected metrics.

### Data Quality & Dataset Info

This final section provides technical information about the data shown in the dashboard.

Main panels:

- **Threshold Profile**: warning and critical thresholds for the selected metric.
- **Records by Bucket**: number of records available in the selected time range for each data layer.
- **Data Availability by Layer**: latest available timestamp for raw, hourly, daily, and alert data.

This section increases trust in the dashboard by making data freshness, coverage, and threshold definitions visible.

---

## 7. Design decisions

### Why polyglot persistence?

The project uses different data stores for different types of data.

InfluxDB and PostgreSQL/PostGIS solve different problems:

- InfluxDB is optimized for time-series ingestion, retention policies, time-window queries, and aggregation tasks.
- PostgreSQL/PostGIS is better suited for stable relational data, threshold profiles, spatial metadata, and contextual attributes.

This separation avoids forcing spatial and contextual data into a time-series model and avoids forcing high-volume telemetry into a relational schema.

### Why InfluxDB for telemetry?

InfluxDB is used because the core telemetry is time-dependent and naturally queried by time range, metric, and tags.

It supports:

- efficient time-series writes;
- retention policies for different data granularities;
- Flux queries for time-window analysis;
- task-based materialization of aggregates and alerts;
- efficient grouping by tags such as `grid_id`, `borough`, and `zone_type`.

### Why PostgreSQL/PostGIS for context?

Urban context is relatively stable and relational.

PostgreSQL/PostGIS stores:

- grid metadata;
- sensor metadata;
- green-space percentage;
- estimated population;
- threshold profiles;
- spatial attributes and geometries.

This avoids duplicating all static information inside the time-series database and provides a reliable master source for contextual information.

### Why denormalize some context into InfluxDB tags?

Some contextual attributes, such as `grid_id`, `borough`, and `zone_type`, are also stored as InfluxDB tags.

This is a controlled denormalization. These attributes are not measurements, but analytical dimensions used for filtering and grouping.

The advantage is that Grafana queries remain simple and efficient:

```text
filter by metric
filter by borough
filter by zone type
group by grid_id
```

Without these tags, most dashboard panels would require frequent joins with PostgreSQL, making the dashboard more complex and less responsive.

The trade-off is some redundancy and higher series cardinality. In this project, the number of zones and sensors is limited, so the cardinality remains manageable.

### Why Kafka and Telegraf?

Kafka simulates a realistic streaming layer. It decouples the live producer from the storage layer and makes the ingestion pipeline closer to a real-world architecture.

Telegraf acts as a lightweight ingestion agent that consumes live messages and writes them into InfluxDB.

This combination makes the live ingestion path modular and extensible.

### Why Grafana?

Grafana is used because the final goal is not only data storage, but decision support.

The dashboard combines:

- operational KPIs;
- spatial risk maps;
- alert tables;
- temporal trends;
- distribution heatmaps;
- green infrastructure analysis;
- data quality panels.

Grafana also supports variables, mixed data sources, maps, tables, time-series charts, and interactive filtering.

### Why derived alerts instead of precomputed alerts in the dataset?

Alerts are derived inside the system rather than being hardcoded in the generated dataset.

This is important because alert generation depends on threshold profiles stored in PostgreSQL. If thresholds change, alerts can be recomputed without modifying the original telemetry data.

This design separates:

- observed data;
- static threshold definitions;
- derived alert events.

### Why multiple temporal layers?

The system uses different buckets for different granularities:

| Layer | Purpose |
|---|---|
| `urban_raw` | Recent raw telemetry and live ingestion. |
| `urban_hourly` | Operational analytics and alert generation. |
| `urban_daily` | Long-term historical trends. |
| `urban_alerts` | Derived warning and critical events. |

This improves query performance and keeps each dashboard panel aligned with the appropriate data granularity.

---

## 8. Data quality

The dashboard includes a dedicated data quality section because visualizations are only useful if the underlying data is trustworthy.

The project tracks data quality through three main dimensions.

### Data freshness

The dashboard shows the latest available point for each data layer:

- raw telemetry;
- hourly aggregates;
- daily aggregates;
- derived alerts.

This makes it possible to understand whether the system is receiving and processing data correctly.

A key distinction is that old alert data does not always indicate a pipeline issue. It may simply mean that no recent measurement exceeded a threshold.

### Data availability by bucket

The dashboard shows how many records are available in each layer for the selected metric and filters.

This helps explain whether the current view is based on enough data and whether each layer of the pipeline is populated as expected.

For example:

- `urban_raw` should contain recent raw measurements;
- `urban_hourly` should contain aggregated operational data;
- `urban_daily` should contain long-term historical aggregates;
- `urban_alerts` should contain derived events only when thresholds are exceeded.

### Threshold transparency

The threshold profile panel shows the warning and critical thresholds used for the selected metric.

This is important because many dashboard panels depend on these thresholds:

- warning counts;
- critical counts;
- critical zones;
- alert maps;
- exceedance percentages;
- green infrastructure pressure markers.

Making thresholds visible improves the interpretability and defensibility of the dashboard.

### Idempotent alert generation

The alert-generation workflow is designed to avoid duplicate alert events.

Even if the `hourly_to_alerts` task runs frequently, each alert is written using the timestamp of the hourly measurement window, not the task execution timestamp.

Therefore, repeated executions over the same hourly window update the same logical point in InfluxDB instead of appending duplicate alert events.

This behavior is essential to keep alert counts, maps, and top-zone rankings consistent.

---

## Summary

Urban Environmental Monitoring is an end-to-end data architecture project that integrates time-series telemetry, spatial context, streaming ingestion, derived aggregations, threshold-based alerting, and an interactive Grafana dashboard.

The project demonstrates how heterogeneous data can be modeled using a polyglot persistence approach:

- InfluxDB manages time-series telemetry, aggregates, and alerts.
- PostgreSQL/PostGIS manages static urban context and spatial metadata.
- Kafka and Telegraf simulate the live ingestion pipeline.
- InfluxDB Tasks materialize hourly, daily, and alert layers.
- Grafana transforms the resulting data into an operational and strategic decision-support dashboard.

The dashboard follows a clear analytical path: it starts from the operational status of the selected metric, localizes critical areas, analyzes temporal trends, connects environmental pressure with urban context, and finally verifies data quality.
