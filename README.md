# Multi-Timeframe-CNN
This is a PRD derived from Zhang Wei's paper "Neural Network-Based Algorithmic Trading Systems: Multi-Timeframe Analysis and High-Frequency Execution in Cryptocurrency Markets." The main blocker to implement this methodology is sourcing the necessary historical orderbook data. Section 6 (not fully debugged) of the PRD outlines the scaffolding implementation using Steve Yegge's `beads` and Ralph mode.

Share, like, and subscribe for more! 


---

# Product Requirements Document & Architecture Specification
**Project:** Zhang Wei Methodology: Multi-Timeframe CNN for Binary Markets
**Target Execution:** Polymarket Binary Options (UP/DOWN)
**Current Phase:** Phase 1 (Core NN Architecture & Inference Pipeline Only)

## 1. Core Architecture & Stack
* **Language:** Python 3.12+
* **Deep Learning Framework:** PyTorch
* **Data Management:** `pandas`, `NumPy`, `SQLModel`
* **Containerization:** Docker (Multi-arch, macOS development -> Linux deployment)
* **Database:** PostgreSQL (managed via `SQLModel` for ORM and validation).
* **Inference Instantiation:** The Data Ingestion Pipeline acts as a shared singleton, feeding a unified database. However, the core neural network architecture serves as a blueprint. Completely independent Inference Engines and models will be instantiated for each target execution timeframe (e.g., one dedicated network for the 5-minute market, and a separate dedicated network for the 15-minute market) to eliminate negative transfer and allow for isolated hyperparameter optimization.
* **Output:** The absolute source of truth is the PostgreSQL database. The inference engine must execute a strict, atomic database-first write sequence. Timestamped JSON files are generated and dropped into `/signals` (to be consumed by the Gengar bot in Phase 2) *only* upon catching a successful database commit, and the JSON payload must explicitly contain the newly generated PostgreSQL `row_id` to establish an unbroken forensic chain.
* **Orchestration:** Ralph loop, supervised via Beads (`bd`) issue tracking.
* **Experimentation Engine:** Customized `autoresearch` (https://github.com/karpathy/autoresearch) autonomous loop. The LLM agent will iteratively modify the model architecture and hyperparameters within a fixed training time budget, optimizing strictly for lowest Validation BCE Loss rather than the default `val_bpb`.

## 2. "Immutable Truth": Data & Mathematical Specifications
*This section dictates the exact calculations and transformations. No module may deviate from these formulas.*

### 2.1 Multi-Timeframe Alignment
The network requires data to be aligned across multiple timeframes simultaneously to capture micro-structure and macro-trend context.
* **Target Timeframes:** 1m (1 Minute), 1H (Hourly), and 1D (Daily).

* **Alignment Rule:** All timeframes must be forward-filled (`ffill`) to synchronize on the lowest resolution timestamp. No look-ahead bias is permitted; row `T` must only contain data completely finalized at `T`.

* **Strict Anti-Leakage Constraint:** Higher-timeframe features (1H, 1D) must ONLY be calculated on fully closed periods. For example, the 1D indicator values applied to the 1-minute rows on Tuesday must be derived strictly from Monday's closing data. Interpolation or partial-period calculations are strictly forbidden.

* **Global Warm-Up Truncation:** To prevent statistical artifacts introduced by zero-padding or backward-filling, the system must strictly truncate (drop) the initial segment of any dataset or live session. The `DataLoader` is forbidden from yielding any 1-minute row until a complete 120-day historical context has accrued (ensuring the absolute maximum lookback, or rolling Z-score window, requirement for the technical channel specified in Section 2.2 is fully populated across all timeframes).

### 2.2 Feature Tensor Definition & Architecture
The network ingests a multi-modal feature set, aligned via forward-filling (`ffill`) to prevent look-ahead bias:
* **Signal Transformation:** Before normalization, all raw OHLCV price inputs across all timeframes must be converted into continuous log returns ($\ln(P_t/P_{t-1})$) to ensure statistical stationarity for the network.

* **CNN Heads (Temporal Data):** Three parallel 1D-CNN branches processing Minute, Hourly, and Daily OHLCV + technical indicators.

* **Context Head:** A fourth CNN branch processing market context (e.g., S&P 500, BTC Dominance).

* **Linear Branches:** Separate feed-forward layers processing Sentiment, On-chain data, and Orderbook dynamics.
  * **Sentiment Definition:** The pipeline natively integrates the GDELT 2.0 dataset. The feature tensor must extract the `V2Tone` metric (a continuous variable representing average document sentiment). To capture immediate contextual mood, the network strictly ingests a sequence of the "last 10 values" sampled at 15-minute intervals, passing this directly into the dedicated sentiment linear branch.
  * **On-Chain Definition:** The pipeline evaluates network activity, transaction patterns, and holder behavior. The linear branch must ingest the following six specific metrics: Active Addresses, Transaction Count/Volume, Exchange NetFlow, Bitcoin MVRV Ratio, Bitcoin MVRV Z-Score, and Supply in Profit (Percent Addresses in Profit).
  * **Orderbook Metrics:** Ingests the top $X$ levels of Bid/Ask depth, alongside derived micro-structure metrics including Orderbook Imbalance Ratios (OIR) and Cumulative Depth Slopes. The absolute maximum ingestion depth is hardcoded to 30 levels to prevent excessive storage and deep-book noise. The parameter $X$ (where $X \le 30$) is explicitly designated as a slicing free variable to be optimized by the `autoresearch` loop.

* **Technical Channels:** Integrates Trend, Momentum, and Volatility indicators computed per-timeframe: **Ichimoku Cloud** (crypto-adjusted settings: 20, 60, 120, 30), Relative Strength Index (RSI), and Bollinger Bands. For the final feature tensors, raw price series must be converted into log returns to ensure stationarity for the input signals. Indicator distances must remain as raw differences and rely on the rolling Z-score for normalization to prevent mathematical errors from negative values.

* **Normalization**: All inputs (post-log return transformation) must be strictly normalized using a rolling Z-score. The 1m and 1H branches must be hardcoded to represent an identical temporal depth (24 hours) to capture immediate daily cycles: the 1m branch uses a 1440-period window, and the 1H branch uses a 24-period window. However, to allow the 1D (Daily) CNN branch to extract macro-trends, its sequence length (lookback window) is detached from the 24-hour constraint and is defined as an independent free variable for the `autoresearch` loop.

- **Normalization & Statistical Clipping**: All inputs (post-log return transformation) must be strictly normalized using a rolling Z-score. The 1m and 1H branches must be hardcoded to represent an identical temporal depth (24 hours) to capture immediate daily cycles (1m: 1440-period window, 1H: 24-period window). However, to allow the 1D (Daily) CNN branch to extract macro-trends, its sequence length (lookback window) is detached from the 24-hour constraint and is defined as an independent free variable for the `autoresearch` loop.

- **Statistical Clipping**: To prevent vanishing or exploding gradients during backpropagation, the PyTorch `DataLoader` must pass all normalized volume and price tensors through a strict statistical clamp. The resulting Z-scores must be strictly clipped at a maximum and minimum boundary of ±10.0. Empirical offline testing confirms this specific threshold effectively truncates catastrophic anomalies (clipping exactly 0.16% of historical long-tail outliers) without artificially flattening legitimate macro accumulation and distribution variance.

### 2.3 Target Variable (The "Y" Label)
For binary cross-entropy training, the target strictly aligns with the specific prediction market's resolution mechanics:
* **Condition:** `Y = 1` IF `Close_{T+X} > Close_{T}` (where `X` is the target market duration, such as 5m or 15m). Else `Y = 0`.

* **Primary Source Enforcement:** The `Close` value utilized for the target variable calculation must explicitly be the primary stream (Coinbase). In the event of historical patching where Coinbase was missing, the system will seamlessly use the fallback patch generated at that timestamp, but the target definition remains anchored to the primary local pipeline.

- **Output Activation & Classification Head:** The final stage of the network is a static Classification Head (Multi-Layer Perceptron) designed to capture non-linear interactions between the unified branches. It consists of intermediate dense layers with non-linear activations (e.g., `Dense(128) -> ReLU -> Dense(64) -> ReLU`) and strictly utilizes L2 regularization (weight decay) to prevent overfitting. The terminal layer is `Dense(1)` with a `Sigmoid` activation function, returning a continuous probability $P(UP) \in [0, 1]$.

### 2.4 Soft Attention Mechanism
* **Unified Dimensionality & Temporal Collapse**: Before attention weights can be applied, the outputs of all Conv1D heads (which have varying temporal lengths) must be standardized to a unified feature dimension. 

* **Two-Phase Temporal Collapse Strategy:** The Temporal Collapse Mechanism must be evaluated in two strict phases. Phase 1 (Baseline Optimization) locks the mechanism exclusively to Global Average Pooling (GAP) to establish the optimal baseline architecture. Phase 2 (Mechanism Evaluation) then opens the search space to test Flattening, Global Max Pooling (GMP), and Last-Step Extraction against that established GAP baseline.

- **Conditional Architecture Rules:** During Phase 2's expanded search, the network must enforce strict structural routing based on the specific mechanism being evaluated. If 'Flattening' is selected, the network must utilize a Progressive Funneling MLP to compress the massive flattened array down to the unified dimension to prevent catastrophic information loss, with the depth and dimensional step sizes defined as independent free variables. Conversely, if GMP, Last-Step Extraction, or the GAP baseline is evaluated, the use of Progressive Funneling MLPs is strictly forbidden; these mechanisms must project to the unified dimension using only a single-step linear projection.

* **Dynamic Weighting**: The standardized, unified-dimension outputs from the 7 branches (4 CNN, 3 Linear) are passed through a conditioned Branch Attention block. The output of the Market Context CNN head serves strictly as the global context vector ($h$). **The Attention Scoring Function** ($f_{att}$) evaluates this context vector against each of the remaining temporal and linear branches ($x_i$) to calculate their specific pre-activation scores ($e_i$). A Softmax function ($\alpha = \exp(e)/\sum\exp(e)$) is then applied to these scores to dynamically assign a scalar weight to each branch before calculating the final weighted sum.

- **Classification Head Routing:** The final weighted sum produced by the Softmax weighting is passed directly into a static Classification Head (MLP). This multi-layer structure processes the combined, unified-dimension tensors to model complex non-linear interactions before outputting the final prediction, ensuring the network does not encounter a mathematical bottleneck at the final projection.
## 3. Data Integrity & Pipeline Resilience
* **Historical Master Dataset Creation:** The primary training and testing corpus will be constructed by merging the Kaggle 1m BTC/USD dataset (2015-2025) with freshly scraped 2026 historical data to establish an expanded, comprehensive time period.

* **Missing Data Protocol (Historical)**: The ingestion pipeline will specifically utilize the Kaggle "Historical Index" (an average of multiple exchanges) as the baseline to natively eliminate exchange-specific data gaps. If both the primary and fallback data sources are missing for a specific timestamp, the system must immediately halt the historical ingestion pipeline and log a critical data continuity error to `quarantine.log` and `pipeline.log`. Applying zero-volume forward-filling or attempting to bridge the gap with secondary exchanges is strictly forbidden.
  * **Supplementary API Failure (Halt, Ping, and Backfill):** If a supplementary API (GDELT for sentiment, or the live on-chain provider) fails to return a row or times out during live inference, the inference engine must instantly halt and cease generating signals. Graceful degradation (e.g., zeroing attention weights or forward-filling) is explicitly forbidden. The ingestion worker enters a blocking loop, continuously pinging the downed API. Upon recovery, the worker executes an Index-Backed High-Watermark Query (`SELECT MAX(timestamp)`) to instantly identify the exact temporal gap without a full table scan. The worker must backfill this specific missing window into the PostgreSQL database before the inference engine calculates the backlog and resumes real-time operation.

* **Live Bridge & Sourcing:** For live inference, real-time data will be sourced via the Coinbase Advanced Trade API.

* **Dynamic Scraper Watermarking:** The historical REST scraper for current-year data must never execute using hardcoded start dates. Instead, it must implement a dynamic high-watermark pattern. Upon initialization, the scraper will query the local PostgreSQL database for `MAX(timestamp)` from the `raw_market_data` table where `source_exchange` equals the primary or fallback feeds. The scraper will seamlessly resume data extraction from this exact timestamp plus one minute. 

* **Missing Data Protocol (Live):** 
  - **Immediate Micro-Gap Patching (<5 mins)**: Accuracy is strictly prioritized over API bandwidth. If the primary live stream (Coinbase) fails to deliver a candle, the system instantly executes concurrent REST API queries to the Fallback Matrix (Binance, Kraken, Bitstamp). 
    - The synthesis engine must discard static percentage thresholds and apply a Dynamic Outlier Rejection Filter. The ingestion worker maintains a vectorized, rolling buffer (e.g., 24-hour trailing window) calculating the real-time spread between Coinbase and the Fallback Matrix. 
    - The rejection boundary is calculated dynamically as $\mu_{\text{spread}} + (k \times \sigma_{\text{spread}})$ (where $k$ is a strict scaling factor to be empirically determined). The engine must handle `NaN` or infinite spread values gracefully during identical price matches. 
    - Fallback data points breaching this context-aware threshold are excluded. The remaining valid data is averaged to generate a Synthesized Candle, dynamically widening its tolerance during true macro volatility events while blocking genuine API anomalies.
  * **Dynamic Volume Scaling:** At the exact moment of the first dropped candle, the system queries the 24-hour summary endpoints of the valid fallback exchanges. The sum of their combined 24-hour volume is compared against the local database's rolling 24-hour Coinbase volume to generate a volume-scaling multiplier. To prevent drift during high-volatility events, this multiplier must be recalculated dynamically every 60 seconds for the duration of the micro-gap.
  * **Active Fallback Mode (>5 mins):** If >5 consecutive minutes are missing from the primary feed, the `Data Integrity Auditor` logs a major disconnect to `quarantine.log` but does not halt the pipeline. Instead, it fully transitions the live ingestion engine to consume and aggregate the streams from Fallback Matrix simultaneously. The pipeline will continue to operate on these Synthesized Candles until background polling confirms the Coinbase API has fully stabilized, strictly defined as returning exactly 3 consecutive, valid 1-minute REST polls before failing back to the primary stream.
  * **Local State Buffer:** To prevent data loss during local disk I/O spikes or database write-locks, the high-frequency ingestion engine must utilize an in-memory buffer (e.g., an `asyncio.Queue` or temporary Redis queue). Raw WebSocket payloads are held here transiently if `SQLModel` commits experience high latency or time out.

## 4. Standardized Logging & Telemetry
Distributed logging isolates infrastructure failures from mathematical anomalies. All logs MUST be output in strict JSON format (`{"timestamp": ..., "level": ..., "event": ...}`) to allow for programmatic auditing and downstream dashboarding.

* **`pipeline.log`:** Infrastructure lifecycle (API calls, REST backoff triggers, database connection states, WebSocket latency, Docker state).
* **`engine.log`:** Quantitative processes (Tensor shape mismatches, loss convergence anomalies, attention weight extremes, `autoresearch` loop status).
* **`quarantine.log`:** Forensic data for records failing integrity checks (e.g., flatlined orderbooks, >0.5% deviation in fallback matrix synthesis, disconnected API streams).
- **`quarantine.log`**: Forensic data for records failing integrity checks (e.g., flatlined orderbooks, dynamic threshold spread deviation in fallback matrix synthesis, mathematically impossible MAD-based price drops, disconnected API streams).

## 5. Relational Schema (SQLModel)
* **Database Architecture Decisions:** All primary keys must utilize standard Auto-Incrementing Integers (`BigInt`) for performance and simplicity in this single-instance setup. To prevent accidental data duplication during overlaps or reconnects, critical tables must enforce composite unique constraints (e.g., `UNIQUE(timestamp, source_exchange)`).

* **`raw_market_data`**: High-frequency ingest (OHLCV, Orderbook snapshots). Must include a `source_exchange` column (e.g., 'Historical_Index', 'Coinbase', 'Synthesized_Candle', 'Fallback_Matrix_Active') to track data lineage. Orderbook snapshots must be explicitly stored as `JSONB` columns to handle variable arrays efficiently without row-count inflation, strictly capped at a maximum absolute depth of 30 levels (Bid/Ask).

* **`raw_sentiment_data`**: Stores the 15-minute resolution global news sentiment extracted from GDELT 2.0, primarily tracking the `V2Tone` continuous metric.

* **`raw_onchain_data`**: Stores the required network metrics (Active Addresses, Tx Volume, NetFlow, MVRV Ratio, MVRV Z-Score, Supply in Profit). Act as the canonical internal format mapping disparate API/CSV schemas to a unified standard.
  * *Translation Layer Constraint:* Because the system uses a Hybrid Provisioning Strategy (Dune Analytics for historical, CoinMetrics/CryptoQuant for live inference), this schema acts as the canonical internal format. The SQLModel layer must ensure payloads from both the Dune historical CSVs and the live REST APIs map cleanly to these columns without type or scaling mismatches before committing.

* **`feature_tensors`**: Serialized representation of the normalized data ready for PyTorch `DataLoader`. To eliminate I/O bottlenecks during the `autoresearch` training loops, the multi-dimensional tensors must be strictly serialized as compressed binary blobs and stored utilizing the `BYTEA` (SQLModel `LargeBinary`) column type.

* **`inference_logs`**: The absolute source of truth and primary trigger ledger for the pipeline. It stores the generated $P(UP)$ signal, the `target_timeframe` (e.g., 5m, 15m), the timestamp of execution, the exact `git commit hash` of the model weights, and the `execution_status` (defaulting to 'PENDING'). A JSON signal file is strictly forbidden from being generated unless a row is successfully committed to this table first.

## 6. Execution Plan: Beads & Ralph Orchestration

- Issue: Initialize Multi-Arch Docker & Project Scaffold. Set up the Python 3.12+ project structure with Docker containerization explicitly configured for multi-arch compatibility. Install foundational dependencies: PyTorch, pandas, NumPy, SQLModel, PostgreSQL (via docker-compose), and ccxt. (Gate: Container launches cleanly, connects to local Postgres, and passes a basic health-check script).

- Issue: Implement complete PostgreSQL/SQLModel relational schema. Define tables for `raw_market_data`, `raw_sentiment_data`, `raw_onchain_data`, `feature_tensors`, and `inference_logs`. Implement Auto-Incrementing `BigInt` primary keys across all tables. Enforce composite unique constraints (e.g., timestamp + source) to prevent duplicate records. Configure `raw_market_data` to store orderbooks natively via `JSONB`. Configure `feature_tensors` to store PyTorch arrays natively via `LargeBinary` (BYTEA) for fast I/O. (Gate: Script automatically generates tables, successfully executes basic DB write/read tests, and correctly rejects a duplicate simulated payload violating the composite constraint).

- Issue: Implement core async worker-pool and scheduler patterns. Mirror the go-ohlcv-collector architecture using Python asyncio and ccxt. (Gate: Unit tests verifying worker concurrency and isolated ccxt instantiation).

- Issue: Implement Data Integrity Auditor. Route errors to `pipeline.log`, `engine.log`, and `quarantine.log` based on error class. Actively monitor the primary stream for absolute log returns exceeding ±0.20. Actively quarantine these corrupted rows and treat them as missing data to trigger a patch, unless they meet the exact criteria for an immediate, opposite "Rebound Effect". (Gate: Unit tests triggering the April 2017 flash-crash artifact assert it is quarantined, routed to the correct log, and triggers a fallback patch).

- Issue: Ingest the 2015-2025 local datasets. Map the primary local dataset (`BTCUSD_1m_Coinbase.csv`) and the fallback dataset (`BTCUSD_1m_Combined_Index.csv`) into the `raw_market_data` schema, properly tagging the `source_exchange` column. (Gate: Script outputs confirmation of successful, non-overlapping ingestion for both datasets into the database).

- Issue: Implement timeframe-specific rolling Z-score normalizer. Hardcode the lookback windows to their 24-hour equivalents (1m: 1440, 1H: 24, 1D: 1). (Gate: Feed mocked static tensors into the normalizer and assert mathematically correct scaled outputs without temporal leakage).

- Issue: Develop the Dynamic Outlier Rejection & Synthesis Utilities. Create isolated Python functions for calculating the dynamic $k \times \sigma$ spread threshold and the dynamic 24-hour volume scaling multiplier. These functions must handle `NaN` or infinite spread values gracefully during identical price matches. (Gate: Unit tests verifying mathematically correct synthesis and rejection using mocked, static fallback matrix arrays containing known extreme outliers).

- Issue: Execute historical 2015-2025 gap-patching. Patch the local database records by substituting missing Coinbase rows with the local `BTCUSD_1m_Combined_Index` data, bypassing spread filters entirely for this historical block. (Gate: Script outputs continuity report proving zero missing timestamps from 2015 to 2025).

- Issue: Develop historical REST scraper for 2026 Primary Data. Use `ccxt` to fetch 1m OHLCV data from Coinbase up to the present. Implement exponential backoff with randomized jitter to strictly avoid HTTP 429 rate limits. (Gate: 2026 data successfully persists to DB without overlaps or rate-limit crashes).

- Issue: Develop historical REST scraper for 2026 Primary Data. Use `ccxt` to fetch 1m OHLCV data from Coinbase up to the present. The initialization logic must dynamically execute a high-watermark query (`MAX(timestamp)`) against the `raw_market_data` table to determine the exact delta-loading start point. Implement exponential backoff with randomized jitter to strictly avoid HTTP 429 rate limits. (Gate: Script dynamically identifies the local dataset cutoff, bridges the gap to the current timestamp, and successfully persists data to the DB without overlaps or rate-limit crashes).

- Issue: Execute 2026 gap-patching via Fallback Matrix. Scan the freshly scraped 2026 historical data for missing candles. Execute concurrent REST API queries to the Fallback Matrix (Binance, Kraken, Bitstamp) to synthesize missing rows, natively routing the data through the Dynamic Outlier Rejection functions built previously. (Gate: Script outputs continuity report proving zero missing timestamps up to the present without triggering HTTP 429 errors).

- Issue: Develop the Live Database Ingestion Buffer. Implement an in-memory `asyncio.Queue` worker specifically designed to catch and hold incoming payloads during SQLModel commit latency or transaction locks, ensuring atomic, database-first writes. (Gate: Buffer successfully holds data during a mocked database lock and correctly flushes to Postgres upon release).

- Issue: Create the Live Primary WebSocket Pipeline. Integrate the Coinbase WebSocket feed with the `asyncio.Queue` buffer. (Gate: Live data streams cleanly into Postgres for a 5-minute observation window without dropping packets).

- Issue: Integrate Live Fallback Matrix Micro-Gap Patching. Connect the active pipeline to the Fallback Matrix REST endpoints. If a WebSocket packet drops, query the matrix and route through the dynamic synthesis utility before writing to the queue. (Gate: Unit tests successfully patching a mocked dropped Coinbase candle using real-time API fallback queries).

- Issue: Generate historical Sentiment and On-Chain datasets. Write the Dune Analytics SQL queries to extract the 6 canonical on-chain metrics (2015-2025) and extract the corresponding GDELT 2.0 V2Tone data. Map these CSVs into the Postgres database. (Gate: DB query confirms successful ingestion and perfect temporal alignment against the Kaggle dataset without look-ahead bias).

- Issue: Implement Live Ingestion for GDELT Sentiment API. Build the REST polling worker for the `raw_sentiment_data` table. Strictly implement the halt-and-ping protocol: if the API fails, the worker must block and ping the endpoint. Upon recovery, it must utilize an Index-Backed High-Watermark query (`SELECT MAX(timestamp)`) to efficiently locate the exact disconnect point and execute a targeted database backfill. Forward-filling is explicitly forbidden. (Gate: Unit tests successfully trigger a mocked API outage, verifying the engine halts, queries the index, and backfills exactly the missing temporal gap).

- Issue: Implement Live Ingestion for On-Chain Data. Build the REST polling worker for CoinMetrics/CryptoQuant targeting the `raw_onchain_data` schema. Replicate the exact halt-and-ping backfill protocol used for GDELT. (Gate: Unit tests pass the mocked API outage and verify payloads correctly map to the canonical Dune schema).
