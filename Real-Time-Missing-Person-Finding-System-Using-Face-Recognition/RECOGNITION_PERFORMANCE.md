# Recognition Performance Evaluation (logs/recognition.log)

This report summarizes and evaluates recognition behavior captured in `logs/recognition.log` during a real-time run of the pipeline.

## Log Overview
- Threads started: VideoReader, Tracker, Embedding, Recognition, CrowdMonitor, BatchRecognition
- Device: cpu for detector, embedder, and recognition model
- Health endpoint: `http://localhost:8080/health`
- Runtime embeddings loaded: 222
- Precomputed embeddings loaded: 35
- Exported images processed: multiple entries (e.g., `69330fc88dd7f9b6f6a859d9_1.jpg`)

## Recognition Results Summary
- Matches per frame: log shows repeated “Frame X, Track 1: 1 matches found”
- Number of match lines observed: 61
- Cosine similarity values:
  - Range observed: ~0.49 to ~0.63
  - Typical values: ~0.55–0.60
- Score01 values (mapped to [0..1]):
  - Range observed: ~0.743 to ~0.817
  - Typical values: ~0.78–0.80
- Interpretation:
  - Scores in ~0.78–0.80 indicate moderately strong similarity under the current model and preprocessing.
  - Consistency of matches suggests stable embeddings across frames for Track 1.

## Quantitative Summary (from Terminal#0–1014)
- Saved embeddings: 131 entries
- “Processing new image” events: 15 entries
- “No matches found” warnings: 10 entries
- FileNotFoundError occurrences when saving annotated results: 129 entries
- Example match lines with cosine and score01:
  - 0.5295 / 0.7648, 0.5848 / 0.7924, 0.6032 / 0.8016, 0.6344 / 0.8172, 0.4901 / 0.7450
- Approximate statistics (cosine):
  - Min ≈ 0.486, Max ≈ 0.634, Mean ≈ 0.57, P95 ≈ 0.61

## Errors Observed
- Frequent `FileNotFoundError` during annotated result saving:
  - Example: `No such file or directory: 'exported_images\69330fc88dd7f9b6f6a859d9_1.jpg'`
  - Likely cause: the image is moved to a processed folder before `annotate_and_save` reads it, or path mismatch between batch-recognition and runtime crops.
  - Effect: annotated montage is not saved, but matches are still computed and logged.
  - Suggested fixes:
    - Copy or reference images before moving; move only after annotations are saved.
    - Update `annotate_and_save` to fall back to:
      - Runtime crops directory: `results/crops/`
      - Dataset directories (if applicable)
    - Confirm batch-recognition flow writes annotated outputs before relocating source images.

## Latency and Throughput
- The recognition log does not include per-call latency values directly.
- The pipeline records recognition latency via metrics:
  - Latency recording: [pipeline.py:L331](file:///d:/Real-Time-Missing-Person-Finding-System-Using-Face-Recognition/pipeline.py#L331)
  - Used in recognition thread: [pipeline.py:L974](file:///d:/Real-Time-Missing-Person-Finding-System-Using-Face-Recognition/pipeline.py#L974)
  - Health endpoint exposes averages, including `avg_recognition_latency_ms`: [pipeline.py:L375](file:///d:/Real-Time-Missing-Person-Finding-System-Using-Face-Recognition/pipeline.py#L375) and server init [pipeline.py:L1572](file:///d:/Real-Time-Missing-Person-Finding-System-Using-Face-Recognition/pipeline.py#L1572)
- How to view latency:
  - Open `http://localhost:8080/health` to see `avg_recognition_latency_ms`, `fps`, and other metrics.
  - If a file-based latency report is needed, add a periodic logger to dump metrics to `logs/recognition_metrics.log`.
- Throughput hints:
  - With 131 saved embeddings and 61 match events in the sampled window, recognition attempts are sustained and consistent.
  - Actual recognition FPS depends on embedding worker count, device (cpu/cuda), and queue backpressure.

## Configuration Notes
- Threshold tuning:
  - Current matches typically around cosine 0.55–0.60; adjust `recognition_threshold` to control sensitivity.
  - Higher threshold reduces false positives; lower finds more candidates.
- Embeddings sources:
  - Runtime embeddings: saved during live processing
  - Precomputed embeddings: loaded from `embeddings_index.csv` in your embeddings directory

## Recommendations
- Resolve annotation path issues:
  - Ensure source images remain accessible until after annotation is saved.
  - Add a fallback path search in the recognition result saver for runtime crops.
- Improve performance:
  - Use `cuda` if available to reduce embedding and recognition latency.
  - Consider tuning input sizes (`embedding_input_size`) and the number of embedding workers.
  - Keep the embeddings index clean and representative to improve match quality.
- Add metrics logging:
  - Periodically log `avg_recognition_latency_ms`, `fps`, queue sizes to a file for historical analysis.
  - Optionally, log per-match latency if needed for detailed profiling.

## How to Reproduce and Monitor
- Run real-time pipeline:
  ```
  python pipeline.py --config configs/default.yaml --video-source 0
  ```
- Visit health endpoint:
  - `http://localhost:8080/health`
- Check recognition logs:
  - `logs/recognition.log` for match lines and errors
  - Add a metrics log file for latency aggregation if required

