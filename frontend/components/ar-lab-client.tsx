"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";

import { apiUrl } from "../lib/api";
import type {
  Detection,
  DetectionResponse,
  Instrument,
  MetricsResponse
} from "../lib/types";

type RuntimeStatus = "idle" | "starting" | "running" | "error";
type GuidanceMode = "ai" | "module";
type InputSource = "camera" | "upload";
type DetectionMode = "model" | "ai";

const DETECTION_INTERVAL_MS = 250;

function boxWidth(box: Detection["bbox"]): number {
  return Math.max(1, box.x2 - box.x1);
}

function boxHeight(box: Detection["bbox"]): number {
  return Math.max(1, box.y2 - box.y1);
}

function drawDetections(
  canvas: HTMLCanvasElement,
  detections: Detection[],
  selectedLabel: string | null
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  detections.forEach((det) => {
    const box = det.smoothed_bbox;
    const active = selectedLabel === det.label;
    const color = active ? "#f97316" : "#22d3ee";

    ctx.strokeStyle = color;
    ctx.lineWidth = active ? 4 : 2;
    ctx.strokeRect(box.x1, box.y1, boxWidth(box), boxHeight(box));

    const label = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
    ctx.font = '14px "IBM Plex Mono", monospace';
    const textMetrics = ctx.measureText(label);
    const textWidth = Math.max(120, textMetrics.width + 12);
    const textHeight = 22;

    const textX = box.x1;
    const textY = Math.max(0, box.y1 - textHeight - 3);

    ctx.fillStyle = "rgba(8, 47, 73, 0.85)";
    ctx.fillRect(textX, textY, textWidth, textHeight);

    ctx.fillStyle = "#f8fafc";
    ctx.fillText(label, textX + 6, textY + 15);
  });
}

function loadImageFromDataUrl(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new window.Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Unable to decode uploaded image"));
    image.src = dataUrl;
  });
}

export default function ARLabClient() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const captureRef = useRef<HTMLCanvasElement | null>(null);
  const inFlightRef = useRef(false);

  const [runtime, setRuntime] = useState<RuntimeStatus>("idle");
  const [runtimeError, setRuntimeError] = useState<string | null>(null);

  const [detections, setDetections] = useState<Detection[]>([]);
  const [instruments, setInstruments] = useState<Instrument[]>([]);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [latestLatency, setLatestLatency] = useState<number>(0);
  const [latestFps, setLatestFps] = useState<number>(0);
  const [latestDetectionSourceUsed, setLatestDetectionSourceUsed] = useState<string>("model");
  const [latestDetectionSourceNote, setLatestDetectionSourceNote] = useState<string | null>(null);
  const [inputSource, setInputSource] = useState<InputSource>("camera");
  const [detectionMode, setDetectionMode] = useState<DetectionMode>("model");
  const [uploadedImageData, setUploadedImageData] = useState<string | null>(null);
  const [uploadedImageName, setUploadedImageName] = useState<string>("");

  const [selectedInstrument, setSelectedInstrument] = useState<string>("");
  const [question, setQuestion] = useState<string>(
    "What is the safest step-by-step way to operate this instrument?"
  );
  const [guidanceMode, setGuidanceMode] = useState<GuidanceMode>("ai");
  const [guidanceAnswer, setGuidanceAnswer] = useState<string>("");
  const [guidanceModel, setGuidanceModel] = useState<string>("");
  const [isAsking, setIsAsking] = useState<boolean>(false);

  const instrumentOptions = useMemo(
    () => [...new Set([...detections.map((d) => d.label), ...instruments.map((i) => i.name)])],
    [detections, instruments]
  );

  const selectedDetection = useMemo(
    () => detections.find((d) => d.label === selectedInstrument) ?? null,
    [detections, selectedInstrument]
  );

  const selectedInstrumentInfo = useMemo(
    () => instruments.find((instrument) => instrument.name === selectedInstrument) ?? null,
    [instruments, selectedInstrument]
  );
  const detectionSourceLabel = useMemo(() => {
    if (latestDetectionSourceUsed === "ai") {
      return "AI";
    }
    if (latestDetectionSourceUsed === "ai-empty") {
      return "AI (No Match)";
    }
    if (latestDetectionSourceUsed === "ai-unavailable") {
      return "AI (Unavailable)";
    }
    if (latestDetectionSourceUsed === "ai-error") {
      return "AI (Error)";
    }
    return "Model";
  }, [latestDetectionSourceUsed]);

  const clearOverlay = useCallback(() => {
    const overlay = overlayRef.current;
    if (!overlay) {
      return;
    }
    const context = overlay.getContext("2d");
    if (!context) {
      return;
    }
    context.clearRect(0, 0, overlay.width, overlay.height);
  }, []);

  const stopCamera = useCallback(() => {
    const media = videoRef.current?.srcObject as MediaStream | null;
    media?.getTracks().forEach((track) => track.stop());
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setRuntime("idle");
  }, []);

  useEffect(() => {
    const fetchStaticData = async () => {
      try {
        const [instrumentRes, metricsRes] = await Promise.all([
          fetch(apiUrl("/instruments")),
          fetch(apiUrl("/metrics"))
        ]);

        if (instrumentRes.ok) {
          const instrumentData: Instrument[] = await instrumentRes.json();
          setInstruments(instrumentData);
          setSelectedInstrument((previous) => previous || instrumentData[0]?.name || "");
        }

        if (metricsRes.ok) {
          const metricsData: MetricsResponse = await metricsRes.json();
          setMetrics(metricsData);
        }
      } catch {
        // Non-fatal for initial rendering.
      }
    };

    void fetchStaticData();
  }, []);

  useEffect(() => {
    if (selectedInstrument || instrumentOptions.length === 0) {
      return;
    }
    setSelectedInstrument(instrumentOptions[0]);
  }, [instrumentOptions, selectedInstrument]);

  const runDetectionRequest = useCallback(
    async (imageData: string, frameWidth: number, frameHeight: number) => {
      const response = await fetch(apiUrl("/detection/predict"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          image_b64: imageData,
          run_refinement: true,
          mode: detectionMode
        })
      });

      if (!response.ok) {
        throw new Error("Detection request failed");
      }

      const payload: DetectionResponse = await response.json();
      setDetections(payload.detections);
      setLatestLatency(payload.latency_ms);
      setLatestFps(payload.fps_estimate);
      setLatestDetectionSourceUsed(payload.detection_source_used || "model");
      setLatestDetectionSourceNote(payload.detection_source_note ?? null);
      if (detectionMode === "ai" && payload.detection_source_used !== "ai") {
        if (payload.detection_source_used === "ai-unavailable") {
          setRuntimeError(
            payload.detection_source_note ||
              "AI detection unavailable. Check GROQ_API_KEY and restart backend."
          );
        } else if (payload.detection_source_used === "ai-error") {
          setRuntimeError(payload.detection_source_note || "AI detection request failed.");
        } else if (payload.detection_source_used === "ai-empty") {
          setRuntimeError(
            payload.detection_source_note ||
              "AI detection returned no known lab instrument for this image."
          );
        }
      } else {
        setRuntimeError(null);
      }

      if (payload.detections.length > 0 && !payload.detections.some((d) => d.label === selectedInstrument)) {
        setSelectedInstrument(payload.detections[0].label);
      }

      const overlay = overlayRef.current;
      if (overlay) {
        overlay.width = frameWidth;
        overlay.height = frameHeight;
        drawDetections(overlay, payload.detections, selectedInstrument || null);
      }
      setRuntimeError(null);
    },
    [detectionMode, selectedInstrument]
  );

  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setRuntime("error");
      setRuntimeError("Browser does not support webcam capture.");
      return;
    }

    stopCamera();
    setInputSource("camera");
    setUploadedImageData(null);
    setUploadedImageName("");
    clearOverlay();

    setRuntime("starting");
    setRuntimeError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setRuntime("running");
    } catch {
      setRuntime("error");
      setRuntimeError("Unable to access webcam. Check browser permissions.");
    }
  }, [clearOverlay, stopCamera]);

  const analyzeUploadedImage = useCallback(
    async (imageData: string) => {
      try {
        const image = await loadImageFromDataUrl(imageData);
        await runDetectionRequest(imageData, image.naturalWidth, image.naturalHeight);
        setRuntimeError(null);
      } catch {
        setRuntimeError("Failed to process uploaded image.");
      }
    },
    [runDetectionRequest]
  );

  const handleUploadFile = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = "";
      if (!file) {
        return;
      }
      if (!file.type.startsWith("image/")) {
        setRuntimeError("Please upload a valid image file.");
        return;
      }

      stopCamera();
      setInputSource("upload");
      setRuntimeError(null);

      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result !== "string") {
          setRuntimeError("Failed to read uploaded image.");
          return;
        }
        setUploadedImageData(reader.result);
        setUploadedImageName(file.name);
        clearOverlay();
        void analyzeUploadedImage(reader.result);
      };
      reader.onerror = () => {
        setRuntimeError("Failed to read uploaded image.");
      };
      reader.readAsDataURL(file);
    },
    [analyzeUploadedImage, clearOverlay, stopCamera]
  );

  const triggerFileUpload = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  useEffect(() => {
    return () => {
      const media = videoRef.current?.srcObject as MediaStream | null;
      media?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    if (inputSource !== "upload" || !uploadedImageData) {
      return;
    }
    clearOverlay();
    void analyzeUploadedImage(uploadedImageData);
  }, [analyzeUploadedImage, clearOverlay, detectionMode, inputSource, uploadedImageData]);

  useEffect(() => {
    if (runtime !== "running" || inputSource !== "camera") {
      return;
    }

    const activeInterval = detectionMode === "ai" ? 3000 : DETECTION_INTERVAL_MS;
    const interval = window.setInterval(async () => {
      if (inFlightRef.current) {
        return;
      }

      const video = videoRef.current;
      const capture = captureRef.current;
      const overlay = overlayRef.current;

      if (!video || !capture || !overlay || video.videoWidth === 0 || video.videoHeight === 0) {
        return;
      }

      inFlightRef.current = true;

      try {
        capture.width = video.videoWidth;
        capture.height = video.videoHeight;

        const captureCtx = capture.getContext("2d");
        if (!captureCtx) {
          return;
        }

        captureCtx.drawImage(video, 0, 0, capture.width, capture.height);
        const imageData = capture.toDataURL("image/jpeg", 0.65);
        await runDetectionRequest(imageData, video.videoWidth, video.videoHeight);
      } catch {
        setRuntimeError("Detection request failed. Please try again.");
      } finally {
        inFlightRef.current = false;
      }
    }, activeInterval);

    return () => window.clearInterval(interval);
  }, [detectionMode, inputSource, runDetectionRequest, runtime]);

  useEffect(() => {
    const interval = window.setInterval(async () => {
      try {
        const response = await fetch(apiUrl("/metrics"));
        if (!response.ok) {
          return;
        }

        const payload: MetricsResponse = await response.json();
        setMetrics(payload);
      } catch {
        // Metrics refresh errors are not fatal.
      }
    }, 1800);

    return () => window.clearInterval(interval);
  }, []);

  const submitQuestion = useCallback(async () => {
    if (!selectedInstrument || !question.trim() || isAsking) {
      return;
    }

    setIsAsking(true);
    setGuidanceAnswer("");

    try {
      const sceneContext = detections
        .map((d) => `${d.label}(${(d.confidence * 100).toFixed(0)}%)`)
        .join(", ");

      const response = await fetch(apiUrl("/guidance/ask"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          instrument_name: selectedInstrument,
          question,
          context: sceneContext,
          mode: guidanceMode
        })
      });

      if (!response.ok) {
        setGuidanceAnswer("Guidance request failed. Please retry.");
        setGuidanceModel("error");
        return;
      }

      const payload = (await response.json()) as {
        instrument_name: string;
        answer: string;
        model_used: string;
      };

      setGuidanceAnswer(payload.answer);
      setGuidanceModel(payload.model_used);
    } finally {
      setIsAsking(false);
    }
  }, [detections, guidanceMode, isAsking, question, selectedInstrument]);

  return (
    <main className="page-shell">
      <section className="hero-row">
        <h1>Scientific Laboratory AR Framework</h1>
        <p>
          Browser-native instrument recognition and procedural guidance with
          EfficientDet-ready detection, Kalman-smoothed overlays, and AI
          assistance.
        </p>
      </section>

      <section className="content-grid">
        <div className="camera-card">
          <header>
            <strong>Live AR Workspace</strong>
            <span className={`status status-${runtime}`}>{runtime}</span>
          </header>

          <div className="camera-stage">
            <video
              ref={videoRef}
              playsInline
              muted
              className={inputSource === "camera" ? "" : "media-hidden"}
            />
            {uploadedImageData ? (
              <img
                src={uploadedImageData}
                alt={uploadedImageName || "Uploaded laboratory image"}
                className={inputSource === "upload" ? "" : "media-hidden"}
              />
            ) : null}
            {inputSource === "upload" && !uploadedImageData ? (
              <div className="stage-placeholder">Upload a laboratory image to analyze.</div>
            ) : null}
            <canvas ref={overlayRef} />
            <canvas ref={captureRef} className="hidden-capture" />
          </div>

          <div className="camera-actions">
            <button onClick={startCamera} disabled={runtime === "running" || runtime === "starting"}>
              {runtime === "running" ? "Camera Active" : "Start Camera"}
            </button>
            <button className="button-secondary" onClick={triggerFileUpload}>
              Upload Image
            </button>
            <div className="mode-control">
              <label htmlFor="detection-mode">Detection Source</label>
              <select
                id="detection-mode"
                value={detectionMode}
                onChange={(event) => setDetectionMode(event.target.value as DetectionMode)}
              >
                <option value="model">Model</option>
                <option value="ai">AI</option>
              </select>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden-file-input"
              onChange={handleUploadFile}
            />
            {uploadedImageData ? (
              <button className="button-secondary" onClick={() => void analyzeUploadedImage(uploadedImageData)}>
                Re-run Upload
              </button>
            ) : null}
            <div className="metric-pill">
              <span>Latency</span>
              <strong>{latestLatency.toFixed(1)} ms</strong>
            </div>
            <div className="metric-pill">
              <span>FPS</span>
              <strong>{latestFps.toFixed(1)}</strong>
            </div>
            <div className="metric-pill">
              <span>Source</span>
              <strong>{inputSource === "camera" ? "Live" : "Upload"}</strong>
            </div>
            <div className="metric-pill">
              <span>Detector</span>
              <strong>{detectionSourceLabel}</strong>
            </div>
          </div>

          {inputSource === "upload" && uploadedImageName ? (
            <p className="upload-meta">File: {uploadedImageName}</p>
          ) : null}
          {latestDetectionSourceNote && latestDetectionSourceUsed !== "ai" ? (
            <p className="upload-meta">Detection Note: {latestDetectionSourceNote}</p>
          ) : null}

          {runtimeError ? <p className="error-text">{runtimeError}</p> : null}
        </div>

        <aside className="panel-stack">
          <div className="panel">
            <h2>Detected Instruments</h2>
            <div className="scroll-list">
              {detections.length === 0 ? (
                <p className="muted">No detections yet. Start camera to begin recognition.</p>
              ) : (
                detections.map((det) => (
                  <button
                    key={`${det.label}-${det.confidence}`}
                    className={`detection-item ${selectedInstrument === det.label ? "active" : ""}`}
                    onClick={() => setSelectedInstrument(det.label)}
                  >
                    <div>
                      <strong>{det.label}</strong>
                      <small>{(det.confidence * 100).toFixed(1)}% confidence</small>
                    </div>
                    <small>Stability {(det.stability_score * 100).toFixed(0)}%</small>
                  </button>
                ))
              )}
            </div>
          </div>

          <div className="panel">
            <h2>Instructional Guidance</h2>
            <label htmlFor="instrument">Instrument</label>
            <select
              id="instrument"
              value={selectedInstrument}
              onChange={(event) => setSelectedInstrument(event.target.value)}
            >
              {instrumentOptions.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>

            <label htmlFor="question">Question</label>
            <textarea
              id="question"
              rows={3}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
            />

            <label htmlFor="guidance-mode">Guidance Source</label>
            <select
              id="guidance-mode"
              value={guidanceMode}
              onChange={(event) => setGuidanceMode(event.target.value as GuidanceMode)}
            >
              <option value="ai">AI</option>
              <option value="module">Module</option>
            </select>

            <button onClick={submitQuestion} disabled={!selectedInstrument || isAsking}>
              {isAsking ? "Generating..." : "Generate Guidance"}
            </button>

            {selectedDetection?.safety_warnings || selectedInstrumentInfo?.safety_warnings ? (
              <p className="safety-banner">
                <strong>Safety:</strong>{" "}
                {selectedDetection?.safety_warnings || selectedInstrumentInfo?.safety_warnings}
              </p>
            ) : null}

            {guidanceAnswer ? (
              <div className="answer-box">
                <small>Model: {guidanceModel}</small>
                <pre>{guidanceAnswer}</pre>
              </div>
            ) : null}
          </div>

          <div className="panel metrics-panel">
            <h2>System Metrics</h2>
            <div className="metric-grid">
              <div>
                <span>Events</span>
                <strong>{metrics?.events_observed ?? 0}</strong>
              </div>
              <div>
                <span>Mean Latency</span>
                <strong>{(metrics?.mean_latency_ms ?? 0).toFixed(1)} ms</strong>
              </div>
              <div>
                <span>Mean FPS</span>
                <strong>{(metrics?.mean_fps ?? 0).toFixed(1)}</strong>
              </div>
              <div>
                <span>Confidence</span>
                <strong>{((metrics?.mean_confidence ?? 0) * 100).toFixed(1)}%</strong>
              </div>
              <div>
                <span>Stability</span>
                <strong>{((metrics?.confidence_stability ?? 0) * 100).toFixed(1)}%</strong>
              </div>
            </div>
          </div>
        </aside>
      </section>
    </main>
  );
}
