export type DetectionBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type Detection = {
  label: string;
  confidence: number;
  bbox: DetectionBox;
  smoothed_bbox: DetectionBox;
  stability_score: number;
  instrument_id: number | null;
  description: string | null;
  operation_steps: string | null;
  safety_warnings: string | null;
};

export type DetectionResponse = {
  timestamp: string;
  latency_ms: number;
  fps_estimate: number;
  detection_source_used: string;
  detection_source_note?: string | null;
  detections: Detection[];
};

export type Instrument = {
  id: number;
  name: string;
  category: string;
  description: string;
  operation_steps: string;
  safety_warnings: string;
};

export type MetricsResponse = {
  events_observed: number;
  mean_latency_ms: number;
  mean_fps: number;
  mean_confidence: number;
  confidence_stability: number;
  latest_timestamp: string | null;
};
