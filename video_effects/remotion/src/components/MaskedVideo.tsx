import React, { useCallback, useRef } from "react";
import { AbsoluteFill, OffthreadVideo, useVideoConfig } from "remotion";

/**
 * Canvas compositing component that combines an original video with a
 * grayscale mask video to produce a transparent person layer.
 *
 * Per frame:
 *  1. Draw original video frame to a temp canvas
 *  2. Read mask video frame luminance
 *  3. Set pixel alpha = mask luminance (0=transparent, 255=opaque)
 *  4. Draw modified ImageData to the visible canvas
 *
 * Both OffthreadVideo elements call compositeFrames() on each frame callback.
 * Whichever fires second produces the correct composite with both sources.
 */

interface MaskedVideoProps {
  originalSrc: string;
  maskSrc: string;
}

export const MaskedVideo: React.FC<MaskedVideoProps> = ({
  originalSrc,
  maskSrc,
}) => {
  const { width, height } = useVideoConfig();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const originalCanvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);

  // Composite must be declared before the frame callbacks that reference it
  const compositeFrames = useCallback(() => {
    const canvas = canvasRef.current;
    const originalCanvas = originalCanvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    if (!canvas || !originalCanvas || !maskCanvas) return;

    const ctx = canvas.getContext("2d");
    const origCtx = originalCanvas.getContext("2d");
    const maskCtx = maskCanvas.getContext("2d");
    if (!ctx || !origCtx || !maskCtx) return;

    const origData = origCtx.getImageData(0, 0, width, height);
    const maskData = maskCtx.getImageData(0, 0, width, height);
    const pixels = origData.data;
    const maskPixels = maskData.data;

    // Set alpha channel from mask luminance (R channel of grayscale)
    for (let i = 0; i < pixels.length; i += 4) {
      pixels[i + 3] = maskPixels[i];
    }

    ctx.putImageData(origData, 0, 0);
  }, [width, height]);

  const onOriginalFrame = useCallback(
    (frame: CanvasImageSource) => {
      const originalCanvas = originalCanvasRef.current;
      if (!originalCanvas) return;
      const ctx = originalCanvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(frame, 0, 0, width, height);
      compositeFrames();
    },
    [width, height, compositeFrames]
  );

  const onMaskFrame = useCallback(
    (frame: CanvasImageSource) => {
      const maskCanvas = maskCanvasRef.current;
      if (!maskCanvas) return;
      const ctx = maskCanvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(frame, 0, 0, width, height);
      compositeFrames();
    },
    [width, height, compositeFrames]
  );

  return (
    <AbsoluteFill>
      {/* Hidden canvases for reading video frames */}
      <canvas
        ref={originalCanvasRef}
        width={width}
        height={height}
        style={{ display: "none" }}
      />
      <canvas
        ref={maskCanvasRef}
        width={width}
        height={height}
        style={{ display: "none" }}
      />

      {/* Hidden video elements with frame callbacks */}
      <OffthreadVideo
        src={originalSrc}
        style={{ display: "none" }}
        onVideoFrame={onOriginalFrame}
      />
      <OffthreadVideo
        src={maskSrc}
        style={{ display: "none" }}
        onVideoFrame={onMaskFrame}
      />

      {/* Visible composite canvas */}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ width: "100%", height: "100%" }}
      />
    </AbsoluteFill>
  );
};
