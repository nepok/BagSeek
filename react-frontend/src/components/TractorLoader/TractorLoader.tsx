import { useEffect, useRef } from "react";
import "./TractorLoader.css";

interface TractorLoaderProps {
  progress?: number; // 0–100
}

export default function TractorLoader({ progress = 0 }: TractorLoaderProps) {
  const fraction = Math.min(1, Math.max(0, progress));
  const containerRef = useRef<HTMLDivElement>(null);
  const targetRef = useRef(fraction);
  targetRef.current = fraction;

  const stateRef = useRef({
    pos: -1,        // current pixel position (-1 = uninitialised)
    smoothTarget: -1, // smoothed target position
    backRot: 0,
    frontRot: 0,
  });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const BACK_CIRC = Math.PI * 36;   // back wheel circumference in px
    const FRONT_CIRC = Math.PI * 20;   // front wheel circumference in px
    const TARGET_SMOOTH = 0.03;  // how fast the smoothed target follows the real target
    const POS_LERP = 0.09;      // how fast the tractor follows the smoothed target
    const SNAP = 0.3;

    let id: number;
    const tick = () => {
      const w = el.clientWidth;
      const rawTarget = 10 + targetRef.current * (w - 90);
      const s = stateRef.current;

      // First frame: jump to target
      if (s.pos < 0) { s.pos = rawTarget; s.smoothTarget = rawTarget; }

      // Smooth the target so discrete progress jumps become steady motion
      s.smoothTarget += (rawTarget - s.smoothTarget) * TARGET_SMOOTH;

      const prev = s.pos;
      const delta = s.smoothTarget - s.pos;
      s.pos = Math.abs(delta) < SNAP ? s.smoothTarget : s.pos + delta * POS_LERP;

      const moved = s.pos - prev;
      s.backRot += (moved / BACK_CIRC) * 360;
      s.frontRot += (moved / FRONT_CIRC) * 360;

      el.style.setProperty('--tractor-x', `${s.pos}px`);
      el.style.setProperty('--track-w-backwheel', `${s.pos + 15}px`);
      el.style.setProperty('--track-w-frontwheel', `${s.pos + 65}px`);
      el.style.setProperty('--back-rot', `${s.backRot}deg`);
      el.style.setProperty('--front-rot', `${s.frontRot}deg`);
      el.style.setProperty('--bounce-y', Math.abs(moved) > 0.05 ? '-1.5px' : '-0.5px');

      id = requestAnimationFrame(tick);
    };

    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, []);

  return (
    <div className="tractor-loader" ref={containerRef}>
      {/* night sky */}
      <div className="stars">
        <div className="star" style={{ top: '12%', left: '8%', animationDelay: '0s' }} />
        <div className="star" style={{ top: '6%', left: '25%', animationDelay: '1.2s' }} />
        <div className="star" style={{ top: '18%', left: '42%', animationDelay: '0.4s' }} />
        <div className="star" style={{ top: '8%', left: '58%', animationDelay: '2.1s' }} />
        <div className="star" style={{ top: '22%', left: '72%', animationDelay: '0.8s' }} />
        <div className="star" style={{ top: '10%', left: '88%', animationDelay: '1.6s' }} />
        <div className="star" style={{ top: '28%', left: '15%', animationDelay: '2.5s' }} />
        <div className="star" style={{ top: '4%', left: '78%', animationDelay: '0.2s' }} />
      </div>

      {/* scrolling ground */}
      <div className="world" />

      {/* tire tracks behind tractor */}
      <div className="tire-tracks-backwheel" />
      <div className="tire-tracks-frontwheel" />

      {/* tractor */}
      <div className="tractor">
        {/* exhaust */}
        <div className="exhaust">
          <div className="smoke smoke-1" />
          <div className="smoke smoke-2" />
          <div className="smoke smoke-3" />
        </div>

        {/* chassis / frame rail */}
        <div className="tractor-chassis" />

        {/* hood (engine) */}
        <div className="tractor-hood">
          <div className="hood-grille" />
        </div>

        {/* cab */}
        <div className="tractor-cab">
          <div className="cab-window cab-window-front" />
          <div className="cab-window cab-window-side" />
          <div className="cab-roof" />
          {/* lidar on cab roof */}
          <div className="lidar">
            <div className="lidar-beam" />
          </div>
        </div>

        {/* fenders */}
        <div className="tractor-fender-back" />
        <div className="tractor-fender-front" />

        {/* wheels */}
        <div className="wheel wheel-back">
          <div className="wheel-rim" />
          <div className="wheel-spoke" />
          <div className="wheel-spoke spoke-2" />
          <div className="wheel-spoke spoke-3" />
          <div className="wheel-hub" />
        </div>
        <div className="wheel wheel-front">
          <div className="wheel-rim" />
          <div className="wheel-spoke" />
          <div className="wheel-spoke spoke-2" />
          <div className="wheel-hub" />
        </div>
      </div>

      {/* progress percentage */}
      <div className="progress-label">
        {Math.round(fraction * 100)}%
      </div>

      {/* elliptical fade overlay */}
      <div className="vignette" />
    </div>
  );
}
