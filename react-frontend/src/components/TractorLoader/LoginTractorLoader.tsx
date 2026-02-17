import "./TractorLoader.css";
import "./LoginTractorLoader.css";

export default function LoginTractorLoader() {
  return (
    <div className="login-tractor-loader">
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

      {/* scrolling ground + hills */}
      <div className="world" />

      {/* tractor (stationary) */}
      <div className="tractor">
        {/* exhaust */}
        <div className="exhaust">
          <div className="smoke smoke-1" />
          <div className="smoke smoke-2" />
          <div className="smoke smoke-3" />
        </div>

        {/* chassis */}
        <div className="tractor-chassis" />

        {/* hood */}
        <div className="tractor-hood">
          <div className="hood-grille" />
        </div>

        {/* cab */}
        <div className="tractor-cab">
          <div className="cab-window cab-window-front" />
          <div className="cab-window cab-window-side" />
          <div className="cab-roof" />
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

      {/* vignette */}
      <div className="vignette" />
    </div>
  );
}
