import { useEffect, useState } from 'react';
import { VIDEO_STEPS } from '../utils/constants';
import './ProgressTracker.css';

export default function ProgressTracker({ currentStep, transcription }) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const started = Date.now();
    const timer = setInterval(() => setSeconds(Math.round((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <section className="progress" aria-labelledby="progress-title">
      <div className="progress-head">
        <h2 id="progress-title" className="progress-title">Checking the video</h2>
        <span className="progress-time">{seconds}s</span>
      </div>

      <ol className="progress-steps" aria-live="polite">
        {VIDEO_STEPS.map((label, i) => {
          const step = i + 1;
          const state = step < currentStep ? 'done' : step === currentStep ? 'active' : 'pending';
          return (
            <li key={label} className={`progress-step is-${state}`} aria-current={state === 'active' ? 'step' : undefined}>
              <span className="progress-step-marker" aria-hidden="true">{step}</span>
              <span className="progress-step-label">{label}</span>
              <span className="sr-only">
                {state === 'done' ? ', done' : state === 'active' ? ', in progress' : ''}
              </span>
            </li>
          );
        })}
      </ol>

      {transcription && (
        <div className="progress-transcript">
          <h3 className="progress-transcript-title">Transcript</h3>
          <p>{transcription}</p>
        </div>
      )}
    </section>
  );
}
