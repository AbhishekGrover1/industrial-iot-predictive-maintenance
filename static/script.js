/* Predictive maintenance console.
 * Pointer lighting and tilt, input validation, and the call to /predict. */
(() => {
  'use strict';

  const API_URL = location.protocol === 'file:' ? 'http://127.0.0.1:8000/predict' : '/predict';
  const REQUEST_DEADLINE_MS = 90000; // a spun-down free instance can take about a minute to wake
  const RETRY_DELAY_MS = 2500;
  const SLOW_RESPONSE_MS = 6000;
  const MIN_BUSY_MS = 300;
  const STARTING_STATUSES = new Set([502, 503, 504]);

  const FIELD_LABELS = {
    type: 'Machine type',
    air_temperature_k: 'Air temperature',
    process_temperature_k: 'Process temperature',
    rotational_speed_rpm: 'Rotational speed',
    torque_nm: 'Torque',
    tool_wear_min: 'Tool wear',
  };

  const root = document.documentElement;
  const scene = document.getElementById('scene');
  const plate = document.getElementById('plate');
  const form = document.getElementById('form');
  const keyWrap = document.getElementById('key-wrap');
  const key = document.getElementById('key');
  const keyLabel = key.querySelector('.key__label');
  const readout = document.getElementById('readout');
  const diagnosis = document.getElementById('diagnosis');
  const verdict = document.getElementById('verdict');
  const confidence = document.getElementById('confidence');
  const meter = document.getElementById('meter');
  const notice = document.getElementById('notice');
  const numericInputs = [...form.querySelectorAll('input[type="number"]')];

  const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const percent = new Intl.NumberFormat(undefined, {
    style: 'percent',
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  /* ---------- Pointer light and plate tilt ---------- */

  const pointer = { x: window.innerWidth / 2, y: window.innerHeight * 0.22 };
  const light = { x: pointer.x, y: pointer.y };
  const tilt = { x: 0, y: 0, targetX: 0, targetY: 0 };
  let frame = 0;

  function requestFrame() {
    if (!frame) frame = requestAnimationFrame(render);
  }

  function render() {
    frame = 0;

    // Ease towards the targets so the light trails the pointer slightly.
    light.x += (pointer.x - light.x) * 0.18;
    light.y += (pointer.y - light.y) * 0.18;
    tilt.x += (tilt.targetX - tilt.x) * 0.12;
    tilt.y += (tilt.targetY - tilt.y) * 0.12;

    const box = scene.getBoundingClientRect();
    root.style.setProperty('--mx', `${light.x.toFixed(1)}px`);
    root.style.setProperty('--my', `${light.y.toFixed(1)}px`);
    plate.style.setProperty('--cx', `${(light.x - box.left).toFixed(1)}px`);
    plate.style.setProperty('--cy', `${(light.y - box.top).toFixed(1)}px`);
    plate.style.setProperty('--tx', tilt.x.toFixed(4));
    plate.style.setProperty('--ty', tilt.y.toFixed(4));

    const settled =
      Math.abs(pointer.x - light.x) < 0.1 &&
      Math.abs(pointer.y - light.y) < 0.1 &&
      Math.abs(tilt.targetX - tilt.x) < 0.001 &&
      Math.abs(tilt.targetY - tilt.y) < 0.001;
    if (!settled) requestFrame();
  }

  window.addEventListener(
    'pointermove',
    (event) => {
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      requestFrame();
    },
    { passive: true },
  );

  scene.addEventListener(
    'pointermove',
    (event) => {
      if (!finePointer.matches || reducedMotion.matches) return;
      const box = scene.getBoundingClientRect();
      tilt.targetX = clamp(((event.clientX - box.left) / box.width) * 2 - 1, -1, 1);
      tilt.targetY = clamp(((event.clientY - box.top) / box.height) * 2 - 1, -1, 1);
      plate.dataset.hover = 'true';
      requestFrame();
    },
    { passive: true },
  );

  scene.addEventListener('pointerleave', () => {
    tilt.targetX = 0;
    tilt.targetY = 0;
    plate.dataset.hover = 'false';
    requestFrame();
  });

  window.addEventListener('resize', requestFrame);

  /* ---------- Key: highlight and tilt follow the pointer ---------- */

  keyWrap.addEventListener(
    'pointermove',
    (event) => {
      if (!finePointer.matches) return;
      const box = keyWrap.getBoundingClientRect();
      const x = clamp((event.clientX - box.left) / box.width, 0, 1);
      const y = clamp((event.clientY - box.top) / box.height, 0, 1);
      key.style.setProperty('--kx', `${(x * 100).toFixed(1)}%`);
      key.style.setProperty('--ky', `${(y * 100).toFixed(1)}%`);
      if (!reducedMotion.matches) {
        key.style.setProperty('--ktx', (x * 2 - 1).toFixed(3));
        key.style.setProperty('--kty', (y * 2 - 1).toFixed(3));
      }
    },
    { passive: true },
  );

  keyWrap.addEventListener('pointerleave', () => {
    key.style.setProperty('--ktx', '0');
    key.style.setProperty('--kty', '0');
  });

  /* ---------- Validation ---------- */

  const labelOf = (input) => form.querySelector(`label[for="${input.id}"]`).textContent.trim();
  const unitOf = (input) => input.closest('.control').querySelector('.unit').textContent.trim();

  /** Marks every invalid field and returns a message for the first one, or null. */
  function validate() {
    let message = null;
    for (const input of numericInputs) {
      const control = input.closest('.control');
      delete control.dataset.invalid;
      const { validity } = input;
      if (validity.valid) continue;

      control.dataset.invalid = 'true';
      if (message) continue;
      input.focus();
      const name = labelOf(input);
      if (validity.valueMissing || validity.badInput) {
        message = `${name} needs a number.`;
      } else if (validity.stepMismatch) {
        message = `${name} must be a whole number.`;
      } else {
        message = `${name} must be between ${input.min} and ${input.max} ${unitOf(input)}.`;
      }
    }
    return message;
  }

  form.addEventListener('input', (event) => {
    const control = event.target.closest('.control');
    if (control) delete control.dataset.invalid;
  });

  /* ---------- Request ---------- */

  class ServiceError extends Error {}

  function readPayload() {
    const payload = { type: form.elements.namedItem('type').value };
    for (const input of numericInputs) payload[input.name] = Number(input.value);
    return payload;
  }

  function describe(status, data) {
    const detail = data && data.detail;
    if (Array.isArray(detail) && detail.length) {
      const { loc = [], msg = 'is invalid' } = detail[0];
      const field = FIELD_LABELS[loc[loc.length - 1]] || 'Input';
      return `${field}: ${msg}.`;
    }
    if (typeof detail === 'string') return detail;
    return `The service returned an error (${status}).`;
  }

  async function requestDiagnosis(payload) {
    const deadline = Date.now() + REQUEST_DEADLINE_MS;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_DEADLINE_MS);
    try {
      for (;;) {
        const response = await fetch(API_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });

        // A host that is still starting up answers 502/503/504 until the app is ready.
        if (STARTING_STATUSES.has(response.status) && Date.now() + RETRY_DELAY_MS < deadline) {
          await pause(RETRY_DELAY_MS);
          continue;
        }

        const data = await response.json().catch(() => null);
        if (!response.ok) throw new ServiceError(describe(response.status, data));
        return data;
      }
    } finally {
      clearTimeout(timer);
    }
  }

  function explain(error) {
    if (error instanceof ServiceError) return error.message;
    if (error && error.name === 'AbortError') return 'The request timed out. Try again.';
    return 'The service is unreachable. Check that the server is running.';
  }

  /* ---------- Readout ---------- */

  const setState = (state) => {
    document.body.dataset.state = state;
  };

  function showDiagnosis(result) {
    const firstReading = readout.dataset.open !== 'true';
    notice.hidden = true;
    diagnosis.hidden = false;
    verdict.textContent = result.prediction;
    confidence.textContent = percent.format(result.confidence_score);
    setState(result.prediction === 'No Failure' ? 'ok' : 'fault');

    if (firstReading) {
      meter.style.setProperty('--p', '0');
      void meter.offsetWidth; // commit the empty meter so the fill animates in
    }
    meter.style.setProperty('--p', String(result.confidence_score));
    readout.dataset.open = 'true';
  }

  function showNotice(message) {
    diagnosis.hidden = true;
    notice.hidden = false;
    notice.textContent = message;
    setState('idle');
    readout.dataset.open = 'true';
  }

  let busy = false;

  function setBusy(isBusy) {
    busy = isBusy;
    key.setAttribute('aria-busy', String(isBusy));
    form.setAttribute('aria-busy', String(isBusy));
    keyLabel.textContent = isBusy ? 'Running diagnostics' : 'Run Diagnostics';
  }

  const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busy) return;

    const problem = validate();
    if (problem) {
      showNotice(problem);
      return;
    }

    setBusy(true);
    const slowTimer = setTimeout(() => {
      keyLabel.textContent = 'Waiting for the server';
    }, SLOW_RESPONSE_MS);
    try {
      // A short minimum keeps the busy state from flashing on fast responses.
      const [result] = await Promise.all([requestDiagnosis(readPayload()), pause(MIN_BUSY_MS)]);
      showDiagnosis(result);
    } catch (error) {
      showNotice(explain(error));
    } finally {
      clearTimeout(slowTimer);
      setBusy(false);
    }
  });

  render();
})();
