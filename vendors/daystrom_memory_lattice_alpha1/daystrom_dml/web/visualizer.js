const statusEl = document.querySelector('#visualizer-page-status');
const frameEl = document.querySelector('#visualizer-fullscreen-frame');
const openLinkEl = document.querySelector('#visualizer-open-streamlit');

async function initialiseStandaloneVisualizer() {
  if (!statusEl || !frameEl) {
    return;
  }
  statusEl.textContent = 'Preparing visualizer…';
  try {
    const response = await fetch('/visualizer/launch', { method: 'POST' });
    let payload = {};
    try {
      payload = await response.json();
    } catch (err) {
      payload = {};
    }
    if (!response.ok) {
      const message = payload && payload.detail ? payload.detail : 'Failed to start visualizer';
      throw new Error(message);
    }
    const embedUrl = payload && typeof payload.embed_url === 'string' && payload.embed_url ? payload.embed_url : null;
    const targetUrl = payload && typeof payload.url === 'string' && payload.url ? payload.url : null;

    if (openLinkEl && targetUrl) {
      openLinkEl.href = targetUrl;
    }

    if (embedUrl) {
      frameEl.src = embedUrl;
      statusEl.textContent = 'Visualizer ready.';
    } else if (targetUrl) {
      frameEl.removeAttribute('src');
      statusEl.textContent = 'Visualizer ready. Open the Streamlit visualizer in a new tab.';
    } else {
      frameEl.removeAttribute('src');
      statusEl.textContent = 'Visualizer ready (no URL available).';
    }
  } catch (err) {
    console.error('Visualizer launch failed:', err);
    statusEl.textContent = `Visualizer unavailable: ${err.message}`;
    frameEl.removeAttribute('src');
  }
}

initialiseStandaloneVisualizer();
