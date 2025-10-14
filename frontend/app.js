async function fetchRecs(query, engine, k = 10, alpha = 0.7) {
  // If API_BASE is not set, assume same origin as frontend
  const base = (typeof window.API_BASE === 'string') ? window.API_BASE : '';
  const params = new URLSearchParams({ query, k: String(k) });
  if (engine === 'hybrid') params.set('alpha', String(alpha));
  const url = `${base}/recommend/${engine}?${params.toString()}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

function renderResults(root, data) {
  root.innerHTML = '';
  data.results.forEach(item => {
    const el = document.createElement('div');
    el.className = 'card';
    el.innerHTML = `
      <div class="title">${item.title}</div>
      <div class="meta">${item.year ?? ''} ${item.rating ? `• ⭐ ${item.rating.toFixed(2)}` : ''}</div>
      <div class="score">score: ${item.score.toFixed(4)}</div>
    `;
    root.appendChild(el);
  });
}

window.addEventListener('DOMContentLoaded', () => {
  const q = document.getElementById('query');
  const engine = document.getElementById('engine');
  const go = document.getElementById('go');
  const results = document.getElementById('results');
  const alpha = document.getElementById('alpha');
  const alphaVal = document.getElementById('alphaVal');

  alpha.addEventListener('input', () => {
    alphaVal.textContent = Number(alpha.value).toFixed(2);
  });

  go.addEventListener('click', async () => {
    const query = q.value.trim();
    if (!query) return;
    go.disabled = true;
    go.textContent = 'Loading...';
    try {
      const data = await fetchRecs(query, engine.value, 12, Number(alpha.value));
      renderResults(results, data);
    } catch (e) {
      results.innerHTML = `<div>Failed to load: ${String(e)}</div>`;
    } finally {
      go.disabled = false;
      go.textContent = 'Recommend';
    }
  });
});


