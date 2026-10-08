const API_BASE = () => (typeof window.API_BASE === 'string' ? window.API_BASE : 'https://mc-suggests.onrender.com');

const QUICK_PICKS = ['Solo Leveling', 'Omniscient Reader', 'Tower of God', 'The Beginning After the End', 'Frieren', 'Lookism'];

const PLACEHOLDER_COVER = 'data:image/svg+xml;base64,' + btoa(
  '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400">' +
  '<rect width="300" height="400" fill="#0d1830"/>' +
  '<text x="150" y="210" text-anchor="middle" fill="#4cc9ff" font-family="sans-serif" font-size="28" opacity=".6">NO COVER</text></svg>'
);

const STATUS = {
  completed: 'Completed',
  ongoing: 'Ongoing',
  hiatus: 'Hiatus',
  cancelled: 'Cancelled',
};

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

// Results are already sorted best-first; ranks are by position, like a raid party lineup.
function rankFor(index) {
  if (index < 2) return 'S';
  if (index < 5) return 'A';
  if (index < 9) return 'B';
  return 'C';
}

function prettyTag(tag) {
  return tag.replace(/_/g, ' ');
}

function formatCount(n) {
  if (n == null) return '';
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k`;
  return String(n);
}

async function fetchJson(path, params) {
  const url = `${API_BASE()}${path}?${new URLSearchParams(params)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

function coverSrc(url) {
  if (!url) return PLACEHOLDER_COVER;
  return /^https?:\/\//.test(url) ? url : `${API_BASE()}${url}`;
}

function cardHtml(item, index) {
  const rank = rankFor(index);
  const match = Math.round(Math.max(0, Math.min(1, item.score)) * 100);
  const tags = (item.tags || [])
    .filter(t => !['long_strip', 'web_comic', 'full_color', 'adaptation'].includes(t))
    .slice(0, 4);
  const meta = [
    item.year,
    item.rating ? `★ ${item.rating.toFixed(1)}` : null,
    item.chapters ? `${item.chapters} ch` : null,
    item.follows ? `${formatCount(item.follows)} follows` : null,
  ].filter(Boolean);
  const status = item.status ? (STATUS[item.status] || item.status) : null;

  return `
    <article class="card rank-${rank}" style="--i:${index}">
      <div class="cover">
        <img src="${escapeHtml(coverSrc(item.cover_url))}" alt="" loading="lazy" referrerpolicy="no-referrer" />
        <span class="rank" title="Rank ${rank}">${rank}</span>
        ${item.type ? `<span class="type type-${escapeHtml(item.type)}">${escapeHtml(item.type)}</span>` : ''}
      </div>
      <div class="card-body">
        <h3 class="title">${escapeHtml(item.title)}</h3>
        <div class="meta">${meta.map(escapeHtml).join('<i>·</i>')}</div>
        ${status ? `<span class="status-badge status-${escapeHtml(item.status)}">${escapeHtml(status)}</span>` : ''}
        ${tags.length ? `<div class="tags">${tags.map(t => `<span>${escapeHtml(prettyTag(t))}</span>`).join('')}</div>` : ''}
        ${item.description ? `<p class="desc">${escapeHtml(item.description)}</p>` : ''}
        <div class="card-foot">
          <div class="match" title="Similarity to your pick">
            <div class="match-bar"><span style="width:${match}%"></span></div>
            <span class="match-label">${match}% match</span>
          </div>
          ${item.url ? `<a class="read" href="${escapeHtml(item.url)}" target="_blank" rel="noopener">Read ↗</a>` : ''}
        </div>
      </div>
    </article>`;
}

function renderResults(root, data) {
  root.innerHTML = data.results.map(cardHtml).join('');
  root.querySelectorAll('.cover img').forEach(img => {
    img.addEventListener('error', () => { img.src = PLACEHOLDER_COVER; }, { once: true });
  });
}

function renderSkeletons(root, n = 6) {
  root.innerHTML = Array.from({ length: n }, () => '<div class="card skeleton"><div class="cover"></div><div class="card-body"><i></i><i></i><i></i></div></div>').join('');
}

function setStatus(el, html, kind = 'info') {
  el.className = `status-line ${kind}`;
  el.innerHTML = html;
}

window.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('questForm');
  const q = document.getElementById('query');
  const go = document.getElementById('go');
  const goLabel = go.querySelector('.arise-label');
  const results = document.getElementById('results');
  const status = document.getElementById('status');
  const alpha = document.getElementById('alpha');
  const suggestions = document.getElementById('suggestions');
  const quickPicks = document.getElementById('quickPicks');

  QUICK_PICKS.forEach(title => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.textContent = title;
    chip.addEventListener('click', () => { q.value = title; run(); });
    quickPicks.appendChild(chip);
  });

  async function run() {
    const query = q.value.trim();
    if (!query) { q.focus(); return; }
    hideSuggestions();
    go.disabled = true;
    goLabel.textContent = 'Scanning…';
    renderSkeletons(results);
    setStatus(status, `<b>[SYSTEM]</b> Scanning the archive for titles like “${escapeHtml(query)}”…`);
    // Free hosting sleeps when idle; let people know why the first search can be slow.
    const slow = setTimeout(() => setStatus(status,
      '<b>[SYSTEM]</b> Waking up the server… free hosting can take up to a minute on the first search.'), 5000);
    try {
      const data = await fetchJson('/recommend/hybrid', { query, k: 12, alpha: alpha.value });
      clearTimeout(slow);
      if (!data.seed_count) {
        results.innerHTML = '';
        setStatus(status, `<b>[SYSTEM]</b> No title matching “${escapeHtml(query)}” found in the archive. Try another name or pick one below the search bar.`, 'warn');
        return;
      }
      const seeds = data.seeds && data.seeds.length ? data.seeds : [query];
      const extra = seeds.length > 1 ? ` <span class="dim">(+${seeds.length - 1} related)</span>` : '';
      setStatus(status, `<b>[QUEST COMPLETE]</b> Because you liked <em>${escapeHtml(seeds[0])}</em>${extra}: ${data.results.length} recommendations unlocked.`, 'ok');
      renderResults(results, data);
    } catch (e) {
      clearTimeout(slow);
      results.innerHTML = '';
      setStatus(status, `<b>[ERROR]</b> Quest failed: ${escapeHtml(e.message)}. Try again in a moment.`, 'error');
    } finally {
      go.disabled = false;
      goLabel.textContent = 'Accept Quest';
    }
  }

  form.addEventListener('submit', e => { e.preventDefault(); run(); });

  // Title autocomplete
  let debounce;
  let active = -1;
  let requestId = 0;

  function hideSuggestions() {
    suggestions.hidden = true;
    suggestions.innerHTML = '';
    active = -1;
  }

  function highlight(i) {
    const items = suggestions.querySelectorAll('li');
    items.forEach((li, j) => li.setAttribute('aria-selected', String(j === i)));
    active = i;
  }

  q.addEventListener('input', () => {
    clearTimeout(debounce);
    const term = q.value.trim();
    if (term.length < 2) { hideSuggestions(); return; }
    debounce = setTimeout(async () => {
      const id = ++requestId;
      try {
        const items = await fetchJson('/titles', { q: term, k: 8 });
        if (id !== requestId) return;
        if (!items.length) { hideSuggestions(); return; }
        suggestions.innerHTML = items.map(it => `
          <li role="option" data-title="${escapeHtml(it.title)}">
            <span>${escapeHtml(it.title)}</span>
            <small>${escapeHtml([it.type, it.year].filter(Boolean).join(' · '))}</small>
          </li>`).join('');
        suggestions.hidden = false;
        active = -1;
      } catch {
        hideSuggestions();
      }
    }, 200);
  });

  suggestions.addEventListener('mousedown', e => {
    const li = e.target.closest('li');
    if (!li) return;
    e.preventDefault();
    q.value = li.dataset.title;
    run();
  });

  q.addEventListener('keydown', e => {
    const items = suggestions.querySelectorAll('li');
    if (suggestions.hidden || !items.length) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); highlight((active + 1) % items.length); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); highlight((active - 1 + items.length) % items.length); }
    else if (e.key === 'Enter' && active >= 0) { q.value = items[active].dataset.title; }
    else if (e.key === 'Escape') { hideSuggestions(); }
  });

  q.addEventListener('blur', () => setTimeout(hideSuggestions, 100));
});
