const API_BASE = () => (typeof window.API_BASE === 'string' ? window.API_BASE : 'https://mc-suggests.onrender.com');

const QUICK_PICKS = ['Solo Leveling', 'Omniscient Reader', 'Tower of God', 'The Beginning After the End', 'Frieren'];

// Results arrive best-first; tiers are by position in that list.
const TIERS = [
  { name: 'S', upTo: 2 },
  { name: 'A', upTo: 5 },
  { name: 'B', upTo: 9 },
  { name: 'C', upTo: Infinity },
];

const STATUS = { completed: 'Completed', ongoing: 'Ongoing', hiatus: 'On hiatus', cancelled: 'Cancelled' };

// Format tags that say nothing about the story
const HIDDEN_TAGS = new Set(['long_strip', 'web_comic', 'full_color', 'adaptation', 'official_colored', 'award_winning']);

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function formatCount(n) {
  if (n == null) return null;
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k`;
  return String(n);
}

async function fetchJson(path, params) {
  const res = await fetch(`${API_BASE()}${path}?${new URLSearchParams(params)}`);
  if (!res.ok) throw new Error(`the server answered ${res.status}`);
  return res.json();
}

function coverSrc(url) {
  if (!url) return null;
  return /^https?:\/\//.test(url) ? url : `${API_BASE()}${url}`;
}

function coverHtml(item) {
  const src = coverSrc(item.cover_url);
  const fallback = `<span class="cover-fallback">${escapeHtml(item.title)}</span>`;
  if (!src) return fallback;
  return `<img src="${escapeHtml(src)}" alt="" loading="lazy" referrerpolicy="no-referrer" />${fallback}`;
}

function tierRowsHtml(results) {
  let start = 0;
  return TIERS.map(tier => {
    const items = results.slice(start, Math.min(tier.upTo, results.length));
    const offset = start;
    start += items.length;
    if (!items.length) return '';
    return `
      <div class="tier tier-${tier.name}">
        <div class="tier-label">${tier.name}</div>
        <div class="tier-items">
          ${items.map((item, j) => `
            <button type="button" class="pick" data-index="${offset + j}" aria-label="${escapeHtml(item.title)}">
              <span class="cover">${coverHtml(item)}</span>
              <span class="pick-title">${escapeHtml(item.title)}</span>
            </button>`).join('')}
        </div>
      </div>`;
  }).join('');
}

function tierOf(index) {
  return TIERS.find(t => index < t.upTo).name;
}

function detailHtml(item, index) {
  const facts = [
    item.type ? item.type[0].toUpperCase() + item.type.slice(1) : null,
    item.year,
    item.rating ? `★ ${item.rating.toFixed(1)}` : null,
    item.chapters ? `${item.chapters} chapters` : null,
    item.status ? (STATUS[item.status] || item.status) : null,
    item.follows ? `${formatCount(item.follows)} follows` : null,
  ].filter(Boolean);
  const tags = (item.tags || []).filter(t => !HIDDEN_TAGS.has(t)).slice(0, 6);
  return `
    <span class="cover">${coverHtml(item)}</span>
    <div class="detail-body">
      <p class="detail-tier"><span class="tier-chip tier-${tierOf(index)}">${tierOf(index)}</span> #${index + 1} pick · ${Math.round(item.score * 100)}% match</p>
      <h2>${escapeHtml(item.title)}</h2>
      <p class="facts">${facts.map(escapeHtml).join(' · ')}</p>
      ${item.description ? `<p class="synopsis">${escapeHtml(item.description)}</p>` : ''}
      ${tags.length ? `<p class="tags">${tags.map(t => `<span>${escapeHtml(t.replace(/_/g, ' '))}</span>`).join('')}</p>` : ''}
      ${item.url ? `<a class="read" href="${escapeHtml(item.url)}" target="_blank" rel="noopener">Read on MangaDex →</a>` : ''}
    </div>`;
}

function wireCovers(root) {
  root.querySelectorAll('.cover img').forEach(img => {
    img.addEventListener('error', () => img.remove(), { once: true });
  });
}

function skeletonHtml() {
  return TIERS.map((t, i) => `
    <div class="tier tier-${t.name} loading">
      <div class="tier-label">${t.name}</div>
      <div class="tier-items">${'<span class="pick"><span class="cover"></span></span>'.repeat([2, 3, 4, 3][i])}</div>
    </div>`).join('');
}

window.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('searchForm');
  const q = document.getElementById('query');
  const go = document.getElementById('go');
  const tiers = document.getElementById('tiers');
  const detail = document.getElementById('detail');
  const status = document.getElementById('status');
  const alpha = document.getElementById('alpha');
  const suggestions = document.getElementById('suggestions');
  const quickPicks = document.getElementById('quickPicks');

  let results = [];

  QUICK_PICKS.forEach((title, i) => {
    const link = document.createElement('button');
    link.type = 'button';
    link.className = 'link';
    link.textContent = title;
    link.addEventListener('click', () => { q.value = title; run(); });
    quickPicks.appendChild(link);
    if (i < QUICK_PICKS.length - 1) quickPicks.appendChild(document.createTextNode(i === QUICK_PICKS.length - 2 ? ' or ' : ', '));
  });

  function select(index, scroll) {
    tiers.querySelectorAll('.pick').forEach(el => el.classList.toggle('selected', Number(el.dataset.index) === index));
    detail.innerHTML = detailHtml(results[index], index);
    detail.hidden = false;
    wireCovers(detail);
    if (scroll) detail.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  tiers.addEventListener('click', e => {
    const pick = e.target.closest('.pick[data-index]');
    if (pick) select(Number(pick.dataset.index), true);
  });

  async function run() {
    const query = q.value.trim();
    if (!query) { q.focus(); return; }
    hideSuggestions();
    go.disabled = true;
    detail.hidden = true;
    tiers.hidden = false;
    tiers.innerHTML = skeletonHtml();
    status.className = 'status';
    status.textContent = `Ranking titles like “${query}”…`;
    // Free hosting sleeps when idle, so the first request can take a while.
    const slow = setTimeout(() => {
      status.textContent = 'Waking the server up. The first search after a quiet spell can take up to a minute.';
    }, 5000);
    try {
      const data = await fetchJson('/recommend/hybrid', { query, k: 12, alpha: alpha.value });
      if (!data.seed_count) {
        tiers.hidden = true;
        status.className = 'status warn';
        status.textContent = `Couldn't find “${query}”. Check the spelling, or pick a title from the suggestions as you type.`;
        return;
      }
      results = data.results;
      const seed = (data.seeds && data.seeds[0]) || query;
      status.innerHTML = `If you liked <strong>${escapeHtml(seed)}</strong>, here's your list:`;
      tiers.innerHTML = tierRowsHtml(results);
      wireCovers(tiers);
      if (results.length) select(0, false);
    } catch (e) {
      tiers.hidden = true;
      status.className = 'status error';
      status.textContent = `Something went wrong (${e.message}). Try again in a moment.`;
    } finally {
      clearTimeout(slow);
      go.disabled = false;
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
    suggestions.querySelectorAll('li').forEach((li, j) => li.setAttribute('aria-selected', String(j === i)));
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
            <small>${escapeHtml([it.type, it.year].filter(Boolean).join(', '))}</small>
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
