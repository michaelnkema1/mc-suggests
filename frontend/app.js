// Cache bust: 2024-01-04
async function fetchRecs(query, k = 10, alpha = 0.85) {
  // Always use hybrid model
  const base = (typeof window.API_BASE === 'string') ? window.API_BASE : 'https://mc-suggests.onrender.com';
  const params = new URLSearchParams({ query, k: String(k), alpha: String(alpha) });
  const url = `${base}/recommend/hybrid?${params.toString()}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

function getStatusDisplay(status) {
  if (!status) return '';
  
  const statusMap = {
    'completed': { icon: '✅', text: 'Completed', class: 'status-completed' },
    'ongoing': { icon: '🔄', text: 'Ongoing', class: 'status-ongoing' },
    'hiatus': { icon: '⏸️', text: 'Hiatus', class: 'status-hiatus' },
    'cancelled': { icon: '❌', text: 'Cancelled', class: 'status-cancelled' }
  };
  
  const statusInfo = statusMap[status.toLowerCase()] || { icon: '❓', text: status, class: 'status-unknown' };
  return `<span class="status-badge ${statusInfo.class}">${statusInfo.icon} ${statusInfo.text}</span>`;
}

function renderResults(root, data) {
  root.innerHTML = '';
  data.results.forEach(item => {
    const el = document.createElement('div');
    el.className = 'card';
    
    const img = document.createElement('img');
    img.src = item.cover_url ? `${base}${item.cover_url}` : 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iODAiIHZpZXdCb3g9IjAgMCA2MCA4MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjYwIiBoZWlnaHQ9IjgwIiBmaWxsPSIjMzAzNjNkIi8+CjxwYXRoIGQ9Ik0yMCAyNUg0MFY1NUgyMFYyNVoiIGZpbGw9IiM2Yzc1N2QiLz4KPC9zdmc+';
    img.alt = item.title || 'Cover';
    img.onerror = () => { img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iODAiIHZpZXdCb3g9IjAgMCA2MCA4MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjYwIiBoZWlnaHQ9IjgwIiBmaWxsPSIjMzAzNjNkIi8+CjxwYXRoIGQ9Ik0yMCAyNUg0MFY1NUgyMFYyNVoiIGZpbGw9IiM2Yzc1N2QiLz4KPC9zdmc+'; };
    
    const content = document.createElement('div');
    content.className = 'card-content';
    content.innerHTML = `
      <div class="title">${item.title}</div>
      <div class="meta">${item.year ?? ''} ${item.rating ? `• ⭐ ${item.rating.toFixed(2)}` : ''} ${item.chapters ? `• 📖 ${item.chapters} ch` : ''}</div>
      <div class="status">${getStatusDisplay(item.status)}</div>
      <div class="score">score: ${item.score.toFixed(4)}</div>
    `;
    
    el.appendChild(img);
    el.appendChild(content);
    root.appendChild(el);
  });
}

window.addEventListener('DOMContentLoaded', () => {
  const q = document.getElementById('query');
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
      const data = await fetchRecs(query, 12, Number(alpha.value));
      renderResults(results, data);
    } catch (e) {
      results.innerHTML = `<div>Failed to load: ${String(e)}</div>`;
    } finally {
      go.disabled = false;
      go.textContent = 'Recommend';
    }
  });
});


