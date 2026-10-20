document.addEventListener('DOMContentLoaded', () => {
  const searchInput = document.getElementById('search-input');
  const tierSelect = document.getElementById('tier-select');
  const tableBody = document.getElementById('providers-tbody');

  function filterTable() {
    if (!tableBody) return;
    const term = (searchInput ? searchInput.value : '').toLowerCase().trim();
    const tier = (tierSelect ? tierSelect.value : '').toLowerCase().trim();

    const rows = tableBody.querySelectorAll('tr');
    rows.forEach(row => {
      const providerCode = row.querySelector('code') ? row.querySelector('code').textContent.toLowerCase() : '';
      const tierPill = row.querySelector('.fd-pill') ? row.querySelector('.fd-pill').textContent.toLowerCase() : '';

      const matchSearch = !term || providerCode.includes(term);
      const matchTier = !tier || tier === 'all' || tierPill === tier;

      if (matchSearch && matchTier) {
        row.style.display = '';
      } else {
        row.style.display = 'none';
      }
    });
  }

  if (searchInput) {
    searchInput.addEventListener('input', filterTable);
  }
  if (tierSelect) {
    tierSelect.addEventListener('change', filterTable);
  }
});
