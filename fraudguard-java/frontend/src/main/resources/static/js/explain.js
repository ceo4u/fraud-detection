document.addEventListener('DOMContentLoaded', () => {
  const providerSelect = document.getElementById('provider-select');
  const demoBtn = document.getElementById('load-demo-btn');
  const probValEl = document.getElementById('provider-prob-val');
  const tierValEl = document.getElementById('provider-tier-val');
  const posDriversList = document.getElementById('pos-drivers-list');
  const negDriversList = document.getElementById('neg-drivers-list');

  function loadExplanation(providerId) {
    if (!providerId) return;

    fetch(`/api/explain/${providerId}`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        const prob = data.model_probability || 0.0;
        if (probValEl) probValEl.textContent = prob.toFixed(4);

        const tier = prob >= 0.278 ? 'High' : (prob >= 0.139 ? 'Medium' : 'Low');
        if (tierValEl) {
          tierValEl.textContent = tier;
          tierValEl.className = tier === 'High' ? 'text-danger' : (tier === 'Medium' ? 'text-warning' : 'text-success');
        }

        // Split positive and negative drivers
        if (data.top_drivers) {
          const pos = data.top_drivers.filter(d => d.shap > 0);
          const neg = data.top_drivers.filter(d => d.shap <= 0);

          if (posDriversList) {
            posDriversList.innerHTML = '';
            if (pos.length === 0) {
              posDriversList.innerHTML = '<p class="text-muted">None observed.</p>';
            } else {
              pos.slice(0, 5).forEach(d => {
                const item = document.createElement('div');
                item.className = 'st-factor-item';
                item.innerHTML = `
                  <div class="factor-name"><strong>${d.feature}</strong>: <code>${d.value.toLocaleString(undefined, {maximumFractionDigits: 2})}</code></div>
                  <div class="factor-shap pos">+${d.shap.toFixed(4)} SHAP</div>
                `;
                posDriversList.appendChild(item);
              });
            }
          }

          if (negDriversList) {
            negDriversList.innerHTML = '';
            if (neg.length === 0) {
              negDriversList.innerHTML = '<p class="text-muted">None observed.</p>';
            } else {
              neg.slice(0, 5).forEach(d => {
                const item = document.createElement('div');
                item.className = 'st-factor-item';
                item.innerHTML = `
                  <div class="factor-name"><strong>${d.feature}</strong>: <code>${d.value.toLocaleString(undefined, {maximumFractionDigits: 2})}</code></div>
                  <div class="factor-shap neg">${d.shap.toFixed(4)} SHAP</div>
                `;
                negDriversList.appendChild(item);
              });
            }
          }
        }

        // Render Plotly Waterfall (Light Theme matching Streamlit)
        if (window.Plotly && data.top_drivers) {
          const drivers = data.top_drivers.slice(0, 10).reverse();
          const yLabels = drivers.map(d => d.feature);
          const xValues = drivers.map(d => d.shap);
          const colors = xValues.map(v => v > 0 ? '#ff4b4b' : '#2ecc71');

          Plotly.newPlot('waterfall', [{
            type: 'bar',
            orientation: 'h',
            y: yLabels,
            x: xValues,
            marker: { color: colors, opacity: 0.85 }
          }], {
            paper_bgcolor: '#ffffff',
            plot_bgcolor: '#ffffff',
            font: { color: '#31333f', family: 'Source Sans 3, sans-serif' },
            xaxis: {
              gridcolor: '#f0f2f6',
              zerolinecolor: '#d1d5db',
              title: { text: 'SHAP Impact (Log-odds)', font: { size: 12 } }
            },
            yaxis: { automargin: true },
            margin: { l: 180, r: 24, t: 20, b: 50 },
            height: 380
          }, { responsive: true, displayModeBar: false });
        }
      })
      .catch(err => {
        console.error('Failed to load explanation:', err);
      });
  }

  if (providerSelect) {
    providerSelect.addEventListener('change', () => {
      loadExplanation(providerSelect.value);
    });
    // Initial load
    loadExplanation(providerSelect.value);
  }

  if (demoBtn) {
    demoBtn.addEventListener('click', () => {
      fetch('/api/demo-provider')
        .then(res => res.json())
        .then(demo => {
          if (demo && demo.provider) {
            if (providerSelect) {
              providerSelect.value = demo.provider;
            }
            loadExplanation(demo.provider);
          }
        })
        .catch(err => console.error('Error fetching demo provider:', err));
    });
  }
});
