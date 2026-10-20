document.addEventListener('DOMContentLoaded', () => {
  fetch('/api/metrics')
    .then(res => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then(data => {
      if (!window.Plotly) return;

      const cm = data.confusion_matrix || [[970, 11], [11, 90]];
      const tn = cm[0][0];
      const fp = cm[0][1];
      const fn = cm[1][0];
      const tp = cm[1][1];

      const textMatrix = [
        [`TN — cleared<br><b style="font-size:18px">${tn.toLocaleString()}</b>`,
         `FP — false alarm<br><b style="font-size:18px">${fp.toLocaleString()}</b>`],
        [`FN — missed fraud ⚠️<br><b style="font-size:18px;color:#d93025">${fn.toLocaleString()}</b>`,
         `TP — caught<br><b style="font-size:18px">${tp.toLocaleString()}</b>`]
      ];

      Plotly.newPlot('confusion-matrix', [{
        type: 'heatmap',
        z: [[tn, fp], [fn, tp]],
        x: ['Predicted Non-Fraud', 'Predicted Fraud'],
        y: ['Actual Non-Fraud', 'Actual Fraud'],
        text: textMatrix,
        texttemplate: '%{text}',
        hoverinfo: 'none',
        colorscale: [
          [0.0, '#f8fafc'],
          [0.5, '#e0f2fe'],
          [1.0, '#38bdf8']
        ],
        showscale: false
      }], {
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        font: { color: '#31333f', family: 'Source Sans 3, sans-serif' },
        margin: { l: 140, r: 24, t: 24, b: 50 },
        height: 350,
        yaxis: { autorange: 'reversed' },
        shapes: [{
          type: 'rect',
          x0: -0.48, x1: 0.48,
          y0: 0.52, y1: 1.48,
          line: { color: '#ff4b4b', width: 3 },
          fillcolor: 'rgba(255, 75, 75, 0.12)'
        }]
      }, { responsive: true, displayModeBar: false });
    })
    .catch(err => {
      console.error('Failed to load metrics:', err);
    });
});
