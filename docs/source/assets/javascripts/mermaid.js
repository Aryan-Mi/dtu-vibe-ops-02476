// Mermaid initialization for Material for MkDocs
// This script initializes mermaid diagrams after the page loads

mermaid.initialize({
  startOnLoad: false,
  theme: 'default',
  securityLevel: 'loose',
  flowchart: {
    useMaxWidth: true,
    htmlLabels: true,
    curve: 'basis'
  },
  sequence: {
    useMaxWidth: true
  }
});

// Function to render mermaid diagrams
async function renderMermaidDiagrams() {
  const mermaidDivs = document.querySelectorAll('.mermaid');
  
  for (let i = 0; i < mermaidDivs.length; i++) {
    const div = mermaidDivs[i];
    const graphDefinition = div.textContent.trim();
    
    if (graphDefinition && !div.dataset.processed) {
      try {
        const { svg } = await mermaid.render(`mermaid-diagram-${i}`, graphDefinition);
        div.innerHTML = svg;
        div.dataset.processed = 'true';
      } catch (error) {
        console.error('Mermaid rendering error:', error);
        div.innerHTML = `<pre class="mermaid-error">${graphDefinition}</pre>`;
      }
    }
  }
}

// Run on page load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', renderMermaidDiagrams);
} else {
  renderMermaidDiagrams();
}

// Re-run when navigating with instant loading (Material for MkDocs)
if (typeof document$ !== 'undefined') {
  document$.subscribe(() => {
    renderMermaidDiagrams();
  });
}
