// Initialize Mermaid diagrams for Material for MkDocs
document$.subscribe(() => {
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({ 
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
      },
      themeVariables: {
        primaryColor: '#bb2528',
        primaryTextColor: '#fff',
        primaryBorderColor: '#7C0000',
        lineColor: '#b0bec5',
        sectionBkColor: '#f5f5f5',
        altSectionBkColor: '#ffffff',
        gridColor: '#b0bec5'
      }
    });
    
    // Process all mermaid diagrams on the page
    const mermaidDivs = document.querySelectorAll('.mermaid');
    mermaidDivs.forEach((div, index) => {
      try {
        mermaid.render(`mermaid-${index}`, div.textContent, (svg) => {
          div.innerHTML = svg;
        });
      } catch (error) {
        console.warn('Mermaid rendering error:', error);
        div.innerHTML = `<pre>${div.textContent}</pre>`;
      }
    });
  }
});