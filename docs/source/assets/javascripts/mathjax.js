// MathJax configuration for mathematical expressions
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})

// Custom JavaScript for enhanced functionality
document.addEventListener("DOMContentLoaded", function() {
    // Add copy button functionality to code blocks
    document.querySelectorAll("pre > code").forEach((block) => {
        // Only add to blocks without existing copy button
        if (!block.parentElement.querySelector(".copy-button")) {
            const button = document.createElement("button");
            button.className = "copy-button";
            button.innerHTML = "📋";
            button.title = "Copy to clipboard";
            
            button.addEventListener("click", () => {
                navigator.clipboard.writeText(block.textContent).then(() => {
                    button.innerHTML = "✅";
                    setTimeout(() => {
                        button.innerHTML = "📋";
                    }, 2000);
                });
            });
            
            block.parentElement.appendChild(button);
        }
    });

    // Add smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add external link indicators
    document.querySelectorAll('a[href^="http"]').forEach(link => {
        if (!link.hostname === location.hostname) {
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
            link.innerHTML += ' <small>🔗</small>';
        }
    });
});