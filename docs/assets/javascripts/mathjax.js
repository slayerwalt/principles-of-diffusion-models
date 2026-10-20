/* The renderer version is pinned in scripts/vendor_mathjax.py. */
window.MathJax = {
  loader: {
    load: ["[tex]/mathtools", "[tex]/textmacros"]
  },
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
    packages: { "[+]": ["mathtools", "textmacros"] },
    tags: "ams",
    macros: {
      // The source uses stmaryrd's short arrow for time intervals.
      shortrightarrow: "\\rightarrow"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  chtml: {
    scale: 1,
    matchFontHeight: false
  }
};
