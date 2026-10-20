/* Switch book languages at the corresponding page and reading anchor. */
(() => {
  "use strict";

  function synchronizeFragments() {
    document.querySelectorAll("[data-language-pair]").forEach(link => {
      const target = new URL(link.href);
      target.hash = location.hash;
      link.href = target.href;
    });
  }

  function initializeLanguageMenus() {
    synchronizeFragments();
    document.querySelectorAll("[data-language-menu]").forEach(menu => {
      if (menu.dataset.languageMenuReady) return;
      menu.dataset.languageMenuReady = "true";

      const trigger = menu.querySelector("summary");
      const links = [...menu.querySelectorAll("a[href]")];
      const close = () => { menu.open = false; };

      document.addEventListener("pointerdown", event => {
        if (menu.open && !menu.contains(event.target)) close();
      });

      menu.addEventListener("focusout", () => {
        requestAnimationFrame(() => {
          if (menu.open && !menu.contains(document.activeElement)) close();
        });
      });

      menu.addEventListener("keydown", event => {
        if (event.key === "Escape" && menu.open) {
          event.preventDefault();
          event.stopPropagation();
          close();
          trigger.focus();
          return;
        }

        if (!["ArrowDown", "ArrowUp"].includes(event.key) || !links.length) return;
        event.preventDefault();
        menu.open = true;
        const current = links.indexOf(document.activeElement);
        const next = event.key === "ArrowDown"
          ? (current + 1) % links.length
          : (current <= 0 ? links.length - 1 : current - 1);
        links[next].focus();
      });

      links.forEach(link => link.addEventListener("click", event => {
        // The theme's language handler treats hreflang URLs as site roots and
        // can replace our page-specific destination, dropping its fragment.
        event.stopPropagation();
        synchronizeFragments();
        if (event.defaultPrevented || event.button !== 0 ||
            event.metaKey || event.ctrlKey || event.shiftKey || event.altKey ||
            link.hasAttribute("download") ||
            (link.target && link.target !== "_self")) return;
        event.preventDefault();
        const destination = link.href;
        close();
        window.location.assign(destination);
      }));
    });

    if (document.documentElement.lang.startsWith("zh")) {
      const back = document.querySelector(".book-reading-back");
      if (back) {
        back.title = "返回之前的阅读位置";
        const label = back.querySelector("span");
        if (label) label.textContent = "返回阅读";
      }
    }
  }

  window.addEventListener("hashchange", synchronizeFragments);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeLanguageMenus);
  } else {
    initializeLanguageMenus();
  }
})();
