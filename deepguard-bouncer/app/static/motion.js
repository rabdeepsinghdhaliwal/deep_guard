// Deep-Guard — purely cosmetic scroll-reveal for static chrome.
// Only touches elements marked .reveal, which are never toggled via
// [hidden] by script.js/registry.js/bias-map.js — no interaction with
// app state, no DOM elements it doesn't already see at parse time.
(function () {
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  if (!("IntersectionObserver" in window)) return;

  const targets = document.querySelectorAll(".reveal");
  if (!targets.length) return;

  const io = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-in");
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });

  targets.forEach((t) => io.observe(t));
})();
