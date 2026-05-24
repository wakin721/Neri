const root = document.documentElement;
const body = document.body;
const themeToggle = document.querySelector("#themeToggle");
const legacyMenuToggle = document.querySelector("#menuToggle");
const mobileMenuBtn = document.querySelector("#mobileMenuBtn");
const closeDrawerBtn = document.querySelector("#closeDrawerBtn");
const mobileDrawer = document.querySelector("#mobileDrawer");
const topAppBar = document.querySelector(".top-app-bar");

function readThemePreference() {
  try {
    return localStorage.getItem("neri-site-theme");
  } catch {
    return null;
  }
}

function writeThemePreference(theme) {
  try {
    localStorage.setItem("neri-site-theme", theme);
  } catch {}
}

const storedTheme = readThemePreference();
const preferredDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
root.dataset.theme = storedTheme || (preferredDark ? "dark" : "light");

function updateBarElevation() {
  if (!topAppBar) return;
  topAppBar.dataset.elevated = window.scrollY > 12 ? "true" : "false";
}

function closeLegacyMenu() {
  body.dataset.menuOpen = "false";
  legacyMenuToggle?.setAttribute("aria-label", "打开导航");
}

function openDrawer() {
  if (!mobileDrawer) return;
  mobileDrawer.dataset.open = "true";
  mobileDrawer.setAttribute("aria-hidden", "false");
  body.dataset.drawerOpen = "true";
}

function closeDrawer() {
  if (!mobileDrawer) return;
  mobileDrawer.dataset.open = "false";
  mobileDrawer.setAttribute("aria-hidden", "true");
  body.dataset.drawerOpen = "false";
}

themeToggle?.addEventListener("click", () => {
  const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
  root.dataset.theme = nextTheme;
  writeThemePreference(nextTheme);
});

legacyMenuToggle?.addEventListener("click", () => {
  const isOpen = body.dataset.menuOpen === "true";
  body.dataset.menuOpen = isOpen ? "false" : "true";
  legacyMenuToggle.setAttribute("aria-label", isOpen ? "打开导航" : "关闭导航");
});

mobileMenuBtn?.addEventListener("click", openDrawer);
closeDrawerBtn?.addEventListener("click", closeDrawer);

mobileDrawer?.addEventListener("click", (event) => {
  if (event.target === mobileDrawer) {
    closeDrawer();
  }
});

document.querySelectorAll(".nav-links a, .mobile-nav-link").forEach((link) => {
  link.addEventListener("click", () => {
    closeLegacyMenu();
    closeDrawer();
  });
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeLegacyMenu();
    closeDrawer();
  }
});

function updateSectionNavigation() {
  const sections = Array.from(document.querySelectorAll(".site-section[id]"));
  if (sections.length === 0) return;

  let current = sections[0].id;
  for (const section of sections) {
    const top = section.getBoundingClientRect().top + window.scrollY;
    if (window.scrollY >= top - 160) {
      current = section.id;
    }
  }

  document.querySelectorAll(".nav-link, .mobile-nav-link").forEach((link) => {
    const href = link.getAttribute("href");
    link.classList.toggle("active", href === `#${current}`);
  });
}

function updateDocNavigation() {
  const docs = Array.from(document.querySelectorAll(".docs-card section[id]"));
  if (docs.length === 0) return;

  let current = docs[0].id;
  for (const section of docs) {
    const top = section.getBoundingClientRect().top + window.scrollY;
    if (window.scrollY >= top - 180) {
      current = section.id;
    }
  }

  document.querySelectorAll(".doc-menu-link").forEach((link) => {
    link.classList.toggle("active", link.getAttribute("href") === `#${current}`);
  });
}

async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return true;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  return copied;
}

function selectCodeText(block) {
  const code = block?.querySelector("code");
  if (!code) return false;
  const selection = window.getSelection();
  const range = document.createRange();
  range.selectNodeContents(code);
  selection?.removeAllRanges();
  selection?.addRange(range);
  return true;
}

document.querySelectorAll(".code-block button").forEach((button) => {
  button.addEventListener("click", async () => {
    const block = button.closest(".code-block");
    const text = block?.dataset.code || block?.querySelector("code")?.textContent || "";
    const original = button.textContent;
    try {
      const copied = await copyText(text);
      button.textContent = copied ? "已复制" : "已选中";
      setTimeout(() => {
        button.textContent = original;
      }, 1600);
    } catch {
      button.textContent = selectCodeText(block) ? "已选中" : "复制失败";
      setTimeout(() => {
        button.textContent = original;
      }, 1600);
    }
  });
});

function syncNavigationSoon() {
  window.requestAnimationFrame(() => {
    scrollCurrentHashIntoView();
    updateSectionNavigation();
    updateDocNavigation();
    window.setTimeout(() => {
      scrollCurrentHashIntoView();
      updateSectionNavigation();
      updateDocNavigation();
    }, 120);
  });
}

function scrollCurrentHashIntoView() {
  if (!window.location.hash) return;
  const id = decodeURIComponent(window.location.hash.slice(1));
  const target = document.getElementById(id);
  target?.scrollIntoView({ block: "start" });
}

window.addEventListener(
  "scroll",
  () => {
    updateBarElevation();
    updateSectionNavigation();
    updateDocNavigation();
  },
  { passive: true },
);

window.addEventListener("hashchange", syncNavigationSoon);
window.addEventListener("load", syncNavigationSoon);

updateBarElevation();
updateSectionNavigation();
updateDocNavigation();
