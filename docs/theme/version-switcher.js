(() => {
  const siteBase = "/";
  const pathMatch = window.location.pathname.match(/^\/([^/]+)\/(.*)$/);
  if (!pathMatch) return;

  const [, currentVersion, currentPage] = pathMatch;

  async function fetchVersions() {
    const response = await fetch(`${siteBase}versions.json`);
    if (!response.ok) return [];

    const manifest = await response.json();
    if (!Array.isArray(manifest.versions)) return [];

    return [
      ...new Set(
        manifest.versions.filter((version) => typeof version === "string" && version.length > 0),
      ),
    ];
  }

  async function navigateToVersion(version) {
    if (version === currentVersion) return;

    let destination = new URL(`${siteBase}${version}/${currentPage}`, window.location.origin);
    destination.search = window.location.search;
    destination.hash = window.location.hash;

    try {
      const target = await fetch(destination, { method: "HEAD" });
      if (!target.ok) {
        destination = new URL(`${siteBase}${version}/`, window.location.origin);
      }
    } catch {
      // Navigate normally when a browser or intermediary rejects HEAD.
    }

    window.location.assign(destination);
  }

  function initializeVersionMenu(versions, versionSwitcher, versionButton, versionMenu) {
    function closeMenu(restoreFocus = false) {
      versionMenu.hidden = true;
      versionButton.setAttribute("aria-expanded", "false");
      if (restoreFocus) versionButton.focus();
    }

    const versionItems = versions.map((version) => {
      const item = document.createElement("button");
      item.type = "button";
      item.setAttribute("role", "menuitemradio");
      item.setAttribute("aria-checked", String(version === currentVersion));
      item.tabIndex = -1;
      item.textContent = version === "dev" ? "dev" : `v${version}`;
      item.addEventListener("click", () => {
        closeMenu(true);
        navigateToVersion(version);
      });
      versionMenu.append(item);
      return item;
    });
    const currentIndex = Math.max(0, versions.indexOf(currentVersion));

    function openMenu(index = currentIndex) {
      versionMenu.hidden = false;
      versionButton.setAttribute("aria-expanded", "true");
      versionItems[index].focus();
    }

    function handleButtonKeyDown(event) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        event.stopPropagation();
        openMenu(event.key === "ArrowDown" ? 0 : versionItems.length - 1);
      } else if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        event.stopPropagation();
        versionMenu.hidden ? openMenu() : closeMenu(true);
      }
    }

    function handleMenuKeyDown(event) {
      const index = versionItems.indexOf(document.activeElement);
      if (
        ["Enter", " ", "ArrowDown", "ArrowUp", "Home", "End", "Escape", "Tab"].includes(event.key)
      ) {
        event.stopPropagation();
      }
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        versionItems[index]?.click();
      } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const step = event.key === "ArrowDown" ? 1 : -1;
        versionItems[(index + step + versionItems.length) % versionItems.length].focus();
      } else if (event.key === "Home" || event.key === "End") {
        event.preventDefault();
        versionItems[event.key === "Home" ? 0 : versionItems.length - 1].focus();
      } else if (event.key === "Escape" || event.key === "Tab") {
        if (event.key === "Escape") event.preventDefault();
        closeMenu(true);
      }
    }

    versionButton.addEventListener("click", () => (versionMenu.hidden ? openMenu() : closeMenu()));
    versionButton.addEventListener("keydown", handleButtonKeyDown);
    versionMenu.addEventListener("keydown", handleMenuKeyDown);
    document.addEventListener("click", (event) => {
      if (!versionSwitcher.contains(event.target)) closeMenu();
    });
    versionSwitcher.addEventListener("focusout", (event) => {
      if (!versionSwitcher.contains(event.relatedTarget)) closeMenu();
    });
    versionButton.disabled = false;
    versionButton.removeAttribute("title");
  }

  async function initializeVersionSwitcher() {
    const versionSwitcher = document.querySelector(".right-buttons .version-switcher");
    const versionButton = versionSwitcher?.querySelector(".version-toggle");
    const versionMenu = versionSwitcher?.querySelector(".version-menu");
    if (!versionSwitcher || !versionButton || !versionMenu) return;

    try {
      const versions = await fetchVersions();
      if (versions.length < 2) return;
      initializeVersionMenu(versions, versionSwitcher, versionButton, versionMenu);
    } catch {
      // Keep the pre-rendered current version visible when the manifest is unavailable.
    }
  }

  window.addEventListener("DOMContentLoaded", initializeVersionSwitcher);
})();
