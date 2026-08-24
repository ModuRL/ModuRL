(() => {
  const siteBase = "/ModuRL/";
  const match = window.location.pathname.match(
    /^\/ModuRL\/([^/]+)\/(.*)$/,
  );

  if (!match) {
    return;
  }

  const currentVersion = match[1];
  const currentPage = match[2];

  window.addEventListener("DOMContentLoaded", async () => {
    const controls = document.querySelector(".right-buttons");
    if (!controls) {
      return;
    }

    try {
      const response = await fetch(`${siteBase}versions.json`);
      if (!response.ok) {
        return;
      }

      const { versions } = await response.json();
      if (!Array.isArray(versions) || versions.length < 2) {
        return;
      }

      const select = document.createElement("select");
      select.setAttribute("aria-label", "Documentation version");
      select.title = "Documentation version";
      select.style.margin = "0 0.5rem";
      select.style.padding = "0.25rem";

      for (const version of versions) {
        const option = document.createElement("option");
        option.value = version;
        option.textContent = version === "dev" ? "dev" : `v${version}`;
        option.selected = version === currentVersion;
        select.append(option);
      }

      select.addEventListener("change", async () => {
        let destination = new URL(
          `${siteBase}${select.value}/${currentPage}`,
          window.location.origin,
        );
        destination.search = window.location.search;
        destination.hash = window.location.hash;

        try {
          const target = await fetch(destination, { method: "HEAD" });
          if (!target.ok) {
            destination = new URL(
              `${siteBase}${select.value}/`,
              window.location.origin,
            );
          }
        } catch {
          // Navigate normally when a browser or intermediary rejects HEAD.
        }

        window.location.assign(destination);
      });
      controls.prepend(select);
    } catch {
      // Documentation remains fully usable if the version manifest is absent.
    }
  });
})();
