(() => {
  const searchDialog = document.getElementById("guide-search-dialog");
  const searchWrapper = document.getElementById("mdbook-search-wrapper");
  const searchInput = document.getElementById("mdbook-searchbar");
  const searchToggle = document.getElementById("mdbook-search-toggle");
  const searchResults = document.getElementById("mdbook-searchresults");
  if (!searchDialog || !searchWrapper || !searchInput || !searchToggle || !searchResults) return;

  let previousFocus = searchToggle;
  let savedScrollPosition = { x: scrollX, y: scrollY };
  let previousOverflow;

  function syncDialogVisibility() {
    if (!searchWrapper.classList.contains("hidden") && !searchDialog.open) {
      previousOverflow = document.documentElement.style.overflow;
      document.documentElement.style.overflow = "hidden";
      searchDialog.showModal();
      searchInput.focus({ preventScroll: true });
      searchInput.select();
    } else if (searchWrapper.classList.contains("hidden") && searchDialog.open) {
      searchDialog.close();
      document.documentElement.style.overflow = previousOverflow;
      if (previousFocus.isConnected) previousFocus.focus({ preventScroll: true });
      window.scrollTo(savedScrollPosition.x, savedScrollPosition.y);
    }
  }

  function closeSearch() {
    searchWrapper.classList.add("hidden");
    searchToggle.setAttribute("aria-expanded", "false");
    searchResults.querySelectorAll(".focus").forEach((item) => item.classList.remove("focus"));
    const url = new URL(location.href);
    url.searchParams.delete("search");
    history.replaceState(null, "", url);
    syncDialogVisibility();
  }

  function savePageState() {
    if (!searchDialog.open) {
      previousFocus = document.activeElement;
      savedScrollPosition = { x: scrollX, y: scrollY };
    }
  }

  function handleSearchShortcut(event) {
    const editing =
      event.target instanceof Element &&
      event.target.closest("input, textarea, select, [contenteditable]");
    const command = (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k";
    const shortcut =
      !editing &&
      !event.metaKey &&
      !event.ctrlKey &&
      !event.altKey &&
      (event.key === "s" || event.key === "/");
    if (command || shortcut) {
      event.preventDefault();
      event.stopImmediatePropagation();
      if (!searchDialog.open) searchToggle.click();
      else searchInput.focus({ preventScroll: true });
    }
  }

  function handleResultNavigation(event) {
    event.preventDefault();
    const items = [...searchResults.querySelectorAll("li")];
    const currentIndex = items.findIndex((item) => item.classList.contains("focus"));
    if (event.key === "Enter") {
      (items[currentIndex] || items[0])?.querySelector("a").click();
      return;
    }

    items.forEach((item) => item.classList.remove("focus"));
    const nextIndex =
      event.key === "ArrowDown" ? Math.min(currentIndex + 1, items.length - 1) : currentIndex - 1;
    if (nextIndex < 0) searchInput.focus({ preventScroll: true });
    else {
      items[nextIndex].classList.add("focus");
      items[nextIndex].querySelector("a").focus({ preventScroll: true });
      items[nextIndex].scrollIntoView({ block: "nearest" });
    }
  }

  function trapDialogFocus(event) {
    const controls = [...searchDialog.querySelectorAll("input, button, a[href]")];
    const firstControl = controls[0];
    const lastControl = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === firstControl) {
      event.preventDefault();
      lastControl.focus();
    } else if (!event.shiftKey && document.activeElement === lastControl) {
      event.preventDefault();
      firstControl.focus();
    }
  }

  function handleDialogKeyDown(event) {
    event.stopPropagation();
    if (event.key === "Escape") {
      event.preventDefault();
      closeSearch();
    } else if (["ArrowDown", "ArrowUp", "Enter"].includes(event.key)) {
      handleResultNavigation(event);
    } else if (event.key === "Tab") {
      trapDialogFocus(event);
    }
  }

  function handleBackdropClick(event) {
    const bounds = searchDialog.getBoundingClientRect();
    const outsideDialog =
      event.clientX < bounds.left ||
      event.clientX > bounds.right ||
      event.clientY < bounds.top ||
      event.clientY > bounds.bottom;
    if (event.target === searchDialog && outsideDialog) closeSearch();
  }

  searchToggle.addEventListener("click", savePageState, true);
  searchToggle.addEventListener("click", () => {
    window.scrollTo(savedScrollPosition.x, savedScrollPosition.y);
    syncDialogVisibility();
  });
  // mdBook owns search state; its wrapper visibility drives the dialog.
  new MutationObserver(syncDialogVisibility).observe(searchWrapper, {
    attributes: true,
    attributeFilter: ["class"],
  });
  document.addEventListener("keydown", handleSearchShortcut, true);
  searchDialog.addEventListener("keydown", handleDialogKeyDown);
  searchDialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeSearch();
  });
  searchDialog.addEventListener("click", handleBackdropClick);
  syncDialogVisibility();
})();
