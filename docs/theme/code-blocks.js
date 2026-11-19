(() => {
  function removeCommonIndentation(codeBlock) {
    const lines = codeBlock.textContent.split("\n").filter((line) => line.trim());
    if (!lines.length) return;

    let indent = lines[0].match(/^[ \t]*/)[0];
    for (const line of lines) {
      while (indent && !line.startsWith(indent)) {
        indent = indent.slice(0, -1);
      }
    }
    if (!indent) return;

    const walker = document.createTreeWalker(codeBlock, NodeFilter.SHOW_TEXT);
    let remainingIndent = indent.length;
    while (walker.nextNode()) {
      const node = walker.currentNode;
      node.nodeValue = node.nodeValue
        .split(/(\n)/)
        .map((part) => {
          if (part === "\n") {
            remainingIndent = indent.length;
            return part;
          }
          const removed = Math.min(remainingIndent, part.match(/^[ \t]*/)[0].length);
          remainingIndent -= removed;
          if (part.length > removed) remainingIndent = 0;
          return part.slice(removed);
        })
        .join("");
    }
  }

  document.querySelectorAll(".content pre > code").forEach(removeCommonIndentation);
})();
