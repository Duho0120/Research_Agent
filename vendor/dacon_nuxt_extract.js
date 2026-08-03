// Extracts the leaderboard array embedded in a DACON leaderboard page's
// server-rendered `window.__NUXT__=(function(...){...})(...)` payload.
//
// DACON's SSR page only renders the first 100 rows as <li> HTML elements;
// the full ranking list (used for ranks beyond 100, e.g. via the page's
// own "전체보기" button) is already present in this payload, just
// deduplicated through a minifier-style IIFE with hundreds of positional
// string arguments. That means it cannot be parsed as JSON directly -- it
// has to be evaluated as JavaScript. This script does that in an isolated
// vm context (no filesystem/network access exposed to the evaluated code)
// and prints whichever array looks like the leaderboard as JSON.
//
// Usage: node dacon_nuxt_extract.js < page.html
const vm = require("vm");

function readStdin() {
  const chunks = [];
  process.stdin.on("data", (c) => chunks.push(c));
  return new Promise((resolve) => {
    process.stdin.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
  });
}

function extractNuxtSource(html) {
  const marker = "window.__NUXT__=";
  const start = html.indexOf(marker);
  if (start === -1) return null;
  // The assignment is always the last statement of its <script> tag in
  // DACON's markup, so cutting at the following </script> is reliable.
  const end = html.indexOf("</script>", start);
  if (end === -1) return null;
  return html.slice(start, end);
}

// Matchers for the two payload shapes we look for. "leaderboard" rows are
// keyed by team (ranking + team_name); "submissions" rows are one team's own
// submission history (sub_id + score, no team_name) -- exact submission-row
// field names are inferred from the leaderboard row shape (which reuses
// sub_id/score/c_time per-team) and have not yet been confirmed against a
// real authenticated /mysubmission response with rows in it, so this is
// best-effort until verified against real data.
const MATCHERS = {
  leaderboard: (first) => "ranking" in first && "team_name" in first && "score" in first,
  submissions: (first) => "sub_id" in first && "score" in first && !("team_name" in first),
};

function collectArrays(node, matcher, depth, seen, results) {
  if (!node || typeof node !== "object" || depth > 8) return;
  if (seen.has(node)) return;
  seen.add(node);
  if (Array.isArray(node)) {
    if (node.length > 0 && node[0] && typeof node[0] === "object" && matcher(node[0])) {
      results.push(node);
    }
    for (const item of node) collectArrays(item, matcher, depth + 1, seen, results);
    return;
  }
  for (const key of Object.keys(node)) {
    collectArrays(node[key], matcher, depth + 1, seen, results);
  }
}

function findArray(root, mode) {
  const matcher = MATCHERS[mode];
  const results = [];
  collectArrays(root, matcher, 0, new Set(), results);
  if (results.length === 0) return null;
  // DACON embeds both a capped "result" (top 100, what the page renders by
  // default) and a "result_full" (every row, revealed by the page's own
  // "전체보기" button) in the same payload -- always the same underlying
  // rows, just truncated differently, so the longest one is always the
  // complete, correct list to use.
  return results.reduce((longest, candidate) => (candidate.length > longest.length ? candidate : longest));
}

async function main() {
  const mode = process.argv[2] === "submissions" ? "submissions" : "leaderboard";
  const html = await readStdin();
  const source = extractNuxtSource(html);
  if (!source) {
    process.stderr.write("nuxt_payload_not_found\n");
    process.exit(2);
  }
  const sandbox = { window: {} };
  vm.createContext(sandbox);
  try {
    vm.runInContext(source, sandbox, { timeout: 5000 });
  } catch (err) {
    process.stderr.write("nuxt_payload_eval_error: " + err.message + "\n");
    process.exit(3);
  }
  const rows = findArray(sandbox.window.__NUXT__, mode);
  if (!rows) {
    process.stderr.write(mode + "_array_not_found\n");
    process.exit(4);
  }
  process.stdout.write(JSON.stringify(rows));
}

main();
