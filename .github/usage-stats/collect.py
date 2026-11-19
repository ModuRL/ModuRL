"""Collect ModuRL adoption metrics and merge them into a history directory.

Sources: crates.io downloads and reverse dependencies, GitHub repository
counters, and (when USAGE_STATS_TOKEN is set) GitHub traffic. GitHub keeps
traffic for 14 days and crates.io keeps daily downloads for 90, so each run
merges the latest window into CSV files that grow over time.
"""

import csv
import json
import os
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

CRATES = ["modurl", "modurl_gym", "modurl_ale", "modurl_mujoco", "modurl_logger"]
REPOS = ["ModuRL/ModuRL", "ModuRL/Modurl-Site"]
USER_AGENT = "modurl-usage-stats (https://github.com/ModuRL/ModuRL)"


def get_json(url, token=None):
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/vnd.github+json"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


def read_csv(path):
    if not path.exists():
        return {}
    with path.open(newline="") as file:
        return {tuple(row[k] for k in key_fields(row)): row for row in csv.DictReader(file)}


def key_fields(row):
    return [k for k in ("date", "name") if k in row]


def merge_csv(path, fields, rows):
    """Upsert rows keyed by their date (and name) columns; newer values win."""
    merged = read_csv(path)
    for row in rows:
        row = {k: str(row[k]) for k in fields}
        merged[tuple(row[k] for k in key_fields(row))] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(merged.values(), key=lambda r: tuple(r[k] for k in key_fields(r))))
    return sorted(merged.values(), key=lambda r: tuple(r[k] for k in key_fields(r)))


def collect_crates(data, today):
    summary = []
    for crate in CRATES:
        try:
            info = get_json(f"https://crates.io/api/v1/crates/{crate}")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue
            raise
        downloads = get_json(f"https://crates.io/api/v1/crates/{crate}/downloads")
        daily = defaultdict(int)
        for entry in downloads["version_downloads"]:
            daily[entry["date"]] += entry["downloads"]
        for entry in downloads["meta"].get("extra_downloads", []):
            daily[entry["date"]] += entry["downloads"]
        # Today's count is still growing; record only finished days.
        history = merge_csv(
            data / "crates" / f"{crate}.csv",
            ["date", "downloads"],
            [{"date": d, "downloads": n} for d, n in daily.items() if d < today.isoformat()],
        )
        reverse = get_json(f"https://crates.io/api/v1/crates/{crate}/reverse_dependencies?per_page=100")
        dependents = sorted({v["crate"] for v in reverse["versions"]} - set(CRATES))
        crate_info = info["crate"]
        summary.append(
            {
                "crate": crate,
                "total": crate_info["downloads"],
                "recent_90d": crate_info.get("recent_downloads") or 0,
                "last_7d": window_sum(history, today, 7),
                "last_30d": window_sum(history, today, 30),
                "latest_version": crate_info.get("max_stable_version") or crate_info["max_version"],
                "external_dependents": dependents,
            }
        )
    return summary


def window_sum(history, today, days):
    start = (today - timedelta(days=days)).isoformat()
    return sum(int(r["downloads"]) for r in history if start <= r["date"] < today.isoformat())


def collect_repos(data, today, token):
    summary = []
    for repo in REPOS:
        slug = repo.split("/")[1].lower()
        info = get_json(f"https://api.github.com/repos/{repo}", os.environ.get("GITHUB_TOKEN"))
        counters = {
            "date": today.isoformat(),
            "stars": info["stargazers_count"],
            "forks": info["forks_count"],
            "watchers": info["subscribers_count"],
            "open_issues": info["open_issues_count"],
        }
        counter_history = merge_csv(data / "github" / f"{slug}-counters.csv", list(counters), [counters])
        repo_summary = {"repo": repo, **counters, "stars_7d_ago": value_days_ago(counter_history, today, 7, "stars")}
        if token:
            try:
                repo_summary.update(collect_traffic(data, today, token, repo, slug))
            except urllib.error.HTTPError as error:
                print(f"warning: cannot read {repo} traffic: HTTP {error.code}", file=sys.stderr)
        summary.append(repo_summary)
    return summary


def value_days_ago(history, today, days, field):
    cutoff = (today - timedelta(days=days)).isoformat()
    older = [r for r in history if r["date"] <= cutoff]
    return int(older[-1][field]) if older else None


def collect_traffic(data, today, token, repo, slug):
    base = f"https://api.github.com/repos/{repo}/traffic"
    result = {}
    for kind in ("views", "clones"):
        payload = get_json(f"{base}/{kind}?per=day", token)
        rows = [
            {"date": e["timestamp"][:10], "count": e["count"], "uniques": e["uniques"]}
            for e in payload[kind]
            if e["timestamp"][:10] < today.isoformat()
        ]
        history = merge_csv(data / "github" / f"{slug}-{kind}.csv", ["date", "count", "uniques"], rows)
        start = (today - timedelta(days=7)).isoformat()
        week = [r for r in history if r["date"] >= start]
        result[f"{kind}_7d"] = sum(int(r["count"]) for r in week)
    # Referrers and paths are 14-day aggregates, so they are stored per snapshot date.
    referrers = get_json(f"{base}/popular/referrers", token)
    merge_csv(
        data / "github" / f"{slug}-referrers.csv",
        ["date", "name", "count", "uniques"],
        [{"date": today.isoformat(), "name": r["referrer"], "count": r["count"], "uniques": r["uniques"]} for r in referrers],
    )
    paths = get_json(f"{base}/popular/paths", token)
    merge_csv(
        data / "github" / f"{slug}-paths.csv",
        ["date", "name", "count", "uniques"],
        [{"date": today.isoformat(), "name": p["path"], "count": p["count"], "uniques": p["uniques"]} for p in paths],
    )
    result["top_referrers_14d"] = [(r["referrer"], r["count"]) for r in referrers[:5]]
    return result


def render_report(crates, repos, today, token_set):
    has_traffic = all("views_7d" in r for r in repos)
    lines = [f"# ModuRL usage — {today.isoformat()}", "", "## crates.io downloads", ""]
    lines += ["| Crate | Total | Last 7 days | Last 30 days | Latest | External dependents |", "|---|---|---|---|---|---|"]
    for c in crates:
        dependents = ", ".join(c["external_dependents"]) or "none"
        lines.append(
            f"| [{c['crate']}](https://crates.io/crates/{c['crate']}) | {c['total']} | {c['last_7d']} | "
            f"{c['last_30d']} | {c['latest_version']} | {dependents} |"
        )
    lines += [
        "",
        "Download counts include CI runs and registry mirrors, so treat them as an upper bound on real users.",
        "",
        "## GitHub",
        "",
    ]
    if has_traffic:
        lines += ["| Repository | Stars (7d change) | Forks | Views 7d | Clones 7d | Top referrers (14d) |", "|---|---|---|---|---|---|"]
    else:
        lines += ["| Repository | Stars (7d change) | Forks | Watchers |", "|---|---|---|---|"]
    for r in repos:
        change = "" if r["stars_7d_ago"] is None else f" ({r['stars'] - r['stars_7d_ago']:+d})"
        stars = f"{r['stars']}{change}"
        link = f"[{r['repo']}](https://github.com/{r['repo']})"
        if has_traffic:
            referrers = ", ".join(f"{name} ({count})" for name, count in r["top_referrers_14d"]) or "none"
            lines.append(f"| {link} | {stars} | {r['forks']} | {r['views_7d']} | {r['clones_7d']} | {referrers} |")
        else:
            lines.append(f"| {link} | {stars} | {r['forks']} | {r['watchers']} |")
    if not has_traffic:
        reason = (
            "`USAGE_STATS_TOKEN` cannot read traffic for every repository"
            if token_set
            else "set the `USAGE_STATS_TOKEN` secret to collect it"
        )
        lines += ["", f"Traffic (views, clones, referrers) is skipped: {reason}."]
    return "\n".join(lines) + "\n"


def main():
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "usage-stats")
    today = datetime.now(timezone.utc).date()
    token = os.environ.get("USAGE_STATS_TOKEN") or None
    crates = collect_crates(data, today)
    repos = collect_repos(data, today, token)
    (data / "latest.json").write_text(
        json.dumps({"date": today.isoformat(), "crates": crates, "repos": repos}, indent=2) + "\n"
    )
    report = render_report(crates, repos, today, token is not None)
    (data / "README.md").write_text(report)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as summary:
            summary.write(report)
    print(report)


if __name__ == "__main__":
    main()
