import gzip, json, aiohttp, asyncio, pathlib, datetime as dt
from typing import AsyncIterator, Dict, List
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
RAW_DIR   = settings.DATA_DIR / "raw/gharchive"
RAW_DIR.mkdir(parents=True, exist_ok=True)

KEEP_EVENT_TYPES: set[str] = {
    "PushEvent", "PullRequestEvent", "IssueCommentEvent",
}

async def stream_hour(year: int, month: int, day: int, hour: int
                      ) -> AsyncIterator[Dict]:
    """
    Yield events from a single GH-Archive `<yyyy>-<mm>-<dd>-<hh>.json.gz`
    file.

    A single hour is ≤ 15 MB → reading it entirely into memory keeps the
    code simple and avoids the tricky byte-chunk / gzip framing bug that
    raised “TypeError: 'int' object is not subscriptable”.
    """
    url = f"https://data.gharchive.org/{year}-{month:02d}-{day:02d}-{hour}.json.gz"

    async with aiohttp.ClientSession() as s, s.get(url) as r:
        if r.status != 200:
            return                        # file not found (still within 1st minute)
        blob = await r.read()             # read whole .gz blob
        try:
            payload = gzip.decompress(blob).splitlines()
        except OSError:
            # corrupt hour (rare) – just skip
            return

    for line in payload:
        try:
            evt = json.loads(line)
            if evt["type"] in KEEP_EVENT_TYPES:
                yield {
                    "type":        evt["type"],
                    "repo":        evt["repo"]["name"],
                    "actor":       evt["actor"]["login"],
                    "created_at":  evt["created_at"],
                }
        except Exception:
            continue


async def collect_recent(days_back: int = 30,
                         out_file: pathlib.Path | None = None) -> None:
    """
    Stream GH-Archive JSON for the last `days_back` days and write only
    Push / PR / IssueComment events to `out_file` as newline-delimited
    JSON.  ~40 k events per day ⇒ fits easily in <3 GB.
    """
    if out_file is None:
        out_file = RAW_DIR / f"events_last_{days_back}d.jsonl"

    cutoff  = dt.datetime.utcnow() - dt.timedelta(days=days_back)
    hours   = int((dt.datetime.utcnow() - cutoff).total_seconds() // 3600)

    print(f"▶ collecting {hours:,} hours ({days_back} days) of GH Archive…")

    processed = 0
    async with aiohttp.ClientSession() as session:
        with open(out_file, "w") as fp:
            # grab at most ~8 files concurrently to stay polite
            sem = asyncio.Semaphore(8)

            async def handle_hour(h_delta: int):
                nonlocal processed
                async with sem:
                    ts = dt.datetime.utcnow() - dt.timedelta(hours=h_delta)
                    async for evt in stream_hour(ts.year, ts.month, ts.day, ts.hour):
                        fp.write(json.dumps(evt) + "\n")
                        processed += 1
                        if processed % 10_000 == 0:
                            print(f"  …{processed:,} events")

            await asyncio.gather(*(handle_hour(h) for h in range(hours)))

    print(f"✅ wrote {processed:,} events → {out_file}")


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    asyncio.run(collect_recent(days_back=days))
