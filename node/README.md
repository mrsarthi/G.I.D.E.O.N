# G.I.D.E.O.N Remote Reminder Node

The always-on half. Holds upcoming reminders and sends them on Telegram at the
right moment, so **a reminder still fires when the desktop is shut down**.

It runs on Cloudflare Workers + D1, both inside the free tier, and never sleeps.

## What it is not

- It does **not** poll Telegram. `getUpdates` allows exactly one consumer per
  bot token, and that consumer is `gideon/interfaces/telegram.py` on the desktop. A second
  poller would knock the desktop bot offline with `409 Conflict`. This node
  only ever *sends*.
- It does **not** run a model or hold conversations. Ask it a question and
  nothing happens; it has no idea what a question is.
- It stores **only** reminder text, a due timestamp, and the origin chat id.
  Conversations, vector memory, documents and clipboard never leave the
  machine.

## Delivery ownership

Exactly one side sends any given reminder, tracked by `reminders.remote_synced`
in the desktop's SQLite:

| `remote_synced` | Who sends the Telegram copy |
| :--- | :--- |
| `1` — the node accepted it | **The node** (fires whether or not the PC is on) |
| `0` — the push failed | **The desktop** (unchanged from before) |

So a node outage degrades to the old local-only behaviour rather than to
silence, and nothing is ever delivered twice. When the desktop fires a
reminder itself it `DELETE`s the node's copy so it cannot arrive again later.

## Deploy

```bash
cd node
npm install -g wrangler
wrangler login

# 1. Create the database and paste the printed id into wrangler.toml
wrangler d1 create gideon-reminders

# 2. Apply the schema
wrangler d1 execute gideon-reminders --remote --file=schema.sql

# 3. Secrets (never put these in wrangler.toml)
wrangler secret put TELEGRAM_BOT_TOKEN      # same bot as the desktop
wrangler secret put GIDEON_NODE_TOKEN       # invent a long random string

# 4. Edit wrangler.toml: TELEGRAM_ALLOWED_USERS and GIDEON_TZ_OFFSET_MINUTES
# 5. Ship
wrangler deploy
```

Then point the desktop at it in `.env`:

```ini
GIDEON_REMOTE_SCHEDULER_URL=https://gideon-reminders.<your-subdomain>.workers.dev
GIDEON_REMOTE_SCHEDULER_TOKEN=<the same GIDEON_NODE_TOKEN>
```

Restart the engine. The log will say `Remote scheduler reachable`, and every
new reminder logs `remote=synced`. Existing pending reminders are mirrored by
the retry daemon within five minutes.

## Timezone

The desktop writes **naive local** timestamps (`datetime.isoformat()`), so the
node has to compare in the same frame. Set `GIDEON_TZ_OFFSET_MINUTES` to your
UTC offset in minutes — `330` for IST. Get it wrong and every reminder fires
hours early or late; that is the single most likely thing to misconfigure here.

This does mean DST-observing timezones need the value updated twice a year.
India does not observe DST, so for the default setup it is set once.

## API

All routes require `Authorization: Bearer <GIDEON_NODE_TOKEN>`.

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness plus a count of queued reminders |
| `PUT` | `/reminders/{id}` | Queue or update one (upsert — retries are safe) |
| `DELETE` | `/reminders/{id}` | Withdraw one the desktop already delivered |

## Cost

Cron fires once a minute — ~43,200 invocations a month against a free-tier
limit of 100,000/day, and D1 reads/writes in the low thousands. In practice
this is free.

## If you would rather not use Cloudflare

The contract is three HTTP routes and a once-a-minute loop. A Raspberry Pi
running the same logic against SQLite, or any small VPS, substitutes cleanly —
only `GIDEON_REMOTE_SCHEDULER_URL` changes on the desktop.
